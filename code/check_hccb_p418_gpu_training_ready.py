#!/usr/bin/env python3
"""Stop formal P418 graph training before an accidental CPU fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def assess(available: bool, free_gb: float, measured_peak_gb: float, tensor_ok: bool) -> dict[str, object]:
    ready = bool(available and tensor_ok and free_gb >= measured_peak_gb)
    return {
        "ready": ready,
        "cuda_available": bool(available),
        "cuda_tensor_backward_passed": bool(tensor_ok),
        "free_gpu_memory_GB": float(free_gb),
        "measured_full_graph_peak_memory_GB": float(measured_peak_gb),
        "memory_above_measured_peak_GB": float(free_gb - measured_peak_gb),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measured-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    measured = json.loads(args.measured_summary.resolve().read_text(encoding="utf-8"))
    required = float(measured["peak_gpu_GB"])

    import torch

    available = torch.cuda.is_available()
    free_gb = 0.0
    tensor_ok = False
    device_name = None
    tensor_error = None
    if available:
        try:
            device_name = torch.cuda.get_device_name(0)
            free_gb = float(torch.cuda.mem_get_info(0)[0] / 1.0e9)
            value = torch.ones(16, device="cuda", requires_grad=True)
            value.square().mean().backward()
            tensor_ok = bool(value.grad is not None and torch.isfinite(value.grad).all().item())
        except Exception as error:  # CUDA errors must be written before stopping.
            tensor_error = repr(error)
    result = {
        "status": "hccb_p418_graph_gpu_ready" if available and tensor_ok and free_gb >= required else "hccb_p418_graph_gpu_not_ready",
        **assess(available, free_gb, required, tensor_ok),
        "device_name": device_name,
        "tensor_error": tensor_error,
        "measured_summary": str(args.measured_summary.resolve()),
        "new_physical_parameters": [],
        "interpretation": (
            "Formal graph-Transformer training may start only when CUDA tensor differentiation "
            "works and free memory is not below the measured full-graph peak."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
