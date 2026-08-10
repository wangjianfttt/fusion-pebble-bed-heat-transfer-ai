#!/usr/bin/env python3
"""Compare matched CPU and GPU updates on the full P418 regional graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_EQUAL = (
    "nodes",
    "edges",
    "time_points",
    "spatial_temporal_mode",
    "model_parameter_count",
    "physical_parameter_ids",
    "new_physical_parameters",
)


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def calculation_passed(result: dict[str, object]) -> bool:
    return bool(
        result.get("loss_finite")
        and result.get("all_gradients_present")
        and result.get("all_gradients_finite")
        and float(result.get("initial_maximum_absolute_error", 1.0)) == 0.0
        and float(result.get("hydrodynamic_maximum_absolute_error", 1.0)) == 0.0
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-summary", type=Path, required=True)
    parser.add_argument("--gpu-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cpu = load(args.cpu_summary.resolve())
    gpu = load(args.gpu_summary.resolve())
    differences = {
        key: (cpu.get(key), gpu.get(key))
        for key in REQUIRED_EQUAL
        if cpu.get(key) != gpu.get(key)
    }
    if differences:
        raise ValueError(f"CPU and GPU calculations differ: {differences}")
    if not calculation_passed(cpu) or not calculation_passed(gpu):
        raise ValueError("CPU or GPU calculation failed the field/gradient checks")
    if cpu.get("peak_gpu_GB") is not None or not float(gpu.get("peak_gpu_GB", 0.0)) > 0.0:
        raise ValueError("the supplied summaries are not a CPU/GPU pair")
    cpu_seconds = float(cpu["elapsed_seconds"])
    gpu_seconds = float(gpu["elapsed_seconds"])
    if cpu_seconds <= 0.0 or gpu_seconds <= 0.0:
        raise ValueError("elapsed times must be positive")
    result = {
        "status": "completed_hccb_p418_full_graph_cpu_gpu_update_comparison",
        "nodes": cpu["nodes"],
        "edges": cpu["edges"],
        "time_points": cpu["time_points"],
        "spatial_temporal_mode": cpu["spatial_temporal_mode"],
        "model_parameter_count": cpu["model_parameter_count"],
        "cpu_threads": cpu["torch_num_threads"],
        "cpu_elapsed_seconds": cpu_seconds,
        "gpu_elapsed_seconds": gpu_seconds,
        "gpu_peak_memory_GB": float(gpu["peak_gpu_GB"]),
        "gpu_update_speedup": cpu_seconds / gpu_seconds,
        "initial_state_and_hydrodynamics_exact": True,
        "both_backward_passes_finite": True,
        "new_physical_parameters": [],
        "scientific_scope": (
            "One matched forward, transient energy/flux evaluation and backward update on a "
            "repeated steady regional state. This measures computing cost, not prediction accuracy."
        ),
        "source_summaries": {
            "cpu": str(args.cpu_summary.resolve()),
            "gpu": str(args.gpu_summary.resolve()),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
