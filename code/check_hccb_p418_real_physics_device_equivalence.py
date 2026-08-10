#!/usr/bin/env python3
"""Compare the registered P418 regional physics residual on CPU and CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hccb_p418_comparison_contract import sha256_file
from hccb_p418_regional_cht_adapter import load_p418_subface_geometry
from hccb_p418_transient_regional_physics import (
    assemble_p418_transient_regional_residual,
)
from train_hccb_p418_spatiotemporal_regional_operator import (
    PHYSICS_REFERENCE_FIELDS,
)

RELATIVE_LINF_TOLERANCE = 5.0e-5


def sliced_mass_flux(values: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Slice time-dependent fluxes while preserving static flux arrays."""
    if values.ndim >= 2 and values.shape[0] >= stop:
        return values[start:stop]
    return values


def tensor_record(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, object]:
    reference_cpu = reference.detach().cpu()
    candidate_cpu = candidate.detach().cpu()
    difference = candidate_cpu - reference_cpu
    maximum_reference = float(torch.max(torch.abs(reference_cpu)))
    maximum_absolute = float(torch.max(torch.abs(difference)))
    rms = float(torch.sqrt(torch.mean(difference.double() ** 2)))
    try:
        torch.testing.assert_close(reference_cpu, candidate_cpu)
        default_assert_close = True
    except AssertionError:
        default_assert_close = False
    return {
        "shape": list(reference_cpu.shape),
        "maximum_reference_absolute": maximum_reference,
        "maximum_absolute_difference": maximum_absolute,
        "rms_difference": rms,
        "relative_linf_difference": maximum_absolute / max(maximum_reference, 1.0e-30),
        "relative_linf_below_declared_tolerance": (
            maximum_absolute / max(maximum_reference, 1.0e-30)
            <= RELATIVE_LINF_TOLERANCE
        ),
        "torch_default_assert_close": default_assert_close,
        "all_finite": bool(
            torch.isfinite(reference_cpu).all() and torch.isfinite(candidate_cpu).all()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--residual-geometry", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--time-count", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real-data device comparison")
    if args.time_count < 3:
        raise ValueError("at least three consecutive times are required")

    index_path = args.dataset_index.resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    records = {
        str(record["sequence_id"]): record for record in index["sequences"]
    }
    if args.sequence_id not in records:
        raise KeyError(f"unknown sequence {args.sequence_id}")
    record = records[args.sequence_id]
    sequence_path = index_path.parent / str(record["sequence_file"])
    with np.load(sequence_path, allow_pickle=False) as data:
        time_all = data["time_s"].astype(np.float32)
        condition = data["condition_physical"].astype(np.float32)
        state_all = data["state_physical"].astype(np.float32)
        internal_all = data["fluid_internal_mass_flux_kg_s"].astype(np.float32)
        boundary_all = data["fluid_boundary_mass_flux_kg_s"].astype(np.float32)

    start = args.start_index
    stop = start + args.time_count
    if start < 0 or stop > len(time_all):
        raise ValueError(f"time slice [{start}:{stop}] is outside {len(time_all)} points")
    time = time_all[start:stop]
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("selected physical times must be strictly increasing")
    state = state_all[start:stop]
    internal = sliced_mass_flux(internal_all, start, stop)
    boundary = sliced_mass_flux(boundary_all, start, stop)

    patch_names = index["boundary_patch_names"]
    geometry_cpu = load_p418_subface_geometry(
        args.residual_geometry.resolve(),
        fluid_patch_names=patch_names["fluid"],
        solid_patch_names=patch_names["solid"],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    geometry_cuda = load_p418_subface_geometry(
        args.residual_geometry.resolve(),
        fluid_patch_names=patch_names["fluid"],
        solid_patch_names=patch_names["solid"],
        device=torch.device("cuda"),
        dtype=torch.float32,
    )

    def residual(device: torch.device, geometry):
        with torch.no_grad():
            return assemble_p418_transient_regional_residual(
                geometry=geometry,
                step_condition=torch.as_tensor(condition, device=device).unsqueeze(0),
                state_physical=torch.as_tensor(state, device=device).unsqueeze(0),
                time_s=torch.as_tensor(time, device=device).unsqueeze(0),
                fluid_internal_mass_flux_kg_s=torch.as_tensor(
                    internal, device=device
                ).unsqueeze(0),
                fluid_boundary_mass_flux_kg_s=torch.as_tensor(
                    boundary, device=device
                ).unsqueeze(0),
            )

    residual_cpu = residual(torch.device("cpu"), geometry_cpu)
    residual_cuda_1 = residual(torch.device("cuda"), geometry_cuda)
    residual_cuda_2 = residual(torch.device("cuda"), geometry_cuda)

    cpu_cuda = {}
    cuda_repeat = {}
    for name in PHYSICS_REFERENCE_FIELDS:
        cpu_cuda[name] = tensor_record(
            getattr(residual_cpu, name), getattr(residual_cuda_1, name)
        )
        cuda_repeat[name] = tensor_record(
            getattr(residual_cuda_1, name), getattr(residual_cuda_2, name)
        )

    payload = {
        "status": "p418_real_regional_physics_cpu_cuda_comparison_complete",
        "sequence_id": args.sequence_id,
        "time_indices": list(range(start, stop)),
        "time_s": [float(value) for value in time],
        "regional_node_count": int(state.shape[1]),
        "state_shape": list(state.shape),
        "device": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "cpu_vs_cuda": cpu_cuda,
        "cuda_repeat": cuda_repeat,
        "all_cpu_cuda_default_assert_close": all(
            bool(row["torch_default_assert_close"]) for row in cpu_cuda.values()
        ),
        "all_cuda_repeat_default_assert_close": all(
            bool(row["torch_default_assert_close"]) for row in cuda_repeat.values()
        ),
        "all_values_finite": all(
            bool(row["all_finite"])
            for group in (cpu_cuda, cuda_repeat)
            for row in group.values()
        ),
        "declared_relative_linf_tolerance": RELATIVE_LINF_TOLERANCE,
        "all_cpu_cuda_below_declared_tolerance": all(
            bool(row["relative_linf_below_declared_tolerance"])
            for row in cpu_cuda.values()
        ),
        "all_cuda_repeat_below_declared_tolerance": all(
            bool(row["relative_linf_below_declared_tolerance"])
            for row in cuda_repeat.values()
        ),
        "maximum_cpu_cuda_relative_linf_difference": max(
            float(row["relative_linf_difference"]) for row in cpu_cuda.values()
        ),
        "maximum_cuda_repeat_relative_linf_difference": max(
            float(row["relative_linf_difference"]) for row in cuda_repeat.values()
        ),
        "input_sha256": {
            "dataset_index": sha256_file(index_path),
            "sequence": sha256_file(sequence_path),
            "residual_geometry": sha256_file(args.residual_geometry.resolve()),
        },
        "new_physical_parameters": [],
        "scope": (
            "The comparison evaluates the registered regional finite-volume residual "
            "on one formal trajectory and three consecutive retained times. It checks "
            "device consistency but does not replace training or independent prediction."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
