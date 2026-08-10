#!/usr/bin/env python3
"""Evaluate regional FV mass/energy residuals on the five P418 pilot cases."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hccb_p418_regional_cht_adapter import (
    assemble_p418_regional_cht_residual,
    conservation_metrics,
    load_p418_subface_geometry,
    volume_average_reference_fields,
)


CONDITION_KEYS = (
    "inlet_velocity_m_s",
    "inlet_temperature_K",
    "solid_heat_source_W_m3",
    "outlet_pressure_Pa",
    "cooling_wall_temperature_K",
)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--subface-geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dataset_path = args.dataset_index.resolve()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    root = dataset_path.parent
    topology = load_npz(root / dataset["shared_topology_file"])
    records = {str(item["condition_id"]): item for item in dataset["conditions"]}
    selected = args.case_id or list(records)
    unknown = sorted(set(selected) - set(records))
    if unknown:
        raise ValueError(f"unknown condition ids: {unknown}")
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    geometry = load_p418_subface_geometry(
        args.subface_geometry.resolve(),
        fluid_patch_names=dataset["boundary_patch_names"]["fluid"],
        solid_patch_names=dataset["boundary_patch_names"]["solid"],
        device=device,
        dtype=dtype,
    )
    fluid_inlet_patch = dataset["boundary_patch_names"]["fluid"].index("inlet")
    inlet_face = topology["fluid_boundary_face_patch"] == fluid_inlet_patch
    generated_volume = float(np.sum(topology["solid_cell_volume_m3"]))

    cases: list[dict[str, object]] = []
    started = time.time()
    with torch.no_grad():
        for case_id in selected:
            record = records[case_id]
            field = load_npz(root / record["field_file"])
            averaged = volume_average_reference_fields(
                geometry=geometry, topology=topology, field=field
            )
            condition = torch.tensor(
                [[float(record[key]) for key in CONDITION_KEYS]],
                device=device,
                dtype=dtype,
            )
            residual = assemble_p418_regional_cht_residual(
                geometry=geometry,
                physical_conditions=condition,
                fluid_velocity_m_s=torch.as_tensor(
                    averaged["fluid_velocity_m_s"], device=device, dtype=dtype
                ).unsqueeze(0),
                fluid_pressure_pa=torch.as_tensor(
                    averaged["fluid_pressure_pa"], device=device, dtype=dtype
                ).unsqueeze(0),
                fluid_temperature_k=torch.as_tensor(
                    averaged["fluid_temperature_k"], device=device, dtype=dtype
                ).unsqueeze(0),
                solid_temperature_k=torch.as_tensor(
                    averaged["solid_temperature_k"], device=device, dtype=dtype
                ).unsqueeze(0),
            )
            inlet_mass = float(
                -np.sum(field["fluid_boundary_face_mass_flow_kg_s"][inlet_face])
            )
            generated_heat = float(record["solid_heat_source_W_m3"]) * generated_volume
            metrics = conservation_metrics(
                residual=residual,
                geometry=geometry,
                inlet_mass_flow_kg_s=inlet_mass,
                generated_heat_w=generated_heat,
            )
            cases.append(
                {
                    "condition_id": case_id,
                    "inlet_mass_flow_kg_s": inlet_mass,
                    "generated_heat_W": generated_heat,
                    "metrics": metrics,
                }
            )
            del field, averaged, residual

    payload = {
        "status": "regional_subface_reference_residual_evaluated",
        "case_count": len(cases),
        "dtype": args.dtype,
        "device": str(device),
        "regional_cells": {
            "fluid": len(geometry.fluid_mesh.cell_volume),
            "solid": len(geometry.solid_mesh.cell_volume),
        },
        "cases": cases,
        "maximum_over_cases": {
            name: max(float(item["metrics"][name]) for item in cases)
            for name in cases[0]["metrics"]
        },
        "elapsed_seconds": time.time() - started,
        "interpretation": (
            "Residuals are evaluated after volume-averaging converged OpenFOAM "
            "fields to level-5 regional states while retaining original crossing "
            "subfaces. They quantify the model-reduction error before neural training."
        ),
        "new_physical_parameters": [],
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
