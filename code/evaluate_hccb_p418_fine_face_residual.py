#!/usr/bin/env python3
"""Evaluate source-backed CHT residuals on original P418 OpenFOAM faces."""

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
    load_p418_fine_geometry,
)


CONDITION_KEYS = (
    "inlet_velocity_m_s",
    "inlet_temperature_K",
    "solid_heat_source_W_m3",
    "outlet_pressure_Pa",
    "cooling_wall_temperature_K",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--native-graph", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    args = parser.parse_args()
    dataset_path = args.dataset_index.resolve()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    root = dataset_path.parent
    topology_path = root / dataset["shared_topology_file"]
    with np.load(topology_path, allow_pickle=False) as loaded:
        topology = {name: loaded[name] for name in loaded.files}
    records = {str(item["condition_id"]): item for item in dataset["conditions"]}
    selected = args.case_id or list(records)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    geometry = load_p418_fine_geometry(
        topology_path,
        args.native_graph.resolve(),
        fluid_patch_names=dataset["boundary_patch_names"]["fluid"],
        solid_patch_names=dataset["boundary_patch_names"]["solid"],
        device=torch.device("cpu"),
        dtype=dtype,
    )
    inlet_patch = dataset["boundary_patch_names"]["fluid"].index("inlet")
    inlet_face = topology["fluid_boundary_face_patch"] == inlet_patch
    solid_volume = float(np.sum(topology["solid_cell_volume_m3"]))
    cases: list[dict[str, object]] = []
    started = time.time()
    with torch.no_grad():
        for case_id in selected:
            record = records[case_id]
            with np.load(root / record["field_file"], allow_pickle=False) as loaded:
                field = {name: loaded[name] for name in loaded.files}
            condition = torch.tensor(
                [[float(record[key]) for key in CONDITION_KEYS]], dtype=dtype
            )
            boundary_pressure = field["fluid_boundary_pressure_Pa"].astype(
                np.float64, copy=True
            )
            missing_boundary_pressure = boundary_pressure <= 0.0
            boundary_pressure[missing_boundary_pressure] = field["fluid_pressure_Pa"][
                topology["fluid_boundary_face_owner"][missing_boundary_pressure]
            ]
            residual = assemble_p418_regional_cht_residual(
                geometry=geometry,
                physical_conditions=condition,
                fluid_velocity_m_s=torch.as_tensor(field["fluid_velocity_m_s"], dtype=dtype).unsqueeze(0),
                fluid_pressure_pa=torch.as_tensor(field["fluid_pressure_Pa"], dtype=dtype).unsqueeze(0),
                fluid_temperature_k=torch.as_tensor(field["fluid_temperature_K"], dtype=dtype).unsqueeze(0),
                solid_temperature_k=torch.as_tensor(field["solid_temperature_K"], dtype=dtype).unsqueeze(0),
                fluid_boundary_pressure_pa=torch.as_tensor(boundary_pressure, dtype=dtype).unsqueeze(0),
                fluid_internal_mass_flux_kg_s=torch.as_tensor(field["fluid_internal_face_mass_flow_kg_s"], dtype=dtype).unsqueeze(0),
                fluid_boundary_mass_flux_kg_s=torch.as_tensor(field["fluid_boundary_face_mass_flow_kg_s"], dtype=dtype).unsqueeze(0),
            )
            inlet_mass = float(-np.sum(field["fluid_boundary_face_mass_flow_kg_s"][inlet_face]))
            generated_heat = float(record["solid_heat_source_W_m3"]) * solid_volume
            cases.append(
                {
                    "condition_id": case_id,
                    "metrics": conservation_metrics(
                        residual=residual,
                        geometry=geometry,
                        inlet_mass_flow_kg_s=inlet_mass,
                        generated_heat_w=generated_heat,
                    ),
                }
            )
            del field, residual
    payload = {
        "status": "fine_face_cht_residual_evaluated",
        "dtype": args.dtype,
        "case_count": len(cases),
        "cases": cases,
        "maximum_over_cases": {
            name: max(float(item["metrics"][name]) for item in cases)
            for name in cases[0]["metrics"]
        },
        "elapsed_seconds": time.time() - started,
        "method": "original OpenFOAM cells and faces with solved mass flux override",
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
