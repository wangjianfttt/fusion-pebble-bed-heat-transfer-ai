#!/usr/bin/env python3
"""Volume-average solved P418 fields to regional operator nodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from hccb_p418_regional_cht_adapter import _volume_mean


CONDITION_KEYS = (
    "inlet_velocity_m_s",
    "inlet_temperature_K",
    "solid_heat_source_W_m3",
    "outlet_pressure_Pa",
    "cooling_wall_temperature_K",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--subface-geometry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset_path = args.dataset_index.resolve()
    geometry_path = args.subface_geometry.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    expected_case_count = int(dataset.get("case_count", len(dataset["conditions"])))
    if expected_case_count <= 0 or expected_case_count != len(dataset["conditions"]):
        raise ValueError("dataset case_count does not match its condition records")
    root = dataset_path.parent
    with np.load(root / dataset["shared_topology_file"], allow_pickle=False) as loaded:
        fluid_volume = loaded["fluid_cell_volume_m3"].astype(np.float64)
        solid_volume = loaded["solid_cell_volume_m3"].astype(np.float64)
    with np.load(geometry_path, allow_pickle=False) as loaded:
        parent = loaded["fine_to_regional_global"].astype(np.int64)
        fluid_global = loaded["fluid_global_region"].astype(np.int64)
        solid_global = loaded["solid_global_region"].astype(np.int64)
        regional_volume = np.zeros(len(fluid_global) + len(solid_global), dtype=np.float64)
        regional_volume[fluid_global] = loaded["fluid_cell_volume_m3"]
        regional_volume[solid_global] = loaded["solid_cell_volume_m3"]
    fluid_count = len(fluid_volume)
    fluid_parent = parent[:fluid_count]
    solid_parent = parent[fluid_count:]
    node_type = np.ones(len(regional_volume), dtype=np.int8)
    node_type[fluid_global] = 0

    state_targets: list[np.ndarray] = []
    conditions: list[np.ndarray] = []
    for record in dataset["conditions"]:
        with np.load(root / record["field_file"], allow_pickle=False) as loaded:
            state = np.zeros((len(regional_volume), 5), dtype=np.float64)
            state[fluid_global, :3] = _volume_mean(
                loaded["fluid_velocity_m_s"], fluid_volume, fluid_parent, fluid_global
            )
            state[fluid_global, 3] = _volume_mean(
                loaded["fluid_pressure_Pa"], fluid_volume, fluid_parent, fluid_global
            )
            state[fluid_global, 4] = _volume_mean(
                loaded["fluid_temperature_K"], fluid_volume, fluid_parent, fluid_global
            )
            state[solid_global, 4] = _volume_mean(
                loaded["solid_temperature_K"], solid_volume, solid_parent, solid_global
            )
        state_targets.append(state)
        conditions.append(np.asarray([float(record[key]) for key in CONDITION_KEYS]))

    state_path = output / "regional_state_targets.npz"
    np.savez_compressed(
        state_path,
        condition_id=np.asarray([record["condition_id"] for record in dataset["conditions"]]),
        condition_physical=np.stack(conditions),
        state_physical=np.stack(state_targets),
        node_type=node_type,
        node_volume_m3=regional_volume,
        fluid_global_region=fluid_global,
        solid_global_region=solid_global,
    )
    checks = {
        "all_dataset_cases_are_present": len(state_targets) == expected_case_count,
        "all_state_targets_are_finite": bool(np.all(np.isfinite(state_targets))),
        "regional_volume_is_positive": bool(np.all(regional_volume > 0.0)),
        "fluid_and_solid_nodes_cover_the_level": len(fluid_global) + len(solid_global) == len(regional_volume),
    }
    summary = {
        "status": "regional_state_targets_ready" if all(checks.values()) else "failed",
        "counts": {
            "cases": len(state_targets),
            "regional_nodes": len(regional_volume),
            "fluid_nodes": len(fluid_global),
            "solid_nodes": len(solid_global),
        },
        "checks": checks,
        "source_dataset_sha256": sha256(dataset_path),
        "source_subface_geometry_sha256": sha256(geometry_path),
        "target_file": state_path.name,
        "target_sha256": sha256(state_path),
        "method": "volume-weighted average of solved OpenFOAM cell fields",
        "new_physical_parameters": [],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
