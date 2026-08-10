#!/usr/bin/env python3
"""Validate the common state, mass-flow and energy-flow inputs before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hccb_p418_comparison_contract import (
    file_record,
    validate_split_and_statistics,
)


def require_finite(name: str, value: np.ndarray) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")


def require_positive(name: str, value: np.ndarray) -> None:
    require_finite(name, value)
    if np.any(value <= 0.0):
        raise ValueError(f"{name} contains non-positive values")


def require_indices(name: str, value: np.ndarray, upper: int) -> None:
    if value.ndim != 1 or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if len(value) and (int(value.min()) < 0 or int(value.max()) >= upper):
        raise ValueError(f"{name} contains an index outside [0, {upper})")


def validate(
    *,
    state_targets: Path,
    mass_targets: Path,
    energy_targets: Path,
    split_file: Path,
    training_statistics: Path,
    expected_cases: int,
) -> dict[str, object]:
    paths = tuple(
        path.resolve()
        for path in (
            state_targets,
            mass_targets,
            energy_targets,
            split_file,
            training_statistics,
        )
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(paths[0], allow_pickle=False) as loaded:
        condition_id = loaded["condition_id"].astype(str)
        condition = loaded["condition_physical"].astype(np.float64)
        state = loaded["state_physical"].astype(np.float64)
        node_type = loaded["node_type"].astype(np.int64)
        node_volume = loaded["node_volume_m3"].astype(np.float64)
        fluid_global = loaded["fluid_global_region"].astype(np.int64)
        solid_global = loaded["solid_global_region"].astype(np.int64)
    case_count = len(condition_id)
    node_count = len(node_type)
    if case_count != expected_cases:
        raise ValueError(
            f"state targets contain {case_count} conditions, expected {expected_cases}"
        )
    if len(set(condition_id.tolist())) != case_count:
        raise ValueError("state targets repeat condition identifiers")
    if condition.shape != (case_count, 5):
        raise ValueError("condition_physical must have five inputs per condition")
    if state.shape != (case_count, node_count, 5):
        raise ValueError("state_physical shape differs from conditions or regional nodes")
    if node_volume.shape != (node_count,):
        raise ValueError("state node-volume shape differs from node count")
    if not np.all(np.isin(node_type, (0, 1))):
        raise ValueError("state node types must be fluid=0 or solid=1")
    if not np.array_equal(fluid_global, np.flatnonzero(node_type == 0)):
        raise ValueError("state fluid-node order differs from node_type")
    if not np.array_equal(solid_global, np.flatnonzero(node_type == 1)):
        raise ValueError("state solid-node order differs from node_type")
    require_positive("state node volumes", node_volume)
    require_finite("physical conditions", condition)
    require_finite("state targets", state)

    with np.load(paths[1], allow_pickle=False) as loaded:
        mass_id = loaded["condition_id"].astype(str)
        mass_fluid_global = loaded["fluid_global_region"].astype(np.int64)
        mass_owner = loaded["internal_owner"].astype(np.int64)
        mass_neighbour = loaded["internal_neighbour"].astype(np.int64)
        mass_boundary_owner = loaded["boundary_owner"].astype(np.int64)
        mass_boundary_patch = loaded["boundary_patch"].astype(np.int64)
        mass_internal_area = loaded["internal_face_area_m2"].astype(np.float64)
        mass_boundary_area = loaded["boundary_face_area_m2"].astype(np.float64)
        internal_mass = loaded["internal_mass_flow_kg_s"].astype(np.float64)
        boundary_mass = loaded["boundary_mass_flow_kg_s"].astype(np.float64)
    if not np.array_equal(condition_id, mass_id):
        raise ValueError("state and mass target condition orders differ")
    if not np.array_equal(fluid_global, mass_fluid_global):
        raise ValueError("state and mass targets use different fluid-node orders")
    fluid_count = len(fluid_global)
    require_indices("mass internal owners", mass_owner, fluid_count)
    require_indices("mass internal neighbours", mass_neighbour, fluid_count)
    require_indices("mass boundary owners", mass_boundary_owner, fluid_count)
    if mass_owner.shape != mass_neighbour.shape or mass_owner.shape != mass_internal_area.shape:
        raise ValueError("mass internal-face geometry arrays have different lengths")
    if mass_boundary_owner.shape != mass_boundary_patch.shape or mass_boundary_owner.shape != mass_boundary_area.shape:
        raise ValueError("mass boundary-face geometry arrays have different lengths")
    if np.any(mass_boundary_patch < 0):
        raise ValueError("mass boundary patches contain negative identifiers")
    if internal_mass.shape != (case_count, len(mass_owner)):
        raise ValueError("internal mass-flow targets differ from case or face count")
    if boundary_mass.shape != (case_count, len(mass_boundary_owner)):
        raise ValueError("boundary mass-flow targets differ from case or face count")
    require_positive("mass internal-face areas", mass_internal_area)
    require_positive("mass boundary-face areas", mass_boundary_area)
    require_finite("internal mass-flow targets", internal_mass)
    require_finite("boundary mass-flow targets", boundary_mass)

    with np.load(paths[2], allow_pickle=False) as loaded:
        energy_id = loaded["condition_id"].astype(str)
        energy_node_type = loaded["node_type"].astype(np.int64)
        energy_node_volume = loaded["node_volume_m3"].astype(np.float64)
        energy_owner = loaded["internal_owner"].astype(np.int64)
        energy_neighbour = loaded["internal_neighbour"].astype(np.int64)
        energy_kind = loaded["internal_kind"].astype(np.int64)
        energy_kind_name = loaded["internal_kind_name"].astype(str)
        energy_boundary_owner = loaded["boundary_owner"].astype(np.int64)
        energy_boundary_kind = loaded["boundary_kind"].astype(np.int64)
        energy_boundary_kind_name = loaded["boundary_kind_name"].astype(str)
        energy_internal_area = loaded["internal_face_area_m2"].astype(np.float64)
        energy_boundary_area = loaded["boundary_face_area_m2"].astype(np.float64)
        internal_energy = loaded["internal_energy_flow_W"].astype(np.float64)
        boundary_energy = loaded["boundary_energy_flow_W"].astype(np.float64)
        source_power = loaded["node_source_power_W"].astype(np.float64)
    if not np.array_equal(condition_id, energy_id):
        raise ValueError("state and energy target condition orders differ")
    if not np.array_equal(node_type, energy_node_type):
        raise ValueError("state and energy targets use different node orders")
    if not np.array_equal(node_volume, energy_node_volume):
        raise ValueError("state and energy targets use different node volumes")
    require_indices("energy internal owners", energy_owner, node_count)
    require_indices("energy internal neighbours", energy_neighbour, node_count)
    require_indices("energy boundary owners", energy_boundary_owner, node_count)
    if energy_owner.shape != energy_neighbour.shape or energy_owner.shape != energy_kind.shape or energy_owner.shape != energy_internal_area.shape:
        raise ValueError("energy internal-face geometry arrays have different lengths")
    if energy_boundary_owner.shape != energy_boundary_kind.shape or energy_boundary_owner.shape != energy_boundary_area.shape:
        raise ValueError("energy boundary-face geometry arrays have different lengths")
    require_indices("energy internal kinds", energy_kind, len(energy_kind_name))
    require_indices("energy boundary kinds", energy_boundary_kind, len(energy_boundary_kind_name))
    if "fluid_to_solid" not in set(energy_kind_name.tolist()):
        raise ValueError("energy targets do not identify fluid-to-solid faces")
    if "fluid:coolingWall" not in set(energy_boundary_kind_name.tolist()):
        raise ValueError("energy targets do not identify the fluid cooling wall")
    if internal_energy.shape != (case_count, len(energy_owner)):
        raise ValueError("internal energy-flow targets differ from case or face count")
    if boundary_energy.shape != (case_count, len(energy_boundary_owner)):
        raise ValueError("boundary energy-flow targets differ from case or face count")
    if source_power.shape != (case_count, node_count):
        raise ValueError("source-power targets differ from case or node count")
    require_positive("energy internal-face areas", energy_internal_area)
    require_positive("energy boundary-face areas", energy_boundary_area)
    require_finite("internal energy-flow targets", internal_energy)
    require_finite("boundary energy-flow targets", boundary_energy)
    require_finite("node source-power targets", source_power)

    split_payload = json.loads(paths[3].read_text(encoding="utf-8"))
    split_names = list(split_payload.get("splits", {}))
    if not split_names:
        raise ValueError("comparison split file defines no splits")
    split_counts: dict[str, dict[str, int]] = {}
    for split_name in split_names:
        exact, _ = validate_split_and_statistics(
            split_file=paths[3],
            training_statistics=paths[4],
            split_name=split_name,
            condition_ids=condition_id,
        )
        split_counts[split_name] = {
            role: len(exact[role]) for role in ("train", "validation", "test")
        }

    return {
        "status": "p418_steady_comparison_inputs_consistent",
        "condition_count": case_count,
        "regional_node_count": node_count,
        "fluid_node_count": fluid_count,
        "solid_node_count": len(solid_global),
        "mass_internal_face_count": len(mass_owner),
        "mass_boundary_face_count": len(mass_boundary_owner),
        "energy_internal_face_count": len(energy_owner),
        "energy_boundary_face_count": len(energy_boundary_owner),
        "split_case_counts": split_counts,
        "files": {
            "state_targets": file_record(paths[0]),
            "mass_targets": file_record(paths[1]),
            "energy_targets": file_record(paths[2]),
            "split_file": file_record(paths[3]),
            "training_statistics": file_record(paths[4]),
        },
        "new_physical_parameters": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-targets", type=Path, required=True)
    parser.add_argument("--mass-targets", type=Path, required=True)
    parser.add_argument("--energy-targets", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--training-statistics", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = validate(
        state_targets=args.state_targets,
        mass_targets=args.mass_targets,
        energy_targets=args.energy_targets,
        split_file=args.split_file,
        training_statistics=args.training_statistics,
        expected_cases=args.expected_cases,
    )
    if args.output is not None:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
