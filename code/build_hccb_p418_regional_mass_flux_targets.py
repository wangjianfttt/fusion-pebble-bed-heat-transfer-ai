#!/usr/bin/env python3
"""Aggregate solved OpenFOAM face mass fluxes to level-reduced regions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from build_hccb_p418_regional_residual_geometry import composed_parent, local_index_map


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regional_internal_pairs(
    *,
    fine_owner: np.ndarray,
    fine_neighbour: np.ndarray,
    parent: np.ndarray,
    global_to_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    owner_region = parent[fine_owner]
    neighbour_region = parent[fine_neighbour]
    crossing = owner_region != neighbour_region
    lower = np.minimum(owner_region[crossing], neighbour_region[crossing])
    upper = np.maximum(owner_region[crossing], neighbour_region[crossing])
    region_count = len(global_to_local)
    key = lower * region_count + upper
    unique, inverse = np.unique(key, return_inverse=True)
    owner_global = unique // region_count
    neighbour_global = unique % region_count
    owner_local = global_to_local[owner_global]
    neighbour_local = global_to_local[neighbour_global]
    if np.any(owner_local < 0) or np.any(neighbour_local < 0):
        raise ValueError("regional internal mass-flux pair leaves the fluid region")
    orientation = np.where(
        owner_region[crossing] == lower, 1.0, -1.0
    )
    return owner_local, neighbour_local, inverse, crossing, orientation


def regional_boundary_groups(
    *,
    fine_owner: np.ndarray,
    fine_patch: np.ndarray,
    parent: np.ndarray,
    global_to_local: np.ndarray,
    patch_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    owner_local_per_face = global_to_local[parent[fine_owner]]
    if np.any(owner_local_per_face < 0):
        raise ValueError("regional boundary mass-flux owner leaves the fluid region")
    key = owner_local_per_face * patch_count + fine_patch.astype(np.int64)
    unique, inverse = np.unique(key, return_inverse=True)
    return unique // patch_count, unique % patch_count, inverse


def regional_balance(
    *,
    internal_flux: np.ndarray,
    internal_owner: np.ndarray,
    internal_neighbour: np.ndarray,
    boundary_flux: np.ndarray,
    boundary_owner: np.ndarray,
    cell_count: int,
) -> np.ndarray:
    balance = np.zeros(cell_count, dtype=np.float64)
    np.add.at(balance, internal_owner, internal_flux)
    np.add.at(balance, internal_neighbour, -internal_flux)
    np.add.at(balance, boundary_owner, boundary_flux)
    return balance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--regional-topology", type=Path, required=True)
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset_path = args.dataset_index.resolve()
    regional_path = args.regional_topology.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    expected_case_count = int(dataset.get("case_count", len(dataset["conditions"])))
    if expected_case_count <= 0 or expected_case_count != len(dataset["conditions"]):
        raise ValueError("dataset case_count does not match its condition records")
    root = dataset_path.parent
    with np.load(root / dataset["shared_topology_file"], allow_pickle=False) as loaded:
        topology = {name: loaded[name] for name in loaded.files}
    with np.load(regional_path, allow_pickle=False) as loaded:
        regional = {name: loaded[name] for name in loaded.files}

    parent = composed_parent(regional, args.level)
    prefix = f"level_{args.level}"
    node_type = regional[f"{prefix}_node_type"].astype(np.int8)
    fluid_global = np.flatnonzero(node_type == 0).astype(np.int64)
    global_to_fluid = local_index_map(node_type, 0)
    fluid_count = len(topology["fluid_cell_volume_m3"])
    fluid_parent = parent[:fluid_count]
    internal_owner, internal_neighbour, internal_inverse, internal_crossing, orientation = regional_internal_pairs(
        fine_owner=topology["fluid_internal_face_owner"].astype(np.int64),
        fine_neighbour=topology["fluid_internal_face_neighbour"].astype(np.int64),
        parent=fluid_parent,
        global_to_local=global_to_fluid,
    )
    patch_count = len(dataset["boundary_patch_names"]["fluid"])
    boundary_owner, boundary_patch, boundary_inverse = regional_boundary_groups(
        fine_owner=topology["fluid_boundary_face_owner"].astype(np.int64),
        fine_patch=topology["fluid_boundary_face_patch"].astype(np.int64),
        parent=fluid_parent,
        global_to_local=global_to_fluid,
        patch_count=patch_count,
    )
    boundary_area_fine = topology["fluid_boundary_face_area_m2"].astype(np.float64)
    boundary_vector_fine = topology[
        "fluid_boundary_face_area_vector_outward_m2"
    ].astype(np.float64)
    boundary_centroid_fine = topology["fluid_boundary_face_centroid_m"].astype(
        np.float64
    )
    boundary_area = np.bincount(
        boundary_inverse,
        weights=boundary_area_fine,
        minlength=len(boundary_owner),
    )
    boundary_area_vector = np.column_stack(
        [
            np.bincount(
                boundary_inverse,
                weights=boundary_vector_fine[:, axis],
                minlength=len(boundary_owner),
            )
            for axis in range(3)
        ]
    )
    boundary_centroid = np.column_stack(
        [
            np.bincount(
                boundary_inverse,
                weights=boundary_centroid_fine[:, axis] * boundary_area_fine,
                minlength=len(boundary_owner),
            )
            for axis in range(3)
        ]
    ) / boundary_area[:, None]

    level_source = regional[f"{prefix}_edge_source"].astype(np.int64)
    level_target = regional[f"{prefix}_edge_target"].astype(np.int64)
    level_kind = regional[f"{prefix}_edge_kind"].astype(np.int8)
    level_unique_fluid = (level_source < level_target) & (level_kind == 0)
    level_pair_key = (
        level_source[level_unique_fluid] * len(node_type)
        + level_target[level_unique_fluid]
    )
    level_order = np.argsort(level_pair_key)
    expected_key = (
        fluid_global[internal_owner] * len(node_type)
        + fluid_global[internal_neighbour]
    )
    if not np.array_equal(level_pair_key[level_order], expected_key):
        raise ValueError("mass-flux pairs differ from the regional topology edges")
    internal_area_vector = regional[f"{prefix}_edge_area_vector_m2"][
        level_unique_fluid
    ][level_order].astype(np.float64)
    internal_area = regional[f"{prefix}_edge_area_m2"][level_unique_fluid][
        level_order
    ].astype(np.float64)
    internal_centroid = regional[f"{prefix}_edge_centroid_m"][
        level_unique_fluid
    ][level_order].astype(np.float64)

    internal_targets: list[np.ndarray] = []
    boundary_targets: list[np.ndarray] = []
    reports: list[dict[str, float | str]] = []
    inlet_patch = dataset["boundary_patch_names"]["fluid"].index("inlet")
    for record in dataset["conditions"]:
        with np.load(root / record["field_file"], allow_pickle=False) as loaded:
            internal_fine = loaded["fluid_internal_face_mass_flow_kg_s"].astype(np.float64)
            boundary_fine = loaded["fluid_boundary_face_mass_flow_kg_s"].astype(np.float64)
        signed = internal_fine[internal_crossing] * orientation
        internal = np.bincount(
            internal_inverse, weights=signed, minlength=len(internal_owner)
        )
        boundary = np.bincount(
            boundary_inverse, weights=boundary_fine, minlength=len(boundary_owner)
        )
        balance = regional_balance(
            internal_flux=internal,
            internal_owner=internal_owner,
            internal_neighbour=internal_neighbour,
            boundary_flux=boundary,
            boundary_owner=boundary_owner,
            cell_count=len(fluid_global),
        )
        inlet_mass = abs(float(np.sum(boundary[boundary_patch == inlet_patch])))
        internal_targets.append(internal)
        boundary_targets.append(boundary)
        reports.append(
            {
                "condition_id": str(record["condition_id"]),
                "inlet_mass_flow_kg_s": inlet_mass,
                "global_mass_imbalance_over_inlet": float(abs(np.sum(balance)) / inlet_mass),
                "maximum_cell_mass_imbalance_over_inlet": float(np.max(np.abs(balance)) / inlet_mass),
                "local_mass_l1_over_two_inlet": float(np.sum(np.abs(balance)) / (2.0 * inlet_mass)),
            }
        )

    target_path = output / "regional_mass_flux_targets.npz"
    np.savez_compressed(
        target_path,
        condition_id=np.asarray([record["condition_id"] for record in dataset["conditions"]]),
        fluid_global_region=fluid_global,
        internal_owner=internal_owner,
        internal_neighbour=internal_neighbour,
        internal_face_centroid_m=internal_centroid,
        internal_face_area_vector_m2=internal_area_vector,
        internal_face_area_m2=internal_area,
        boundary_owner=boundary_owner,
        boundary_patch=boundary_patch.astype(np.int16),
        boundary_face_centroid_m=boundary_centroid,
        boundary_face_area_vector_m2=boundary_area_vector,
        boundary_face_area_m2=boundary_area,
        internal_mass_flow_kg_s=np.stack(internal_targets),
        boundary_mass_flow_kg_s=np.stack(boundary_targets),
    )
    maximum_l1 = max(float(row["local_mass_l1_over_two_inlet"]) for row in reports)
    checks = {
        "all_dataset_cases_are_present": len(reports) == expected_case_count,
        "all_regional_fluxes_are_finite": bool(
            np.all(np.isfinite(internal_targets)) and np.all(np.isfinite(boundary_targets))
        ),
        "regional_mass_balance_retains_openfoam_conservation": maximum_l1 < 1.0e-6,
    }
    summary = {
        "status": "regional_mass_flux_targets_ready" if all(checks.values()) else "failed",
        "regional_level": args.level,
        "counts": {
            "fluid_cells": len(fluid_global),
            "internal_flux_pairs": len(internal_owner),
            "boundary_flux_groups": len(boundary_owner),
            "cases": len(reports),
        },
        "checks": checks,
        "cases": reports,
        "source_dataset_sha256": sha256(dataset_path),
        "source_regional_topology_sha256": sha256(regional_path),
        "target_file": target_path.name,
        "target_sha256": sha256(target_path),
        "method": "sum solved OpenFOAM face mass fluxes on each regional boundary",
        "new_physical_parameters": [],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
