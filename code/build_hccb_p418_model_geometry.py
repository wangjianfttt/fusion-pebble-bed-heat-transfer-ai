#!/usr/bin/env python3
"""Attach published boundary roles to the fixed P418 regional mesh."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--regional-topology", type=Path, required=True)
    parser.add_argument("--boundary-roles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset_path = args.dataset_index.resolve()
    regional_path = args.regional_topology.resolve()
    role_path = args.boundary_roles.resolve()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    roles = json.loads(role_path.read_text(encoding="utf-8"))
    topology_path = dataset_path.parent / dataset["shared_topology_file"]
    topology = load_npz(topology_path)
    regional = load_npz(regional_path)

    role_order = list(roles["role_order"])
    role_index = {name: index for index, name in enumerate(role_order)}
    patch_names = dataset["boundary_patch_names"]
    fluid_count = len(topology["fluid_cell_volume_m3"])
    solid_count = len(topology["solid_cell_volume_m3"])
    node_count = fluid_count + solid_count
    fine_roles = np.zeros((node_count, len(role_order)), dtype=np.float32)
    observed_patch_roles: dict[str, dict[str, str]] = {}
    for region, offset, owner_key, patch_key in (
        ("fluid", 0, "fluid_boundary_face_owner", "fluid_boundary_face_patch"),
        ("solid", fluid_count, "solid_boundary_face_owner", "solid_boundary_face_patch"),
    ):
        names = list(patch_names[region])
        declared = roles["regions"][region]
        missing = sorted(set(names) - set(declared))
        if missing:
            raise ValueError(f"boundary roles are missing {region} patches: {missing}")
        observed_patch_roles[region] = {}
        owners = topology[owner_key].astype(np.int64) + offset
        patches = topology[patch_key].astype(np.int64)
        for patch_id, patch_name in enumerate(names):
            role = declared[patch_name]["role"]
            if role not in role_index:
                raise ValueError(f"unknown boundary role {role}")
            patch_owner = owners[patches == patch_id]
            fine_roles[patch_owner, role_index[role]] = 1.0
            observed_patch_roles[region][patch_name] = role

    fine_volume = np.concatenate(
        (topology["fluid_cell_volume_m3"], topology["solid_cell_volume_m3"])
    ).astype(np.float64)
    arrays: dict[str, np.ndarray] = {
        "fine_boundary_role": fine_roles,
        "boundary_role_names": np.asarray(role_order, dtype="U"),
        "coordinate_center_m": regional["fine_node_centroid_m"].mean(axis=0),
        "coordinate_scale_m": np.ptp(regional["fine_node_centroid_m"], axis=0),
        "volume_scale_m3": np.asarray(np.exp(np.mean(np.log(fine_volume)))),
    }
    current_role = fine_roles.astype(np.float64)
    current_volume = fine_volume
    level = 0
    while f"level_{level}_parent_from_finer" in regional:
        parent = regional[f"level_{level}_parent_from_finer"].astype(np.int64)
        region_count = len(regional[f"level_{level}_node_type"])
        region_volume = regional[f"level_{level}_volume_m3"].astype(np.float64)
        weighted = np.zeros((region_count, len(role_order)), dtype=np.float64)
        np.add.at(weighted, parent, current_role * current_volume[:, None])
        arrays[f"level_{level}_boundary_volume_fraction"] = (
            weighted / region_volume[:, None]
        ).astype(np.float32)
        current_role = arrays[f"level_{level}_boundary_volume_fraction"].astype(np.float64)
        current_volume = region_volume
        level += 1

    if np.any(arrays["coordinate_scale_m"] <= 0.0):
        raise ValueError("mesh must span all three coordinates")
    interface_role = role_index["fluid_solid_interface"]
    fluid_interface = topology["interface_fluid_cell"].astype(np.int64)
    solid_interface = topology["interface_solid_cell"].astype(np.int64) + fluid_count
    checks = {
        "all_boundary_patches_have_declared_roles": True,
        "fluid_interface_cells_are_marked": bool(np.all(fine_roles[fluid_interface, interface_role] == 1.0)),
        "solid_interface_cells_are_marked": bool(np.all(fine_roles[solid_interface, interface_role] == 1.0)),
        "boundary_role_values_are_between_zero_and_one": bool(
            all(
                np.all((value >= 0.0) & (value <= 1.0))
                for name, value in arrays.items()
                if "boundary" in name and value.dtype.kind in "fiu"
            )
        ),
        "all_active_regional_levels_have_boundary_features": level > 0,
        "no_solution_field_is_used": True,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    geometry_path = output / "model_geometry.npz"
    np.savez_compressed(geometry_path, **arrays)
    summary = {
        "status": "p418_regional_model_geometry_ready" if all(checks.values()) else "failed",
        "role_order": role_order,
        "observed_patch_roles": observed_patch_roles,
        "counts": {
            "fine_nodes": node_count,
            "active_regional_levels": level,
            "fine_nodes_on_any_boundary": int(np.count_nonzero(np.any(fine_roles > 0.0, axis=1))),
        },
        "checks": checks,
        "source": {
            "dataset_index_sha256": sha256(dataset_path),
            "shared_topology_sha256": sha256(topology_path),
            "regional_topology_sha256": sha256(regional_path),
            "boundary_roles_sha256": sha256(role_path),
        },
        "model_geometry_file": geometry_path.name,
        "model_geometry_sha256": sha256(geometry_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
