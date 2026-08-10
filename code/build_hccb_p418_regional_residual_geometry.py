#!/usr/bin/env python3
"""Build conservative regional FV geometry for P418 mass/energy residuals."""

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


def composed_parent(regional: dict[str, np.ndarray], level: int) -> np.ndarray:
    parent = regional["level_0_parent_from_finer"].astype(np.int64)
    for index in range(1, level + 1):
        parent = regional[f"level_{index}_parent_from_finer"][parent]
    return parent


def aggregate_boundary_faces(
    *,
    owner: np.ndarray,
    patch: np.ndarray,
    centroid: np.ndarray,
    area_vector: np.ndarray,
    area: np.ndarray,
    parent: np.ndarray,
    patch_count: int,
    excluded_patch: int,
) -> dict[str, np.ndarray]:
    keep = patch != excluded_patch
    regional_owner = parent[owner[keep]]
    selected_patch = patch[keep].astype(np.int64)
    selected_area = area[keep].astype(np.float64)
    selected_vector = area_vector[keep].astype(np.float64)
    selected_centroid = centroid[keep].astype(np.float64)
    key = regional_owner * patch_count + selected_patch
    unique, inverse = np.unique(key, return_inverse=True)
    count = len(unique)
    total_area = np.bincount(inverse, weights=selected_area, minlength=count)
    if np.any(total_area <= 0.0):
        raise ValueError("aggregated boundary face has non-positive area")
    vector = np.column_stack(
        [
            np.bincount(
                inverse, weights=selected_vector[:, axis], minlength=count
            )
            for axis in range(3)
        ]
    )
    weighted_centroid = np.column_stack(
        [
            np.bincount(
                inverse,
                weights=selected_centroid[:, axis] * selected_area,
                minlength=count,
            )
            for axis in range(3)
        ]
    ) / total_area[:, None]
    return {
        "owner_global": unique // patch_count,
        "patch": (unique % patch_count).astype(np.int16),
        "centroid_m": weighted_centroid,
        "area_vector_m2": vector,
        "area_m2": total_area,
        "fine_face_count": np.bincount(inverse, minlength=count).astype(np.int64),
    }


def local_index_map(node_type: np.ndarray, selected_type: int) -> np.ndarray:
    mapping = np.full(len(node_type), -1, dtype=np.int64)
    selected = np.flatnonzero(node_type == selected_type)
    mapping[selected] = np.arange(len(selected), dtype=np.int64)
    return mapping


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
    root = dataset_path.parent
    with np.load(root / dataset["shared_topology_file"], allow_pickle=False) as loaded:
        topology = {name: loaded[name] for name in loaded.files}
    with np.load(regional_path, allow_pickle=False) as loaded:
        regional = {name: loaded[name] for name in loaded.files}
    prefix = f"level_{args.level}"
    required = {
        f"{prefix}_centroid_m",
        f"{prefix}_volume_m3",
        f"{prefix}_node_type",
        f"{prefix}_edge_source",
        f"{prefix}_edge_target",
        f"{prefix}_edge_kind",
        f"{prefix}_edge_area_m2",
        f"{prefix}_edge_area_vector_m2",
        f"{prefix}_edge_centroid_m",
    }
    missing = sorted(required - set(regional))
    if missing:
        raise ValueError(f"regional topology misses {missing}")
    parent = composed_parent(regional, args.level)
    fluid_count = len(topology["fluid_cell_volume_m3"])
    node_type = regional[f"{prefix}_node_type"].astype(np.int8)
    centroid = regional[f"{prefix}_centroid_m"].astype(np.float64)
    volume = regional[f"{prefix}_volume_m3"].astype(np.float64)
    fluid_local = local_index_map(node_type, 0)
    solid_local = local_index_map(node_type, 1)
    source = regional[f"{prefix}_edge_source"].astype(np.int64)
    target = regional[f"{prefix}_edge_target"].astype(np.int64)
    kind = regional[f"{prefix}_edge_kind"].astype(np.int8)
    total_edge_area = regional[f"{prefix}_edge_area_m2"].astype(np.float64)
    area_vector = regional[f"{prefix}_edge_area_vector_m2"].astype(np.float64)
    face_centroid = regional[f"{prefix}_edge_centroid_m"].astype(np.float64)
    unique_direction = source < target

    fluid_patch_names = list(dataset["boundary_patch_names"]["fluid"])
    solid_patch_names = list(dataset["boundary_patch_names"]["solid"])
    fluid_interface_patch = fluid_patch_names.index("fluid_to_solid")
    solid_interface_patch = solid_patch_names.index("solid_to_fluid")
    fluid_boundary = aggregate_boundary_faces(
        owner=topology["fluid_boundary_face_owner"],
        patch=topology["fluid_boundary_face_patch"],
        centroid=topology["fluid_boundary_face_centroid_m"],
        area_vector=topology["fluid_boundary_face_area_vector_outward_m2"],
        area=topology["fluid_boundary_face_area_m2"],
        parent=parent[:fluid_count],
        patch_count=len(fluid_patch_names),
        excluded_patch=fluid_interface_patch,
    )
    solid_boundary = aggregate_boundary_faces(
        owner=topology["solid_boundary_face_owner"],
        patch=topology["solid_boundary_face_patch"],
        centroid=topology["solid_boundary_face_centroid_m"],
        area_vector=topology["solid_boundary_face_area_vector_outward_m2"],
        area=topology["solid_boundary_face_area_m2"],
        parent=parent[fluid_count:],
        patch_count=len(solid_patch_names),
        excluded_patch=solid_interface_patch,
    )

    interface_keep = unique_direction & (kind == 2)
    interface_source = source[interface_keep]
    interface_target = target[interface_keep]
    interface_vector = area_vector[interface_keep].copy()
    interface_total_area = total_edge_area[interface_keep]
    interface_centroid = face_centroid[interface_keep]
    source_is_fluid = node_type[interface_source] == 0
    fluid_global = np.where(source_is_fluid, interface_source, interface_target)
    solid_global = np.where(source_is_fluid, interface_target, interface_source)
    interface_vector[~source_is_fluid] *= -1.0
    interface_area = np.linalg.norm(interface_vector, axis=1)
    if np.any(interface_area <= 0.0):
        raise ValueError("regional interface has zero vector area")

    fluid_external_count = len(fluid_boundary["owner_global"])
    solid_external_count = len(solid_boundary["owner_global"])
    arrays: dict[str, np.ndarray] = {
        "fluid_cell_centroid_m": centroid[node_type == 0],
        "fluid_cell_volume_m3": volume[node_type == 0],
        "solid_cell_centroid_m": centroid[node_type == 1],
        "solid_cell_volume_m3": volume[node_type == 1],
    }
    for region_name, region_kind, mapping in (
        ("fluid", 0, fluid_local),
        ("solid", 1, solid_local),
    ):
        keep = unique_direction & (kind == region_kind)
        arrays[f"{region_name}_internal_face_owner"] = mapping[source[keep]]
        arrays[f"{region_name}_internal_face_neighbour"] = mapping[target[keep]]
        arrays[f"{region_name}_internal_face_centroid_m"] = face_centroid[keep]
        arrays[f"{region_name}_internal_face_area_vector_m2"] = area_vector[keep]
        arrays[f"{region_name}_internal_face_total_area_m2"] = total_edge_area[keep]

    arrays.update(
        {
            "fluid_boundary_face_owner": np.concatenate(
                (fluid_local[fluid_boundary["owner_global"]], fluid_local[fluid_global])
            ),
            "fluid_boundary_face_patch": np.concatenate(
                (
                    fluid_boundary["patch"],
                    np.full(len(fluid_global), fluid_interface_patch, dtype=np.int16),
                )
            ),
            "fluid_boundary_face_centroid_m": np.concatenate(
                (fluid_boundary["centroid_m"], interface_centroid), axis=0
            ),
            "fluid_boundary_face_area_vector_m2": np.concatenate(
                (fluid_boundary["area_vector_m2"], interface_vector), axis=0
            ),
            "fluid_boundary_fine_face_count": np.concatenate(
                (
                    fluid_boundary["fine_face_count"],
                    np.ones(len(fluid_global), dtype=np.int64),
                )
            ),
            "solid_boundary_face_owner": np.concatenate(
                (solid_local[solid_boundary["owner_global"]], solid_local[solid_global])
            ),
            "solid_boundary_face_patch": np.concatenate(
                (
                    solid_boundary["patch"],
                    np.full(len(solid_global), solid_interface_patch, dtype=np.int16),
                )
            ),
            "solid_boundary_face_centroid_m": np.concatenate(
                (solid_boundary["centroid_m"], interface_centroid), axis=0
            ),
            "solid_boundary_face_area_vector_m2": np.concatenate(
                (solid_boundary["area_vector_m2"], -interface_vector), axis=0
            ),
            "solid_boundary_fine_face_count": np.concatenate(
                (
                    solid_boundary["fine_face_count"],
                    np.ones(len(solid_global), dtype=np.int64),
                )
            ),
            "interface_fluid_boundary_face": np.arange(
                fluid_external_count,
                fluid_external_count + len(fluid_global),
                dtype=np.int64,
            ),
            "interface_solid_boundary_face": np.arange(
                solid_external_count,
                solid_external_count + len(solid_global),
                dtype=np.int64,
            ),
            "interface_total_area_m2": interface_total_area,
        }
    )
    unique_area_ratio = np.linalg.norm(area_vector[unique_direction], axis=1) / np.maximum(
        total_edge_area[unique_direction], np.finfo(np.float64).tiny
    )
    unique_kind = kind[unique_direction]
    area_ratio_by_kind = {
        str(region_kind): {
            "minimum": float(np.min(unique_area_ratio[unique_kind == region_kind])),
            "p05": float(np.quantile(unique_area_ratio[unique_kind == region_kind], 0.05)),
            "median": float(np.median(unique_area_ratio[unique_kind == region_kind])),
        }
        for region_kind in (0, 1, 2)
        if np.any(unique_kind == region_kind)
    }
    checks = {
        "fluid_and_solid_nodes_are_separate": bool(
            np.all(arrays["fluid_internal_face_owner"] >= 0)
            and np.all(arrays["solid_internal_face_owner"] >= 0)
        ),
        "interface_pair_counts_match": len(arrays["interface_fluid_boundary_face"])
        == len(arrays["interface_solid_boundary_face"])
        > 0,
        "interface_vectors_are_reciprocal": bool(
            np.allclose(
                arrays["fluid_boundary_face_area_vector_m2"][
                    arrays["interface_fluid_boundary_face"]
                ],
                -arrays["solid_boundary_face_area_vector_m2"][
                    arrays["interface_solid_boundary_face"]
                ],
                rtol=0.0,
                atol=1.0e-14,
            )
        ),
        "fluid_volume_is_preserved": bool(
            np.isclose(
                arrays["fluid_cell_volume_m3"].sum(),
                topology["fluid_cell_volume_m3"].sum(),
                rtol=1.0e-12,
            )
        ),
        "solid_volume_is_preserved": bool(
            np.isclose(
                arrays["solid_cell_volume_m3"].sum(),
                topology["solid_cell_volume_m3"].sum(),
                rtol=1.0e-12,
            )
        ),
    }
    geometry_path = output / "regional_residual_geometry.npz"
    np.savez_compressed(geometry_path, **arrays)
    summary = {
        "status": "regional_residual_geometry_ready" if all(checks.values()) else "failed",
        "regional_level": args.level,
        "counts": {
            "fluid_cells": len(arrays["fluid_cell_volume_m3"]),
            "solid_cells": len(arrays["solid_cell_volume_m3"]),
            "fluid_internal_faces": len(arrays["fluid_internal_face_owner"]),
            "solid_internal_faces": len(arrays["solid_internal_face_owner"]),
            "interface_faces": len(arrays["interface_fluid_boundary_face"]),
            "fluid_external_boundary_groups": fluid_external_count,
            "solid_external_boundary_groups": solid_external_count,
        },
        "checks": checks,
        "net_vector_to_total_area_ratio_by_edge_kind": area_ratio_by_kind,
        "fluid_patch_names": fluid_patch_names,
        "solid_patch_names": solid_patch_names,
        "source_dataset_sha256": sha256(dataset_path),
        "source_regional_topology_sha256": sha256(regional_path),
        "geometry_file": geometry_path.name,
        "geometry_sha256": sha256(geometry_path),
        "new_physical_parameters": [],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
