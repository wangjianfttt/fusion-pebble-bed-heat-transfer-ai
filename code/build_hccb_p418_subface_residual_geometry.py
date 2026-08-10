#!/usr/bin/env python3
"""Build level-reduced HCCB geometry while retaining original FV subfaces.

The neural operator predicts one state per regional cell.  Fluxes are still
evaluated on every original OpenFOAM face that crosses a regional boundary.
This preserves face orientation and area quadrature; faces internal to one
regional cell cancel from its finite-volume balance and are omitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from build_hccb_p418_regional_residual_geometry import (
    composed_parent,
    local_index_map,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preserved_external_boundary(
    *,
    owner: np.ndarray,
    patch: np.ndarray,
    centroid: np.ndarray,
    area_vector: np.ndarray,
    area: np.ndarray,
    parent: np.ndarray,
    local_map: np.ndarray,
    excluded_patch: int,
) -> dict[str, np.ndarray]:
    keep = patch != excluded_patch
    regional_owner = parent[owner[keep]]
    local_owner = local_map[regional_owner]
    if np.any(local_owner < 0):
        raise ValueError("external boundary owner maps to the wrong material region")
    return {
        "owner": local_owner.astype(np.int64),
        "patch": patch[keep].astype(np.int16),
        "centroid_m": centroid[keep].astype(np.float64),
        "area_vector_m2": area_vector[keep].astype(np.float64),
        "area_m2": area[keep].astype(np.float64),
        "openfoam_face": np.flatnonzero(keep).astype(np.int64),
    }


def select_unique_cross_region_subfaces(
    *,
    source: np.ndarray,
    target: np.ndarray,
    kind: np.ndarray,
    centroid: np.ndarray,
    area_vector: np.ndarray,
    area: np.ndarray,
    local_face: np.ndarray,
    parent: np.ndarray,
) -> dict[str, np.ndarray]:
    if not (
        len(source)
        == len(target)
        == len(kind)
        == len(centroid)
        == len(area_vector)
        == len(area)
        == len(local_face)
    ):
        raise ValueError("native graph edge arrays have inconsistent lengths")
    unique_direction = source < target
    owner_region = parent[source]
    neighbour_region = parent[target]
    keep = unique_direction & (owner_region != neighbour_region)
    return {
        "owner_global": owner_region[keep].astype(np.int64),
        "neighbour_global": neighbour_region[keep].astype(np.int64),
        "kind": kind[keep].astype(np.int8),
        "centroid_m": centroid[keep].astype(np.float64),
        "area_vector_m2": area_vector[keep].astype(np.float64),
        "area_m2": area[keep].astype(np.float64),
        "source_global": source[keep].astype(np.int64),
        "target_global": target[keep].astype(np.int64),
        "local_face": local_face[keep].astype(np.int64),
    }


def patch_area_by_id(patch: np.ndarray, area: np.ndarray, patch_count: int) -> np.ndarray:
    return np.bincount(
        patch.astype(np.int64), weights=area.astype(np.float64), minlength=patch_count
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--regional-topology", type=Path, required=True)
    parser.add_argument("--native-graph", type=Path, required=True)
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset_path = args.dataset_index.resolve()
    regional_path = args.regional_topology.resolve()
    native_path = args.native_graph.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_root = dataset_path.parent
    with np.load(dataset_root / dataset["shared_topology_file"], allow_pickle=False) as loaded:
        topology = {name: loaded[name] for name in loaded.files}
    with np.load(regional_path, allow_pickle=False) as loaded:
        regional = {name: loaded[name] for name in loaded.files}
    with np.load(native_path, allow_pickle=False) as loaded:
        native = {name: loaded[name] for name in loaded.files}

    prefix = f"level_{args.level}"
    parent = composed_parent(regional, args.level)
    node_type = regional[f"{prefix}_node_type"].astype(np.int8)
    centroid = regional[f"{prefix}_centroid_m"].astype(np.float64)
    volume = regional[f"{prefix}_volume_m3"].astype(np.float64)
    fluid_local = local_index_map(node_type, 0)
    solid_local = local_index_map(node_type, 1)
    fine_fluid_count = len(topology["fluid_cell_volume_m3"])
    fine_cell_count = fine_fluid_count + len(topology["solid_cell_volume_m3"])

    node_global = native["node_global_index"].astype(np.int64)
    if len(node_global) != fine_cell_count or not np.array_equal(
        node_global, np.arange(fine_cell_count, dtype=np.int64)
    ):
        raise ValueError("native graph node order differs from the shared/regional topology")

    subface = select_unique_cross_region_subfaces(
        source=native["edge_source_global"].astype(np.int64),
        target=native["edge_target_global"].astype(np.int64),
        kind=native["edge_kind"].astype(np.int8),
        centroid=native["edge_face_centroid_m"].astype(np.float64),
        area_vector=native["edge_area_vector_m2"].astype(np.float64),
        area=native["edge_area_m2"].astype(np.float64),
        local_face=native["edge_local_face"].astype(np.int64),
        parent=parent,
    )

    fluid_patch_names = list(dataset["boundary_patch_names"]["fluid"])
    solid_patch_names = list(dataset["boundary_patch_names"]["solid"])
    fluid_interface_patch = fluid_patch_names.index("fluid_to_solid")
    solid_interface_patch = solid_patch_names.index("solid_to_fluid")
    fluid_external = preserved_external_boundary(
        owner=topology["fluid_boundary_face_owner"],
        patch=topology["fluid_boundary_face_patch"],
        centroid=topology["fluid_boundary_face_centroid_m"],
        area_vector=topology["fluid_boundary_face_area_vector_outward_m2"],
        area=topology["fluid_boundary_face_area_m2"],
        parent=parent[:fine_fluid_count],
        local_map=fluid_local,
        excluded_patch=fluid_interface_patch,
    )
    solid_external = preserved_external_boundary(
        owner=topology["solid_boundary_face_owner"],
        patch=topology["solid_boundary_face_patch"],
        centroid=topology["solid_boundary_face_centroid_m"],
        area_vector=topology["solid_boundary_face_area_vector_outward_m2"],
        area=topology["solid_boundary_face_area_m2"],
        parent=parent[fine_fluid_count:],
        local_map=solid_local,
        excluded_patch=solid_interface_patch,
    )

    arrays: dict[str, np.ndarray] = {
        "fine_to_regional_global": parent,
        "fluid_global_region": np.flatnonzero(node_type == 0).astype(np.int64),
        "solid_global_region": np.flatnonzero(node_type == 1).astype(np.int64),
        "fluid_cell_centroid_m": centroid[node_type == 0],
        "fluid_cell_volume_m3": volume[node_type == 0],
        "solid_cell_centroid_m": centroid[node_type == 1],
        "solid_cell_volume_m3": volume[node_type == 1],
    }
    for name, edge_kind, mapping, fine_offset in (
        ("fluid", 0, fluid_local, 0),
        ("solid", 1, solid_local, fine_fluid_count),
    ):
        keep = subface["kind"] == edge_kind
        owner = mapping[subface["owner_global"][keep]]
        neighbour = mapping[subface["neighbour_global"][keep]]
        if np.any(owner < 0) or np.any(neighbour < 0):
            raise ValueError(f"{name} subface maps across material regions")
        arrays[f"{name}_internal_subface_owner"] = owner
        arrays[f"{name}_internal_subface_neighbour"] = neighbour
        arrays[f"{name}_internal_subface_centroid_m"] = subface["centroid_m"][keep]
        arrays[f"{name}_internal_subface_area_vector_m2"] = subface["area_vector_m2"][keep]
        arrays[f"{name}_internal_subface_area_m2"] = subface["area_m2"][keep]
        openfoam_face = subface["local_face"][keep]
        source_local = subface["source_global"][keep] - fine_offset
        openfoam_owner = topology[f"{name}_internal_face_owner"][openfoam_face]
        openfoam_neighbour = topology[f"{name}_internal_face_neighbour"][openfoam_face]
        source_is_owner = source_local == openfoam_owner
        source_is_neighbour = source_local == openfoam_neighbour
        if not np.all(source_is_owner | source_is_neighbour):
            raise ValueError(f"{name} subface cannot be mapped to OpenFOAM owner/neighbour")
        arrays[f"{name}_internal_subface_openfoam_face"] = openfoam_face
        arrays[f"{name}_internal_subface_phi_orientation"] = np.where(
            source_is_owner, 1.0, -1.0
        ).astype(np.float64)

    interface_keep = subface["kind"] == 2
    interface_source = subface["source_global"][interface_keep]
    interface_target = subface["target_global"][interface_keep]
    interface_vector = subface["area_vector_m2"][interface_keep].copy()
    source_is_fluid = interface_source < fine_fluid_count
    fluid_fine = np.where(source_is_fluid, interface_source, interface_target)
    solid_fine = np.where(source_is_fluid, interface_target, interface_source)
    interface_vector[~source_is_fluid] *= -1.0
    fluid_interface_owner = fluid_local[parent[fluid_fine]]
    solid_interface_owner = solid_local[parent[solid_fine]]
    if np.any(fluid_interface_owner < 0) or np.any(solid_interface_owner < 0):
        raise ValueError("interface subface owner maps to the wrong material")

    fluid_external_count = len(fluid_external["owner"])
    solid_external_count = len(solid_external["owner"])
    interface_count = int(np.count_nonzero(interface_keep))
    interface_centroid = subface["centroid_m"][interface_keep]
    interface_area = subface["area_m2"][interface_keep]
    arrays.update(
        {
            "fluid_boundary_face_owner": np.concatenate((fluid_external["owner"], fluid_interface_owner)),
            "fluid_boundary_face_patch": np.concatenate((fluid_external["patch"], np.full(interface_count, fluid_interface_patch, dtype=np.int16))),
            "fluid_boundary_face_centroid_m": np.concatenate((fluid_external["centroid_m"], interface_centroid), axis=0),
            "fluid_boundary_face_area_vector_m2": np.concatenate((fluid_external["area_vector_m2"], interface_vector), axis=0),
            "fluid_boundary_face_area_m2": np.concatenate((fluid_external["area_m2"], interface_area)),
            "fluid_boundary_openfoam_face": np.concatenate(
                (
                    fluid_external["openfoam_face"],
                    np.full(interface_count, -1, dtype=np.int64),
                )
            ),
            "solid_boundary_face_owner": np.concatenate((solid_external["owner"], solid_interface_owner)),
            "solid_boundary_face_patch": np.concatenate((solid_external["patch"], np.full(interface_count, solid_interface_patch, dtype=np.int16))),
            "solid_boundary_face_centroid_m": np.concatenate((solid_external["centroid_m"], interface_centroid), axis=0),
            "solid_boundary_face_area_vector_m2": np.concatenate((solid_external["area_vector_m2"], -interface_vector), axis=0),
            "solid_boundary_face_area_m2": np.concatenate((solid_external["area_m2"], interface_area)),
            "solid_boundary_openfoam_face": np.concatenate(
                (
                    solid_external["openfoam_face"],
                    np.full(interface_count, -1, dtype=np.int64),
                )
            ),
            "interface_fluid_boundary_face": np.arange(fluid_external_count, fluid_external_count + interface_count, dtype=np.int64),
            "interface_solid_boundary_face": np.arange(solid_external_count, solid_external_count + interface_count, dtype=np.int64),
        }
    )

    internal_area_ratios = {}
    for name in ("fluid", "solid"):
        vector = arrays[f"{name}_internal_subface_area_vector_m2"]
        area = arrays[f"{name}_internal_subface_area_m2"]
        ratio = np.linalg.norm(vector, axis=1) / np.maximum(area, np.finfo(np.float64).tiny)
        internal_area_ratios[name] = {
            "minimum": float(np.min(ratio)),
            "maximum": float(np.max(ratio)),
            "p01_abs_error_from_one": float(np.quantile(np.abs(ratio - 1.0), 0.99)),
        }

    original_fluid_external = topology["fluid_boundary_face_patch"] != fluid_interface_patch
    original_solid_external = topology["solid_boundary_face_patch"] != solid_interface_patch
    checks = {
        "fluid_volume_is_preserved": bool(np.isclose(arrays["fluid_cell_volume_m3"].sum(), topology["fluid_cell_volume_m3"].sum(), rtol=1.0e-12)),
        "solid_volume_is_preserved": bool(np.isclose(arrays["solid_cell_volume_m3"].sum(), topology["solid_cell_volume_m3"].sum(), rtol=1.0e-12)),
        "interface_vectors_are_reciprocal": bool(np.array_equal(arrays["fluid_boundary_face_area_vector_m2"][arrays["interface_fluid_boundary_face"]], -arrays["solid_boundary_face_area_vector_m2"][arrays["interface_solid_boundary_face"]])),
        "interface_area_is_preserved": bool(np.isclose(interface_area.sum(), topology["interface_face_area_m2"].sum(), rtol=1.0e-12)),
        "fluid_external_patch_area_is_preserved": bool(np.allclose(patch_area_by_id(fluid_external["patch"], fluid_external["area_m2"], len(fluid_patch_names)), patch_area_by_id(topology["fluid_boundary_face_patch"][original_fluid_external], topology["fluid_boundary_face_area_m2"][original_fluid_external], len(fluid_patch_names)), rtol=1.0e-12, atol=1.0e-18)),
        "solid_external_patch_area_is_preserved": bool(np.allclose(patch_area_by_id(solid_external["patch"], solid_external["area_m2"], len(solid_patch_names)), patch_area_by_id(topology["solid_boundary_face_patch"][original_solid_external], topology["solid_boundary_face_area_m2"][original_solid_external], len(solid_patch_names)), rtol=1.0e-12, atol=1.0e-18)),
        "all_internal_subface_vectors_match_scalar_area": bool(all(values["p01_abs_error_from_one"] < 1.0e-10 for values in internal_area_ratios.values())),
        "fluid_internal_phi_map_is_complete": bool(
            np.all(arrays["fluid_internal_subface_openfoam_face"] >= 0)
            and np.all(np.abs(arrays["fluid_internal_subface_phi_orientation"]) == 1.0)
        ),
        "fluid_external_phi_map_is_complete": bool(
            np.all(arrays["fluid_boundary_openfoam_face"][:fluid_external_count] >= 0)
            and np.all(arrays["fluid_boundary_openfoam_face"][fluid_external_count:] == -1)
        ),
    }

    geometry_path = output / "subface_residual_geometry.npz"
    np.savez_compressed(geometry_path, **arrays)
    summary = {
        "status": "subface_residual_geometry_ready" if all(checks.values()) else "failed",
        "regional_level": args.level,
        "counts": {
            "fluid_cells": len(arrays["fluid_cell_volume_m3"]),
            "solid_cells": len(arrays["solid_cell_volume_m3"]),
            "fluid_internal_subfaces": len(arrays["fluid_internal_subface_owner"]),
            "solid_internal_subfaces": len(arrays["solid_internal_subface_owner"]),
            "interface_subfaces": interface_count,
            "fluid_external_faces": fluid_external_count,
            "solid_external_faces": solid_external_count,
        },
        "checks": checks,
        "internal_subface_area_vector_checks": internal_area_ratios,
        "fluid_patch_names": fluid_patch_names,
        "solid_patch_names": solid_patch_names,
        "source_dataset_sha256": sha256(dataset_path),
        "source_regional_topology_sha256": sha256(regional_path),
        "source_native_graph_sha256": sha256(native_path),
        "geometry_file": geometry_path.name,
        "geometry_sha256": sha256(geometry_path),
        "method": "regional state with original OpenFOAM subface quadrature",
        "new_physical_parameters": [],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
