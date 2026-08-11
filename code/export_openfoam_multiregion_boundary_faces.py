#!/usr/bin/env python3
"""Export native OpenFOAM boundary faces for multiregion finite-volume ML.

The exporter preserves native face ownership, patch metadata, face centroids
and outward area vectors. A generic ``patch`` is deliberately left with an
``unresolved`` physical role unless an explicit, sourced role manifest is
provided; patch names are never used to guess inlet or outlet semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

import export_openfoam_multiregion_interface_pairs as foam


ROOT = Path(__file__).resolve().parents[1]

ROLE_CODES = {
    "unresolved": 0,
    "wall": 1,
    "symmetry": 2,
    "empty": 3,
    "cyclic": 4,
    "processor": 5,
    "mapped_interface": 6,
    "mapped_coupled_unpaired": 7,
    "wedge": 8,
    "inlet": 9,
    "outlet": 10,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_mesh_counts(owner_path: Path) -> dict[str, int]:
    text = owner_path.read_text(encoding="utf-8", errors="strict")
    note = re.search(
        r'nPoints:(\d+)\s+nCells:(\d+)\s+nFaces:(\d+)\s+nInternalFaces:(\d+)', text
    )
    if not note:
        raise ValueError(f"owner header does not expose native mesh counts: {owner_path}")
    n_points, n_cells, n_faces, n_internal = (int(value) for value in note.groups())
    return {
        "n_points": n_points,
        "n_cells": n_cells,
        "n_faces": n_faces,
        "n_internal_faces": n_internal,
    }


def resolved_case(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_role_manifest(path: Path | None, case: Path) -> tuple[dict[tuple[str, str], dict], str | None]:
    if path is None:
        return {}, None
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "case" in payload and resolved_case(payload["case"]) != case:
        raise RuntimeError("boundary-role manifest belongs to a different OpenFOAM case")
    mappings: dict[tuple[str, str], dict] = {}
    for region, patches in payload.get("regions", {}).items():
        if not isinstance(patches, dict):
            raise ValueError(f"role manifest region {region!r} must contain a patch mapping")
        for patch_name, record in patches.items():
            if not isinstance(record, dict):
                raise ValueError(f"role manifest entry {region}/{patch_name} must be an object")
            role = record.get("role")
            source = record.get("source")
            if role not in ROLE_CODES:
                raise ValueError(f"unsupported physical role {role!r} for {region}/{patch_name}")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"role manifest entry {region}/{patch_name} requires a source")
            mappings[(region, patch_name)] = {"role": role, "source": source.strip()}
    return mappings, sha256(path)


def derived_role(patch_type: str, exact_interface: bool, has_mapping_metadata: bool) -> tuple[str, str]:
    if exact_interface:
        return "mapped_interface", "exact reciprocal interface-pair artifact"
    if patch_type in {"mappedWall", "mappedExtrudedWall"} or has_mapping_metadata:
        return "mapped_coupled_unpaired", "native mapped-patch metadata without exact admitted pair"
    if patch_type == "wall":
        return "wall", "native OpenFOAM patch type"
    if patch_type in {"symmetry", "symmetryPlane"}:
        return "symmetry", "native OpenFOAM patch type"
    if patch_type == "empty":
        return "empty", "native OpenFOAM patch type"
    if patch_type == "cyclic":
        return "cyclic", "native OpenFOAM patch type"
    if patch_type == "processor":
        return "processor", "native OpenFOAM patch type"
    if patch_type == "wedge":
        return "wedge", "native OpenFOAM patch type"
    return "unresolved", "mesh patch type does not determine physical inlet/outlet role"


def load_interface_maps(
    summary_path: Path,
    case: Path,
    fluid_region: str,
    solid_region: str,
) -> tuple[dict[tuple[str, int], int], dict, Path]:
    summary_path = summary_path.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "openfoam_multiregion_interface_pairs_passed":
        raise RuntimeError("interface summary has not passed exact reciprocal pairing")
    if resolved_case(summary["case"]) != case:
        raise RuntimeError("interface summary belongs to a different OpenFOAM case")
    if summary.get("fluid_region") != fluid_region or summary.get("solid_region") != solid_region:
        raise RuntimeError("interface summary region names do not match the requested regions")
    npz_path = summary_path.parent / "interface_face_pairs.npz"
    if not npz_path.is_file() or sha256(npz_path) != summary.get("interface_npz_sha256"):
        raise RuntimeError("interface NPZ is missing or does not match its summary")
    with np.load(npz_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    required = {
        "fluid_face_global",
        "solid_face_global",
        "fluid_owner_cell",
        "solid_owner_cell",
        "face_centroid_m",
        "fluid_area_vector_m2",
        "solid_area_vector_m2",
        "face_area_m2",
    }
    if not required.issubset(arrays):
        raise RuntimeError(f"interface NPZ misses arrays: {sorted(required - set(arrays))}")
    pair_count = len(arrays["fluid_face_global"])
    if pair_count == 0 or any(len(arrays[name]) != pair_count for name in required):
        raise RuntimeError("interface arrays are empty or have inconsistent lengths")
    scale = np.maximum(1.0, np.asarray(arrays["face_area_m2"], dtype=np.float64))[:, None]
    tol = 64.0 * np.finfo(np.float64).eps * scale
    if not np.all(
        np.abs(
            np.asarray(arrays["fluid_area_vector_m2"], dtype=np.float64)
            + np.asarray(arrays["solid_area_vector_m2"], dtype=np.float64)
        )
        <= tol
    ):
        raise RuntimeError("interface area vectors are not reciprocal")
    mapping: dict[tuple[str, int], int] = {}
    for pair_index, (fluid_face, solid_face) in enumerate(
        zip(arrays["fluid_face_global"], arrays["solid_face_global"])
    ):
        fluid_key = (fluid_region, int(fluid_face))
        solid_key = (solid_region, int(solid_face))
        if fluid_key in mapping or solid_key in mapping:
            raise RuntimeError("an interface face appears in more than one pair")
        mapping[fluid_key] = pair_index
        mapping[solid_key] = pair_index
    return mapping, arrays, npz_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--fluid-region", default="fluid")
    parser.add_argument("--solid-region", default="solid")
    parser.add_argument("--interface-summary", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    case = args.case.resolve()
    output = (args.output_dir or (case / "boundary_faces")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    regions = [(args.fluid_region, 0), (args.solid_region, 1)]
    interface_map, interface_arrays, interface_npz_path = load_interface_maps(
        args.interface_summary, case, args.fluid_region, args.solid_region
    )
    explicit_roles, role_manifest_hash = load_role_manifest(args.role_manifest, case)

    meshes = {region: foam.region_mesh(case, region) for region, _ in regions}
    known_patch_keys = {
        (region, patch_name)
        for region, _ in regions
        for patch_name in meshes[region]["boundaries"]
    }
    unknown_role_keys = sorted(set(explicit_roles) - known_patch_keys)
    if unknown_role_keys:
        raise ValueError(f"role manifest names unknown patches: {unknown_role_keys}")
    mesh_counts = {
        region: native_mesh_counts(meshes[region]["mesh"] / "owner") for region, _ in regions
    }
    cell_counts = {region: mesh_counts[region]["n_cells"] for region, _ in regions}
    for region, _ in regions:
        mesh = meshes[region]
        counts = mesh_counts[region]
        neighbours = foam.parse_labels(mesh["mesh"] / "neighbour")
        if len(mesh["float_points"]) != counts["n_points"]:
            raise ValueError(f"point count disagrees with owner header for {region}")
        if len(mesh["faces"]) != counts["n_faces"] or len(mesh["owners"]) != counts["n_faces"]:
            raise ValueError(f"face count disagrees with owner header for {region}")
        if len(neighbours) != counts["n_internal_faces"]:
            raise ValueError(f"internal-face count disagrees with owner header for {region}")
        owner_array = np.asarray(mesh["owners"], dtype=np.int64)
        neighbour_array = np.asarray(neighbours, dtype=np.int64)
        if np.any(owner_array < 0) or np.any(owner_array >= counts["n_cells"]):
            raise ValueError(f"owner label is outside native nCells for {region}")
        if np.any(neighbour_array < 0) or np.any(neighbour_array >= counts["n_cells"]):
            raise ValueError(f"neighbour label is outside native nCells for {region}")
    cell_offsets = {args.fluid_region: 0, args.solid_region: cell_counts[args.fluid_region]}
    patch_types = sorted(
        {
            patch.get("type", "unknown")
            for region, _ in regions
            for patch in meshes[region]["boundaries"].values()
        }
    )
    patch_type_ids = {name: index for index, name in enumerate(patch_types)}

    records: dict[str, list] = {
        "face": [],
        "owner_local": [],
        "owner_global": [],
        "region_type": [],
        "patch_index": [],
        "patch_type_id": [],
        "role_id": [],
        "interface_pair": [],
        "centroid": [],
        "area_vector": [],
        "area": [],
    }
    patch_table: list[dict] = []
    region_checks: dict[str, dict] = {}
    global_patch_index = 0

    for region, region_type in regions:
        mesh = meshes[region]
        neighbours = foam.parse_labels(mesh["mesh"] / "neighbour")
        boundary_start = len(neighbours)
        expected = set(range(boundary_start, len(mesh["faces"])))
        coverage: dict[int, int] = {}
        for patch_name, patch in mesh["boundaries"].items():
            if "startFace" not in patch or "nFaces" not in patch:
                raise ValueError(f"patch {region}/{patch_name} misses startFace or nFaces")
            start = int(patch["startFace"])
            count = int(patch["nFaces"])
            if count < 0 or start < boundary_start or start + count > len(mesh["faces"]):
                raise ValueError(f"patch range is outside boundary faces: {region}/{patch_name}")
            patch_type = patch.get("type", "unknown")
            patch_roles: set[str] = set()
            role_bases: set[str] = set()
            patch_area = 0.0
            paired_count = 0
            for face_index in range(start, start + count):
                coverage[face_index] = coverage.get(face_index, 0) + 1
                owner = int(mesh["owners"][face_index])
                if owner < 0 or owner >= cell_counts[region]:
                    raise ValueError(f"boundary owner is outside cell range: {region} face {face_index}")
                centroid, area_vector, area = foam.face_geometry(
                    mesh["faces"][face_index], mesh["float_points"]
                )
                if not np.isfinite(area) or not area > 0.0 or not np.all(np.isfinite(area_vector)):
                    raise ValueError(f"non-positive or non-finite boundary area: {region} face {face_index}")
                pair_index = interface_map.get((region, face_index), -1)
                if pair_index >= 0:
                    paired_count += 1
                    if region == args.fluid_region:
                        expected_face = int(interface_arrays["fluid_face_global"][pair_index])
                        expected_owner = int(interface_arrays["fluid_owner_cell"][pair_index])
                        expected_area_vector = np.asarray(
                            interface_arrays["fluid_area_vector_m2"][pair_index], dtype=np.float64
                        )
                    else:
                        expected_face = int(interface_arrays["solid_face_global"][pair_index])
                        expected_owner = int(interface_arrays["solid_owner_cell"][pair_index])
                        expected_area_vector = np.asarray(
                            interface_arrays["solid_area_vector_m2"][pair_index], dtype=np.float64
                        )
                    expected_centroid = np.asarray(
                        interface_arrays["face_centroid_m"][pair_index], dtype=np.float64
                    )
                    roundoff = 64.0 * np.finfo(np.float64).eps * max(1.0, area)
                    if face_index != expected_face or owner != expected_owner:
                        raise RuntimeError("interface face or owner disagrees with reciprocal pair artifact")
                    if not np.allclose(
                        area_vector, expected_area_vector, rtol=64.0 * np.finfo(np.float64).eps, atol=roundoff
                    ):
                        raise RuntimeError("interface area vector disagrees with reciprocal pair artifact")
                    if not np.allclose(
                        centroid, expected_centroid, rtol=64.0 * np.finfo(np.float64).eps, atol=roundoff
                    ):
                        raise RuntimeError("interface centroid disagrees with reciprocal pair artifact")
                explicit = explicit_roles.get((region, patch_name))
                if pair_index >= 0:
                    role, role_basis = derived_role(
                        patch_type,
                        exact_interface=True,
                        has_mapping_metadata=True,
                    )
                elif explicit is not None:
                    if patch_type != "patch":
                        raise ValueError(
                            f"explicit inlet/outlet role is admitted only for generic patch type: "
                            f"{region}/{patch_name} is {patch_type}"
                        )
                    role, role_basis = explicit["role"], explicit["source"]
                else:
                    role, role_basis = derived_role(
                        patch_type,
                        exact_interface=pair_index >= 0,
                        has_mapping_metadata=("neighbourPatch" in patch or "neighbourRegion" in patch),
                    )
                patch_roles.add(role)
                role_bases.add(role_basis)
                patch_area += area
                records["face"].append(face_index)
                records["owner_local"].append(owner)
                records["owner_global"].append(owner + cell_offsets[region])
                records["region_type"].append(region_type)
                records["patch_index"].append(global_patch_index)
                records["patch_type_id"].append(patch_type_ids[patch_type])
                records["role_id"].append(ROLE_CODES[role])
                records["interface_pair"].append(pair_index)
                records["centroid"].append(centroid)
                records["area_vector"].append(area_vector)
                records["area"].append(area)
            if len(patch_roles) != 1 or len(role_bases) != 1:
                raise RuntimeError(f"patch {region}/{patch_name} received inconsistent role metadata")
            patch_table.append(
                {
                    "patch_index": global_patch_index,
                    "region": region,
                    "region_type": region_type,
                    "patch_name": patch_name,
                    "patch_type": patch_type,
                    "physical_role": next(iter(patch_roles)) if patch_roles else "unresolved",
                    "role_basis": next(iter(role_bases)) if role_bases else "empty patch",
                    "start_face": start,
                    "face_count": count,
                    "exact_interface_face_count": paired_count,
                    "area_m2": patch_area,
                    "neighbour_region": patch.get("neighbourRegion"),
                    "neighbour_patch": patch.get("neighbourPatch"),
                }
            )
            global_patch_index += 1

        covered = set(coverage)
        region_checks[region] = {
            "boundary_face_count": len(expected),
            "covered_face_count": len(covered),
            "all_boundary_faces_covered_once": covered == expected
            and all(coverage[face] == 1 for face in expected),
            "no_internal_face_is_in_a_boundary_patch": all(face >= boundary_start for face in covered),
        }
        if not region_checks[region]["all_boundary_faces_covered_once"]:
            missing = sorted(expected - covered)[:10]
            duplicate = sorted(face for face, count in coverage.items() if count != 1)[:10]
            raise ValueError(
                f"boundary patches do not cover each face exactly once for {region}; "
                f"missing={missing}, duplicate={duplicate}"
            )

    arrays = {
        "boundary_face_global": np.asarray(records["face"], dtype=np.int64),
        "owner_cell_local": np.asarray(records["owner_local"], dtype=np.int64),
        "owner_cell_global": np.asarray(records["owner_global"], dtype=np.int64),
        "region_type": np.asarray(records["region_type"], dtype=np.int8),
        "patch_index": np.asarray(records["patch_index"], dtype=np.int64),
        "patch_type_id": np.asarray(records["patch_type_id"], dtype=np.int16),
        "physical_role_id": np.asarray(records["role_id"], dtype=np.int8),
        "interface_pair_index": np.asarray(records["interface_pair"], dtype=np.int64),
        "face_centroid_m": np.asarray(records["centroid"], dtype=np.float64),
        "outward_area_vector_m2": np.asarray(records["area_vector"], dtype=np.float64),
        "face_area_m2": np.asarray(records["area"], dtype=np.float64),
    }
    if any(len(array) != len(arrays["boundary_face_global"]) for array in arrays.values()):
        raise RuntimeError("boundary arrays have inconsistent lengths")
    paired_mask = arrays["interface_pair_index"] >= 0
    pair_count = len(interface_arrays["fluid_face_global"])
    pair_hist = np.bincount(arrays["interface_pair_index"][paired_mask], minlength=pair_count)
    all_pairs_present_twice = len(pair_hist) == pair_count and bool(np.all(pair_hist == 2))
    if not all_pairs_present_twice:
        raise RuntimeError("each exact interface pair must appear once on each region boundary")

    npz_path = output / "multiregion_boundary_faces.npz"
    np.savez_compressed(npz_path, **arrays)
    unresolved_patches = [
        f"{item['region']}/{item['patch_name']}"
        for item in patch_table
        if item["physical_role"] == "unresolved"
    ]
    checks = {
        "all_boundary_faces_are_covered_exactly_once": all(
            item["all_boundary_faces_covered_once"] for item in region_checks.values()
        ),
        "no_internal_faces_are_exported_as_boundaries": all(
            item["no_internal_face_is_in_a_boundary_patch"] for item in region_checks.values()
        ),
        "all_owner_indices_are_in_range": bool(
            np.all(arrays["owner_cell_global"] >= 0)
            and np.all(arrays["owner_cell_global"] < sum(cell_counts.values()))
        ),
        "all_boundary_areas_are_positive": bool(np.all(arrays["face_area_m2"] > 0.0)),
        "all_boundary_geometry_is_finite": bool(
            np.all(np.isfinite(arrays["face_centroid_m"]))
            and np.all(np.isfinite(arrays["outward_area_vector_m2"]))
        ),
        "all_exact_interface_pairs_appear_on_both_regions": all_pairs_present_twice,
        "generic_patch_role_is_not_guessed_from_patch_name": all(
            item["physical_role"] == "unresolved"
            for item in patch_table
            if item["patch_type"] == "patch"
            and (item["region"], item["patch_name"]) not in explicit_roles
        ),
    }
    summary = {
        "status": "openfoam_multiregion_boundary_faces_passed"
        if all(checks.values())
        else "openfoam_multiregion_boundary_faces_failed",
        "case": str(case.relative_to(ROOT)) if case.is_relative_to(ROOT) else str(case),
        "fluid_region": args.fluid_region,
        "solid_region": args.solid_region,
        "counts": {
            "fluid_cells": cell_counts[args.fluid_region],
            "solid_cells": cell_counts[args.solid_region],
            "boundary_faces": len(arrays["boundary_face_global"]),
            "patches": len(patch_table),
            "exact_interface_pairs": pair_count,
            "exact_interface_boundary_faces": int(np.count_nonzero(paired_mask)),
            "unresolved_physical_role_patches": len(unresolved_patches),
        },
        "checks": checks,
        "region_coverage": region_checks,
        "patch_type_table": [
            {"patch_type_id": patch_type_ids[name], "patch_type": name} for name in patch_types
        ],
        "physical_role_codes": ROLE_CODES,
        "patch_table": patch_table,
        "unresolved_physical_role_patches": unresolved_patches,
        "interface_summary_sha256": sha256(args.interface_summary.resolve()),
        "interface_npz_sha256": sha256(interface_npz_path),
        "role_manifest_sha256": role_manifest_hash,
        "boundary_npz_sha256": sha256(npz_path),
        "new_fitted_physical_parameters": [],
        "neural_training_allowed": False,
        "claim_boundary": (
            "Native boundary topology and geometry only. Unresolved patch roles require explicit "
            "field dictionaries or a sourced role manifest before boundary-condition residuals are used."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
