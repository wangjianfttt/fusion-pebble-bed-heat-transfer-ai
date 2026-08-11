#!/usr/bin/env python3
"""Export exact fluid-solid interface face pairs from ASCII OpenFOAM meshes.

Mapped patches are discovered from neighbourRegion/neighbourPatch metadata.
Faces are paired by an exact, order-independent signature of Decimal vertex
coordinates. No geometric distance tolerance or fitted physical parameter is
introduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from decimal import Decimal
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def body_after_list_start(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="strict")
    if re.search(r"\bformat\s+binary\s*;", text):
        raise ValueError(f"binary OpenFOAM object is not admitted: {path}")
    match = re.search(r"\n\s*(\d+)\s*\n\s*\(\s*\n", text)
    if not match:
        raise ValueError(f"cannot locate OpenFOAM list body in {path}")
    return text[match.end() :]


def parse_points(path: Path) -> tuple[list[tuple[Decimal, Decimal, Decimal]], np.ndarray]:
    body = body_after_list_start(path)
    matches = re.findall(rf"\(\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*\)", body)
    decimals = [tuple(Decimal(value).normalize() for value in record) for record in matches]
    if not decimals:
        raise ValueError(f"no points parsed from {path}")
    return decimals, np.asarray([[float(value) for value in record] for record in decimals], dtype=np.float64)


def parse_faces(path: Path) -> list[list[int]]:
    body = body_after_list_start(path)
    faces: list[list[int]] = []
    for count_text, labels_text in re.findall(r"(\d+)\s*\(([^()]*)\)", body):
        labels = [int(value) for value in labels_text.split()]
        if len(labels) != int(count_text):
            raise ValueError(f"face vertex count mismatch in {path}: {count_text} vs {labels}")
        faces.append(labels)
    if not faces:
        raise ValueError(f"no faces parsed from {path}")
    return faces


def parse_labels(path: Path) -> list[int]:
    body = body_after_list_start(path)
    labels = [int(value) for value in re.findall(r"(?m)^\s*(\d+)\s*$", body)]
    if not labels:
        raise ValueError(f"no labels parsed from {path}")
    return labels


def parse_boundary(path: Path) -> dict[str, dict[str, str]]:
    body = body_after_list_start(path)
    patches: dict[str, dict[str, str]] = {}
    pattern = re.compile(r"(?m)^\s*([A-Za-z0-9_.:+-]+)\s*\n\s*\{")
    for match in pattern.finditer(body):
        name = match.group(1)
        start = body.find("{", match.start())
        depth = 0
        end = None
        for index in range(start, len(body)):
            if body[index] == "{":
                depth += 1
            elif body[index] == "}":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end is None:
            raise ValueError(f"unterminated patch dictionary {name} in {path}")
        block = body[start + 1 : end]
        values: dict[str, str] = {}
        for key in ("type", "nFaces", "startFace", "neighbourRegion", "neighbourPatch", "transformType"):
            value = re.search(rf"\b{key}\s+([^;\s]+)\s*;", block)
            if value:
                values[key] = value.group(1)
        patches[name] = values
    if not patches:
        raise ValueError(f"no boundary patches parsed from {path}")
    return patches


def face_signature(face: list[int], points: list[tuple[Decimal, Decimal, Decimal]]) -> tuple:
    return tuple(sorted(points[index] for index in face))


def face_geometry(face: list[int], points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    vertices = points[np.asarray(face, dtype=np.int64)]
    centroid = vertices.mean(axis=0)
    area_vector = np.zeros(3, dtype=np.float64)
    for index, vertex in enumerate(vertices):
        area_vector += np.cross(vertex, vertices[(index + 1) % len(vertices)])
    area_vector *= 0.5
    return centroid, area_vector, float(np.linalg.norm(area_vector))


def region_mesh(case: Path, region: str) -> dict:
    mesh = case / "constant" / region / "polyMesh"
    decimal_points, float_points = parse_points(mesh / "points")
    faces = parse_faces(mesh / "faces")
    owners = parse_labels(mesh / "owner")
    boundaries = parse_boundary(mesh / "boundary")
    if len(owners) != len(faces):
        raise ValueError(f"owner/face count mismatch for {region}: {len(owners)} vs {len(faces)}")
    return {
        "mesh": mesh,
        "decimal_points": decimal_points,
        "float_points": float_points,
        "faces": faces,
        "owners": owners,
        "boundaries": boundaries,
    }


def mapped_pairs(
    fluid: dict,
    solid: dict,
    fluid_region: str,
    solid_region: str,
    admitted_patch_types: set[str],
) -> tuple[list[tuple[str, str]], list[dict]]:
    pairs: list[tuple[str, str]] = []
    excluded: list[dict] = []
    for fluid_name, fluid_patch in fluid["boundaries"].items():
        if fluid_patch.get("neighbourRegion") != solid_region:
            continue
        solid_name = fluid_patch.get("neighbourPatch")
        if not solid_name or solid_name not in solid["boundaries"]:
            raise ValueError(f"fluid patch {fluid_name} points to missing solid patch {solid_name!r}")
        solid_patch = solid["boundaries"][solid_name]
        if solid_patch.get("neighbourRegion") != fluid_region or solid_patch.get("neighbourPatch") != fluid_name:
            raise ValueError(f"non-reciprocal mapped patch metadata: {fluid_name} <-> {solid_name}")
        if fluid_patch.get("type") not in admitted_patch_types or solid_patch.get("type") not in admitted_patch_types:
            excluded.append(
                {
                    "fluid_patch": fluid_name,
                    "solid_patch": solid_name,
                    "fluid_type": fluid_patch.get("type"),
                    "solid_type": solid_patch.get("type"),
                    "reason": "non-coincident patch type outside the declared exact-coordinate pairing contract",
                }
            )
            continue
        pairs.append((fluid_name, solid_name))
    if not pairs:
        raise ValueError(
            f"no reciprocal patches of admitted types {sorted(admitted_patch_types)} "
            f"between {fluid_region} and {solid_region}"
        )
    return pairs, excluded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--fluid-region", default="fluid")
    parser.add_argument("--solid-region", default="solid")
    parser.add_argument("--patch-types", default="mappedWall")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    case = args.case.resolve()
    output = (args.output_dir or (case / "interface_pairs")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fluid = region_mesh(case, args.fluid_region)
    solid = region_mesh(case, args.solid_region)
    admitted_patch_types = {item.strip() for item in args.patch_types.split(",") if item.strip()}
    patch_pairs, excluded_patches = mapped_pairs(
        fluid,
        solid,
        args.fluid_region,
        args.solid_region,
        admitted_patch_types,
    )

    records: dict[str, list] = {
        "fluid_face": [],
        "solid_face": [],
        "fluid_owner": [],
        "solid_owner": [],
        "fluid_patch_index": [],
        "centroid": [],
        "fluid_area_vector": [],
        "solid_area_vector": [],
        "area_magnitude": [],
    }
    patch_metadata: list[dict] = []
    exact_pairing = True
    reciprocal_orientation = True
    unique_signatures = True
    eps = np.finfo(np.float64).eps

    for patch_index, (fluid_name, solid_name) in enumerate(patch_pairs):
        fluid_patch = fluid["boundaries"][fluid_name]
        solid_patch = solid["boundaries"][solid_name]
        fluid_start = int(fluid_patch["startFace"])
        solid_start = int(solid_patch["startFace"])
        fluid_count = int(fluid_patch["nFaces"])
        solid_count = int(solid_patch["nFaces"])
        if fluid_count != solid_count:
            raise ValueError(f"interface face-count mismatch: {fluid_name}={fluid_count}, {solid_name}={solid_count}")

        solid_by_signature: dict[tuple, int] = {}
        for solid_face in range(solid_start, solid_start + solid_count):
            signature = face_signature(solid["faces"][solid_face], solid["decimal_points"])
            if signature in solid_by_signature:
                unique_signatures = False
            solid_by_signature[signature] = solid_face

        paired_solid: set[int] = set()
        patch_area = 0.0
        for fluid_face in range(fluid_start, fluid_start + fluid_count):
            signature = face_signature(fluid["faces"][fluid_face], fluid["decimal_points"])
            solid_face = solid_by_signature.get(signature)
            if solid_face is None or solid_face in paired_solid:
                exact_pairing = False
                continue
            paired_solid.add(solid_face)
            f_centroid, f_area, f_magnitude = face_geometry(fluid["faces"][fluid_face], fluid["float_points"])
            s_centroid, s_area, s_magnitude = face_geometry(solid["faces"][solid_face], solid["float_points"])
            scale = max(1.0, f_magnitude, s_magnitude)
            roundoff = 64.0 * eps * scale
            if not np.allclose(f_area + s_area, 0.0, rtol=64.0 * eps, atol=roundoff):
                reciprocal_orientation = False
            patch_area += f_magnitude
            records["fluid_face"].append(fluid_face)
            records["solid_face"].append(solid_face)
            records["fluid_owner"].append(fluid["owners"][fluid_face])
            records["solid_owner"].append(solid["owners"][solid_face])
            records["fluid_patch_index"].append(patch_index)
            records["centroid"].append(f_centroid)
            records["fluid_area_vector"].append(f_area)
            records["solid_area_vector"].append(s_area)
            records["area_magnitude"].append(f_magnitude)

        if len(paired_solid) != fluid_count:
            exact_pairing = False
        patch_metadata.append(
            {
                "patch_index": patch_index,
                "fluid_patch": fluid_name,
                "solid_patch": solid_name,
                "patch_type": fluid_patch.get("type"),
                "transform_type": fluid_patch.get("transformType"),
                "face_count": fluid_count,
                "paired_face_count": len(paired_solid),
                "interface_area_m2": patch_area,
            }
        )

    arrays = {
        "fluid_face_global": np.asarray(records["fluid_face"], dtype=np.int64),
        "solid_face_global": np.asarray(records["solid_face"], dtype=np.int64),
        "fluid_owner_cell": np.asarray(records["fluid_owner"], dtype=np.int64),
        "solid_owner_cell": np.asarray(records["solid_owner"], dtype=np.int64),
        "fluid_patch_index": np.asarray(records["fluid_patch_index"], dtype=np.int64),
        "face_centroid_m": np.asarray(records["centroid"], dtype=np.float64),
        "fluid_area_vector_m2": np.asarray(records["fluid_area_vector"], dtype=np.float64),
        "solid_area_vector_m2": np.asarray(records["solid_area_vector"], dtype=np.float64),
        "face_area_m2": np.asarray(records["area_magnitude"], dtype=np.float64),
    }
    npz_path = output / "interface_face_pairs.npz"
    np.savez_compressed(npz_path, **arrays)

    checks = {
        "reciprocal_mapped_patch_metadata": bool(patch_pairs),
        "exact_decimal_vertex_signature_pairing": exact_pairing,
        "face_signatures_are_unique_per_patch": unique_signatures,
        "fluid_and_solid_area_vectors_are_opposite_within_roundoff": reciprocal_orientation,
        "all_interface_faces_are_paired": len(arrays["fluid_face_global"])
        == sum(item["face_count"] for item in patch_metadata),
        "all_interface_areas_are_positive": bool(len(arrays["face_area_m2"]))
        and bool(np.all(arrays["face_area_m2"] > 0.0)),
    }
    source_files = [
        ROOT / "literature/raw/openfoam13_v13/mappedPatches/mappedPatchBase.C",
        ROOT / "literature/raw/openfoam13_v13/mappedPatches/mappedPatchBase.H",
        ROOT / "literature/raw/openfoam13_v13/mappedPatches/mappedWallPolyPatch.C",
        ROOT / "literature/raw/openfoam13_v13/mappedPatches/mappedWallPolyPatch.H",
    ]
    payload = {
        "status": "openfoam_multiregion_interface_pairs_passed" if all(checks.values()) else "openfoam_multiregion_interface_pairs_failed",
        "checks": checks,
        "case": str(case.relative_to(ROOT) if case.is_relative_to(ROOT) else case),
        "fluid_region": args.fluid_region,
        "solid_region": args.solid_region,
        "admitted_patch_types": sorted(admitted_patch_types),
        "patch_pairs": patch_metadata,
        "excluded_reciprocal_patches": excluded_patches,
        "total_paired_faces": int(len(arrays["fluid_face_global"])),
        "total_interface_area_m2": float(arrays["face_area_m2"].sum()),
        "interface_npz_sha256": sha256(npz_path),
        "native_mesh_sha256": {
            f"{region}/{name}": sha256(case / "constant" / region / "polyMesh" / name)
            for region in (args.fluid_region, args.solid_region)
            for name in ("points", "faces", "owner", "boundary")
        },
        "official_openfoam13_source_sha256": {path.name + ":" + path.parent.name: sha256(path) for path in source_files},
        "pairing_rule": "exact sorted Decimal vertex-coordinate signature within reciprocal mapped-patch metadata",
        "roundoff_rule": "64 * IEEE-754 float64 epsilon, used only to audit opposite area-vector arithmetic",
        "new_fitted_physical_parameters": [],
        "neural_training_allowed": False,
        "claim_boundary": (
            "Mesh-topology export only. A passed interface pairing does not admit the HCCB mesh, "
            "the CHT physics, the three-seed dataset, or neural training."
        ),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
