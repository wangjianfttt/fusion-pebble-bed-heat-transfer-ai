#!/usr/bin/env python3
"""Export one solved HCCB fluid-solid case as a reusable neural-model sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from export_openfoam_multiregion_interface_pairs import (
    NUMBER,
    face_geometry,
    parse_boundary,
    parse_labels,
)
from openfoam_ascii_field import read_openfoam_ascii_field


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_openfoam_function(case: Path, region: str, time_name: str, function: str) -> None:
    completed = subprocess.run(
        [
            "postProcess",
            "-case",
            str(case),
            "-region",
            region,
            "-time",
            time_name,
            "-func",
            function,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0 or "FOAM FATAL" in completed.stdout:
        raise RuntimeError(
            f"OpenFOAM {function} failed for {region} at {time_name}:\n{completed.stdout[-2000:]}"
        )


def region_description(case: Path, region: str) -> dict:
    mesh = case / "constant" / region / "polyMesh"
    owners = np.asarray(parse_labels(mesh / "owner"), dtype=np.int64)
    neighbours = np.asarray(parse_labels(mesh / "neighbour"), dtype=np.int64)
    n_cells = int(max(owners.max(), neighbours.max()) + 1)
    patches = parse_boundary(mesh / "boundary")
    patch_sizes = {name: int(entries["nFaces"]) for name, entries in patches.items()}
    return {
        "mesh": mesh,
        "owners": owners,
        "neighbours": neighbours,
        "n_cells": n_cells,
        "patches": patches,
        "patch_sizes": patch_sizes,
    }


def field(case: Path, time_name: str, region: str, name: str, description: dict) -> np.ndarray:
    return read_openfoam_ascii_field(
        case / time_name / region / name,
        internal_count=(
            len(description["neighbours"])
            if name == "phi"
            else description["n_cells"]
        ),
        patch_sizes=description["patch_sizes"],
    ).internal


def boundary_field(
    case: Path,
    time_name: str,
    region: str,
    name: str,
    description: dict,
    *,
    require_all: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stored patch values and a mask for values explicitly written by OpenFOAM."""
    parsed = read_openfoam_ascii_field(
        case / time_name / region / name,
        internal_count=(
            len(description["neighbours"])
            if name == "phi"
            else description["n_cells"]
        ),
        patch_sizes=description["patch_sizes"],
    )
    parts: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for patch_name in description["patches"]:
        values = parsed.boundary_value[patch_name]
        if values is None:
            if require_all:
                raise ValueError(
                    f"{region}/{name} patch {patch_name} has no explicit boundary value"
                )
            count = description["patch_sizes"][patch_name]
            parts.append(np.zeros((count, parsed.internal.shape[1]), dtype=np.float64))
            masks.append(np.zeros(count, dtype=np.bool_))
        else:
            parts.append(values)
            masks.append(np.ones(len(values), dtype=np.bool_))
    return np.concatenate(parts, axis=0), np.concatenate(masks)


def parse_float_points(path: Path) -> np.ndarray:
    """Read ASCII OpenFOAM points without retaining Decimal copies."""
    record = re.compile(
        rf"^\s*\(\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*\)\s*$"
    )
    points: list[tuple[float, float, float]] = []
    with path.open(encoding="utf-8", errors="strict") as stream:
        for line in stream:
            match = record.match(line)
            if match:
                points.append(tuple(float(value) for value in match.groups()))
    if not points:
        raise ValueError(f"no points parsed from {path}")
    return np.asarray(points, dtype=np.float64)


def parse_selected_faces(path: Path, selected: set[int]) -> dict[int, list[int]]:
    """Read only requested ASCII faces, avoiding a full-mesh face list in memory."""
    record = re.compile(r"^\s*(\d+)\s*\(([^()]*)\)\s*$")
    faces: dict[int, list[int]] = {}
    face_index = 0
    with path.open(encoding="utf-8", errors="strict") as stream:
        for line in stream:
            match = record.match(line)
            if not match:
                continue
            if face_index in selected:
                labels = [int(value) for value in match.group(2).split()]
                if len(labels) != int(match.group(1)):
                    raise ValueError(f"face vertex count mismatch at face {face_index}")
                faces[face_index] = labels
                if len(faces) == len(selected):
                    break
            face_index += 1
    missing = sorted(selected.difference(faces))
    if missing:
        raise ValueError(f"faces file misses selected boundary faces: {missing[:8]}")
    return faces


def boundary_arrays(
    description: dict,
    *,
    reused: dict[str, np.ndarray] | None = None,
    prefix: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    patch_names = list(description["patches"])
    owner_parts: list[np.ndarray] = []
    patch_parts: list[np.ndarray] = []
    face_indices: list[int] = []
    for patch_id, name in enumerate(patch_names):
        entries = description["patches"][name]
        start = int(entries["startFace"])
        count = int(entries["nFaces"])
        owner_parts.append(description["owners"][start : start + count])
        patch_parts.append(np.full(count, patch_id, dtype=np.int16))
        face_indices.extend(range(start, start + count))

    owners = np.concatenate(owner_parts)
    patches = np.concatenate(patch_parts)
    if reused is not None:
        if prefix is None:
            raise ValueError("boundary geometry reuse requires a region prefix")
        owner_key = f"{prefix}_boundary_face_owner"
        patch_key = f"{prefix}_boundary_face_patch"
        if not np.array_equal(owners, reused[owner_key]):
            raise ValueError(f"reused {prefix} boundary owner indices differ")
        if not np.array_equal(patches, reused[patch_key]):
            raise ValueError(f"reused {prefix} boundary patch indices differ")
        centroids = reused[f"{prefix}_boundary_face_centroid_m"]
        area_vectors = reused[
            f"{prefix}_boundary_face_area_vector_outward_m2"
        ]
        areas = reused[f"{prefix}_boundary_face_area_m2"]
    else:
        points = parse_float_points(description["mesh"] / "points")
        faces = parse_selected_faces(description["mesh"] / "faces", set(face_indices))
        centroid_parts: list[np.ndarray] = []
        area_vector_parts: list[np.ndarray] = []
        area_parts: list[float] = []
        for face_index in face_indices:
            centroid, area_vector, area = face_geometry(faces[face_index], points)
            centroid_parts.append(centroid)
            area_vector_parts.append(area_vector)
            area_parts.append(area)
        centroids = np.asarray(centroid_parts, dtype=np.float64)
        area_vectors = np.asarray(area_vector_parts, dtype=np.float64)
        areas = np.asarray(area_parts, dtype=np.float64)
    return (
        owners,
        patches,
        np.asarray(centroids, dtype=np.float64),
        np.asarray(area_vectors, dtype=np.float64),
        np.asarray(areas, dtype=np.float64),
        patch_names,
    )


def parameter_rows(path: Path, parameter_ids: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = {row["parameter_id"]: row for row in csv.DictReader(stream)}
    missing = sorted(set(parameter_ids) - set(rows))
    if missing:
        raise ValueError(f"parameter manifest misses {missing}")
    return [rows[parameter_id] for parameter_id in parameter_ids]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--time", required=True)
    parser.add_argument("--parameter-manifest", type=Path, required=True)
    parser.add_argument("--interface-pairs", type=Path)
    parser.add_argument(
        "--reuse-boundary-geometry",
        type=Path,
        help="schema-v3 sample on the identical mesh; geometry is verified by owner/patch arrays",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-postprocess", action="store_true")
    args = parser.parse_args()

    case = args.case.resolve()
    time_name = str(args.time)
    time_dir = case / time_name
    if not time_dir.is_dir():
        raise FileNotFoundError(time_dir)
    output = (args.output_dir or (case / f"training_sample_{time_name}" )).resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.run_postprocess:
        for region in ("fluid", "solid"):
            run_openfoam_function(case, region, time_name, "writeCellCentres")
            run_openfoam_function(case, region, time_name, "writeCellVolumes")

    fluid = region_description(case, "fluid")
    solid = region_description(case, "solid")
    reused_boundary: dict[str, np.ndarray] | None = None
    reused_boundary_source = None
    if args.reuse_boundary_geometry:
        reused_boundary_source = args.reuse_boundary_geometry.resolve()
        required_reuse = {
            f"{region}_boundary_face_{suffix}"
            for region in ("fluid", "solid")
            for suffix in (
                "owner",
                "patch",
                "centroid_m",
                "area_vector_outward_m2",
                "area_m2",
            )
        }
        with np.load(reused_boundary_source, allow_pickle=False) as loaded:
            missing_reuse = sorted(required_reuse.difference(loaded.files))
            if missing_reuse:
                raise ValueError(
                    f"reused boundary geometry misses {missing_reuse}"
                )
            reused_boundary = {name: loaded[name] for name in required_reuse}
    arrays: dict[str, np.ndarray] = {
        "fluid_cell_centroid_m": field(case, time_name, "fluid", "C", fluid),
        "fluid_cell_volume_m3": field(case, time_name, "fluid", "Vc", fluid)[:, 0],
        "fluid_velocity_m_s": field(case, time_name, "fluid", "U", fluid),
        "fluid_pressure_Pa": field(case, time_name, "fluid", "p", fluid)[:, 0],
        "fluid_temperature_K": field(case, time_name, "fluid", "T", fluid)[:, 0],
        "fluid_density_kg_m3": field(case, time_name, "fluid", "rho", fluid)[:, 0],
        "fluid_internal_face_owner": fluid["owners"][: len(fluid["neighbours"])],
        "fluid_internal_face_neighbour": fluid["neighbours"],
        "fluid_internal_face_mass_flow_kg_s": field(
            case, time_name, "fluid", "phi", fluid
        )[:, 0],
        "solid_cell_centroid_m": field(case, time_name, "solid", "C", solid),
        "solid_cell_volume_m3": field(case, time_name, "solid", "Vc", solid)[:, 0],
        "solid_internal_face_owner": solid["owners"][: len(solid["neighbours"])],
        "solid_internal_face_neighbour": solid["neighbours"],
        "solid_temperature_K": field(case, time_name, "solid", "T", solid)[:, 0],
    }
    (
        fluid_boundary_owner,
        fluid_boundary_patch,
        fluid_boundary_centroid,
        fluid_boundary_area_vector,
        fluid_boundary_area,
        fluid_patch_names,
    ) = boundary_arrays(fluid, reused=reused_boundary, prefix="fluid")
    (
        solid_boundary_owner,
        solid_boundary_patch,
        solid_boundary_centroid,
        solid_boundary_area_vector,
        solid_boundary_area,
        solid_patch_names,
    ) = boundary_arrays(solid, reused=reused_boundary, prefix="solid")
    fluid_boundary_velocity, fluid_boundary_velocity_mask = boundary_field(
        case, time_name, "fluid", "U", fluid
    )
    fluid_boundary_pressure, fluid_boundary_pressure_mask = boundary_field(
        case, time_name, "fluid", "p", fluid
    )
    fluid_boundary_temperature, fluid_boundary_temperature_mask = boundary_field(
        case, time_name, "fluid", "T", fluid
    )
    fluid_boundary_density, fluid_boundary_density_mask = boundary_field(
        case, time_name, "fluid", "rho", fluid
    )
    fluid_boundary_mass_flow, fluid_boundary_mass_flow_mask = boundary_field(
        case, time_name, "fluid", "phi", fluid, require_all=True
    )
    solid_boundary_temperature, solid_boundary_temperature_mask = boundary_field(
        case, time_name, "solid", "T", solid
    )
    arrays.update(
        {
            "fluid_boundary_face_owner": fluid_boundary_owner,
            "fluid_boundary_face_patch": fluid_boundary_patch,
            "fluid_boundary_face_centroid_m": fluid_boundary_centroid,
            "fluid_boundary_face_area_vector_outward_m2": fluid_boundary_area_vector,
            "fluid_boundary_face_area_m2": fluid_boundary_area,
            "fluid_boundary_velocity_m_s": fluid_boundary_velocity,
            "fluid_boundary_velocity_value_mask": fluid_boundary_velocity_mask,
            "fluid_boundary_pressure_Pa": fluid_boundary_pressure[:, 0],
            "fluid_boundary_pressure_value_mask": fluid_boundary_pressure_mask,
            "fluid_boundary_temperature_K": fluid_boundary_temperature[:, 0],
            "fluid_boundary_temperature_value_mask": fluid_boundary_temperature_mask,
            "fluid_boundary_density_kg_m3": fluid_boundary_density[:, 0],
            "fluid_boundary_density_value_mask": fluid_boundary_density_mask,
            "fluid_boundary_face_mass_flow_kg_s": fluid_boundary_mass_flow[:, 0],
            "fluid_boundary_mass_flow_value_mask": fluid_boundary_mass_flow_mask,
            "solid_boundary_face_owner": solid_boundary_owner,
            "solid_boundary_face_patch": solid_boundary_patch,
            "solid_boundary_face_centroid_m": solid_boundary_centroid,
            "solid_boundary_face_area_vector_outward_m2": solid_boundary_area_vector,
            "solid_boundary_face_area_m2": solid_boundary_area,
            "solid_boundary_temperature_K": solid_boundary_temperature[:, 0],
            "solid_boundary_temperature_value_mask": solid_boundary_temperature_mask,
        }
    )

    interface_source = None
    if args.interface_pairs:
        interface_source = args.interface_pairs.resolve()
        with np.load(interface_source, allow_pickle=False) as loaded:
            required = {
                "fluid_owner_cell",
                "solid_owner_cell",
                "face_centroid_m",
                "fluid_area_vector_m2",
                "face_area_m2",
            }
            if not required.issubset(loaded.files):
                raise ValueError(f"interface file misses {sorted(required - set(loaded.files))}")
            arrays.update(
                {
                    "interface_fluid_cell": loaded["fluid_owner_cell"],
                    "interface_solid_cell": loaded["solid_owner_cell"],
                    "interface_face_centroid_m": loaded["face_centroid_m"],
                    "interface_area_vector_fluid_outward_m2": loaded[
                        "fluid_area_vector_m2"
                    ],
                    "interface_face_area_m2": loaded["face_area_m2"],
                }
            )

    for name, values in arrays.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values")
    if np.any(arrays["fluid_cell_volume_m3"] <= 0.0) or np.any(
        arrays["solid_cell_volume_m3"] <= 0.0
    ):
        raise ValueError("cell volumes must be positive")
    if np.any(arrays["fluid_boundary_face_area_m2"] <= 0.0) or np.any(
        arrays["solid_boundary_face_area_m2"] <= 0.0
    ):
        raise ValueError("boundary face areas must be positive")
    if interface_source is not None:
        if np.any(arrays["interface_fluid_cell"] >= fluid["n_cells"]):
            raise ValueError("interface fluid cell index is outside the fluid mesh")
        if np.any(arrays["interface_solid_cell"] >= solid["n_cells"]):
            raise ValueError("interface solid cell index is outside the solid mesh")

    sample_path = output / "fields_and_topology.npz"
    np.savez_compressed(sample_path, **arrays)
    case_metadata = json.loads((case / "cht_smoke_metadata.json").read_text(encoding="utf-8"))
    ids = list(case_metadata["parameter_ids"])
    rows = parameter_rows(args.parameter_manifest.resolve(), ids)
    metadata = {
        "schema_version": 3,
        "sample_id": f"{case.name}_t{time_name}",
        "case": str(case),
        "time": time_name,
        "fluid_cells": fluid["n_cells"],
        "solid_cells": solid["n_cells"],
        "fluid_internal_faces": len(fluid["neighbours"]),
        "interface_faces": int(len(arrays.get("interface_fluid_cell", []))),
        "fluid_patch_names": fluid_patch_names,
        "solid_patch_names": solid_patch_names,
        "physical_conditions": case_metadata,
        "literature_parameters": rows,
        "array_shapes": {name: list(values.shape) for name, values in arrays.items()},
        "array_dtypes": {name: str(values.dtype) for name, values in arrays.items()},
        "sample_sha256": sha256(sample_path),
        "parameter_manifest_sha256": sha256(args.parameter_manifest.resolve()),
        "interface_pairs_sha256": sha256(interface_source) if interface_source else None,
        "reused_boundary_geometry": (
            str(reused_boundary_source) if reused_boundary_source else None
        ),
        "reused_boundary_geometry_sha256": (
            sha256(reused_boundary_source) if reused_boundary_source else None
        ),
        "use": case_metadata.get(
            "sample_use",
            "One steady three-dimensional fluid-solid sample for data-pipeline and model-code development.",
        ),
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
