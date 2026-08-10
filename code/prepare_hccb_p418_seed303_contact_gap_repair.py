#!/usr/bin/env python3
"""Check the proposed seed303 contact-gap mesh repair without building a mesh."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from build_hccb_dense_snappy_case import contact_gap_refinement_box


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_close(actual: object, expected: object, label: str) -> None:
    actual_array = np.asarray(actual, dtype=float)
    expected_array = np.asarray(expected, dtype=float)
    if actual_array.shape != expected_array.shape or not np.allclose(
        actual_array, expected_array, rtol=0.0, atol=1.0e-12
    ):
        raise RuntimeError(f"{label} differs from the fixed repair record")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packing", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--reference-mesh-summary", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    reference = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
    reference_summary = json.loads(
        args.reference_mesh_summary.read_text(encoding="utf-8")
    )
    if candidate.get("execution_approved") is not False:
        raise RuntimeError("candidate record must remain unapproved")
    if candidate.get("new_physical_parameters") != []:
        raise RuntimeError("candidate repair adds an unexpected physical parameter")
    if sha256(args.packing) != candidate["source_packing_sha256"]:
        raise RuntimeError("seed303 packing checksum differs from the fixed record")
    if sha256(args.reference_manifest) != candidate["reference_manifest_sha256"]:
        raise RuntimeError("reference manifest checksum differs from the fixed record")
    if (
        sha256(args.reference_mesh_summary)
        != candidate["reference_mesh_summary_sha256"]
    ):
        raise RuntimeError("reference mesh summary checksum differs from the fixed record")
    if reference.get("source_packing_sha256") != candidate["source_packing_sha256"]:
        raise RuntimeError("reference mesh was not generated from the seed303 packing")
    if reference.get("new_physical_parameters") != []:
        raise RuntimeError("reference mesh adds an unexpected physical parameter")
    if float(reference_summary["solid"]["maximum_skewness"]) != float(
        candidate["diagnosed_face"]["maximum_skewness"]
    ):
        raise RuntimeError("diagnosed seed303 skewness differs from the reference log")

    controls = candidate["repair_controls"]
    expected_geometry = candidate["expected_geometry"]
    for key, expected in (
        ("crop_box_dp", expected_geometry["crop_box_dp"]),
        ("physical_particle_diameter_m", expected_geometry["physical_particle_diameter_m"]),
        ("meshing_particle_diameter_m", expected_geometry["meshing_particle_diameter_m"]),
        ("intersecting_particle_count", expected_geometry["intersecting_particle_count"]),
        (
            "retained_particle_fragment_count",
            expected_geometry["retained_particle_fragment_count"],
        ),
        ("omitted_tiny_fragment_count", expected_geometry["omitted_tiny_fragment_count"]),
        ("triangulated_porosity", expected_geometry["triangulated_porosity"]),
        ("solid_obj_sha256", expected_geometry["solid_obj_sha256"]),
    ):
        actual = reference[key]
        if isinstance(expected, list):
            require_close(actual, expected, key)
        elif isinstance(expected, float):
            if not np.isclose(float(actual), expected, rtol=0.0, atol=1.0e-12):
                raise RuntimeError(f"{key} differs from the fixed repair record")
        elif actual != expected:
            raise RuntimeError(f"{key} differs from the fixed repair record")

    reference_controls = reference["numerical_controls"]
    fixed_control_pairs = (
        ("background_cells_per_particle_diameter", controls["background_cells_per_particle_diameter"]),
        ("sphere_icosphere_subdivisions", controls["sphere_icosphere_subdivisions"]),
        ("surface_refinement_level", controls["global_surface_refinement_level"]),
        ("cells_between_levels", controls["cells_between_levels"]),
        ("solid_cell_source", controls["solid_cell_source"]),
    )
    for key, expected in fixed_control_pairs:
        actual = reference_controls[key]
        if isinstance(expected, float):
            if not np.isclose(float(actual), expected, rtol=0.0, atol=1.0e-12):
                raise RuntimeError(f"reference numerical control {key} changed")
        elif actual != expected:
            raise RuntimeError(f"reference numerical control {key} changed")
    if controls["local_refinement_region"] != "contact-gap":
        raise RuntimeError("repair region must remain contact-gap")
    if int(controls["local_refinement_level"]) != 3:
        raise RuntimeError("local repair level must remain 3")
    if int(controls["global_surface_refinement_level"]) != 2:
        raise RuntimeError("global surface refinement must remain 2")
    if controls["particle_ids"] != [1595, 951]:
        raise RuntimeError("repair particle pair differs from the diagnosed pair")

    with np.load(args.packing) as data:
        particle_ids = np.asarray(data["particle_id"], dtype=np.int64)
        centres = np.asarray(data["centres_m"], dtype=float)
        physical_radius = float(data["physical_radius_m"])
        meshing_radius = float(data["meshing_radius_m"])
    selected_indices: list[int] = []
    for particle_id in controls["particle_ids"]:
        matches = np.flatnonzero(particle_ids == particle_id)
        if len(matches) != 1:
            raise RuntimeError(
                f"particle {particle_id} occurs {len(matches)} times in seed303"
            )
        selected_indices.append(int(matches[0]))
    selected_centres = centres[np.asarray(selected_indices, dtype=np.int64)]
    require_close(
        selected_centres,
        candidate["expected_contact_gap"]["particle_centres_global_m"],
        "contact-pair centres",
    )
    particle_diameter = 2.0 * physical_radius
    crop = (
        np.asarray(expected_geometry["crop_box_dp"], dtype=float).reshape(3, 2)
        * particle_diameter
    )
    lower, upper = crop[:, 0], crop[:, 1]
    refinement_box, details = contact_gap_refinement_box(
        selected_centres - lower,
        meshing_radius,
        upper - lower,
        particle_diameter,
        float(controls["background_cells_per_particle_diameter"]),
        int(controls["local_refinement_level"]),
        float(controls["local_refinement_padding_cells"]),
    )
    expected_gap = candidate["expected_contact_gap"]
    require_close(
        refinement_box,
        expected_gap["local_refinement_box_m"],
        "contact-gap refinement box",
    )
    for key in (
        "centre_distance_m",
        "meshing_surface_gap_m",
        "target_level_nominal_cell_size_m",
    ):
        if not np.isclose(
            float(details[key]), float(expected_gap[key]), rtol=0.0, atol=1.0e-12
        ):
            raise RuntimeError(f"{key} differs from the fixed repair record")

    payload = {
        "status": "seed303_contact_gap_repair_preflight_passed_no_mesh_started",
        "seed": 303,
        "source_packing_sha256": sha256(args.packing),
        "reference_manifest_sha256": sha256(args.reference_manifest),
        "reference_mesh_summary_sha256": sha256(args.reference_mesh_summary),
        "particle_ids": controls["particle_ids"],
        "particle_centres_global_m": selected_centres.tolist(),
        "contact_gap_refinement_box_m": refinement_box.tolist(),
        "contact_gap_details": details,
        "global_surface_refinement_level": controls[
            "global_surface_refinement_level"
        ],
        "local_refinement_level": controls["local_refinement_level"],
        "local_refinement_padding_cells": controls[
            "local_refinement_padding_cells"
        ],
        "execution_approved": False,
        "mesh_generator_started": False,
        "heat_transfer_solver_started": False,
        "new_physical_parameters": [],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
