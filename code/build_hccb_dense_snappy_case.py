#!/usr/bin/env python3
"""Build a dense HCCB crop for OpenFOAM snappyHexMesh.

The crop geometry is taken directly from an existing audited sphere packing.
Particles intersecting the crop are translated into a local box and clipped by
the box planes.  snappyHexMesh then creates a hex-dominant fluid/solid mesh.
Only numerical mesh controls are introduced here; physical dimensions and
particle radii are inherited from the packing artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from build_hccb_pore_resolved_openfoam_mesh import foam_header
from check_hccb_source_sequence_lammps_packing import sphere_box_intersection_volume


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contact_gap_refinement_box(
    centres: np.ndarray,
    meshing_radius: float,
    domain_box: np.ndarray,
    particle_diameter: float,
    cells_per_diameter: float,
    refinement_level: int,
    padding_cells: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Bound only the nearest-surface region of one diagnosed particle pair."""
    if centres.shape != (2, 3):
        raise ValueError("contact-gap refinement requires exactly two particle centres")
    separation = centres[1] - centres[0]
    centre_distance = float(np.linalg.norm(separation))
    if centre_distance <= 0.0:
        raise ValueError("contact-gap particle centres must be distinct")
    if padding_cells <= 0.0:
        raise ValueError("contact-gap padding must be positive")
    direction = separation / centre_distance
    surface_points = np.vstack(
        (
            centres[0] + direction * meshing_radius,
            centres[1] - direction * meshing_radius,
        )
    )
    target_cell_size = (
        particle_diameter / cells_per_diameter / float(2**refinement_level)
    )
    padding = padding_cells * target_cell_size
    refinement_box = np.vstack(
        (
            np.maximum(0.0, surface_points.min(axis=0) - padding),
            np.minimum(domain_box, surface_points.max(axis=0) + padding),
        )
    )
    if np.any(refinement_box[1] <= refinement_box[0]):
        raise ValueError("derived contact-gap refinement box is empty")
    metadata: dict[str, object] = {
        "centre_distance_m": centre_distance,
        "meshing_surface_gap_m": centre_distance - 2.0 * meshing_radius,
        "nearest_surface_points_m": surface_points.tolist(),
        "target_level_nominal_cell_size_m": target_cell_size,
        "padding_cells": padding_cells,
        "padding_m": padding,
    }
    return refinement_box, metadata


def block_mesh_dict(box: np.ndarray, dp: float, cells_per_dp: float) -> str:
    lx, ly, lz = map(float, box)
    nx, ny, nz = [
        max(2, int(round(length / dp * cells_per_dp))) for length in box
    ]
    return foam_header("blockMeshDict") + f"""convertToMeters 1;

vertices
(
    (0 0 0)
    ({lx:.12g} 0 0)
    ({lx:.12g} {ly:.12g} 0)
    (0 {ly:.12g} 0)
    (0 0 {lz:.12g})
    ({lx:.12g} 0 {lz:.12g})
    ({lx:.12g} {ly:.12g} {lz:.12g})
    (0 {ly:.12g} {lz:.12g})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges ();

boundary
(
    inlet
    {{
        type patch;
        faces ((0 3 2 1));
    }}
    outlet
    {{
        type patch;
        faces ((4 5 6 7));
    }}
    coolingWall
    {{
        type wall;
        faces ((0 4 7 3));
    }}
    symmetryWalls
    {{
        type symmetry;
        faces ((1 2 6 5) (0 1 5 4) (3 7 6 2));
    }}
);

mergePatchPairs ();
// ************************************************************************* //
"""


def snappy_dict(
    inside: np.ndarray,
    surface_refinement: int,
    cells_between_levels: int,
    local_refinement_box: np.ndarray | None = None,
    local_refinement_level: int | None = None,
) -> str:
    x, y, z = map(float, inside)
    local_geometry = ""
    local_regions = ""
    if local_refinement_box is not None:
        if local_refinement_box.shape != (2, 3):
            raise ValueError("local refinement box must have shape (2, 3)")
        if local_refinement_level is None or local_refinement_level <= surface_refinement:
            raise ValueError("local refinement level must exceed surface refinement")
        lo = " ".join(f"{value:.12g}" for value in local_refinement_box[0])
        hi = " ".join(f"{value:.12g}" for value in local_refinement_box[1])
        local_geometry = f"""
    localRefinement
    {{
        type searchableBox;
        min ({lo});
        max ({hi});
    }}
"""
        local_regions = f"""
        localRefinement
        {{
            mode inside;
            levels ((1e15 {local_refinement_level}));
        }}
"""
    return foam_header("snappyHexMeshDict") + f"""#includeEtc "caseDicts/mesh/generation/snappyHexMeshDict.cfg"

castellatedMesh on;
snap            on;
addLayers       off;

geometry
{{
    solid
    {{
        type triSurface;
        file "solid.obj";
    }}
{local_geometry}
}}

castellatedMeshControls
{{
    features ();
    refinementSurfaces
    {{
        solid
        {{
            level ({surface_refinement} {surface_refinement});
            faceZone solid;
            cellZone solid;
            mode inside;
        }}
    }}
    refinementRegions
    {{
{local_regions}    }}
    insidePoint ({x:.12g} {y:.12g} {z:.12g});
    nCellsBetweenLevels {cells_between_levels};
}}

snapControls
{{
    explicitFeatureSnap off;
    implicitFeatureSnap off;
    nFeatureSnapIter 0;
    multiRegionFeatureSnap off;
}}

addLayersControls {{ layers {{}} }}
writeFlags (scalarLevels layerSets layerFields);
mergeTolerance 1e-6;
// ************************************************************************* //
"""


def topo_set_dict(
    box: np.ndarray,
    dp: float,
    fluid_point: np.ndarray,
    include_cut_cells: bool,
) -> str:
    lx, ly, lz = map(float, box)
    fx, fy, fz = map(float, fluid_point)
    margin = dp
    return foam_header("topoSetDict") + f"""actions
(
    {{
        name surfaceSolidCells;
        type cellSet;
        action new;
        source surfaceToCell;
        sourceInfo
        {{
            file "constant/triSurface/solid.obj";
            outsidePoints (({fx:.12g} {fy:.12g} {fz:.12g}));
            includeCut {str(include_cut_cells).lower()};
            includeInside true;
            includeOutside false;
            useSurfaceOrientation false;
            nearDistance -1;
            curvature -1;
        }}
    }}
    {{
        name fluidCandidates;
        type cellSet;
        action new;
        source boxToCell;
        sourceInfo
        {{
            box ({-margin:.12g} {-margin:.12g} {-margin:.12g})
                ({lx + margin:.12g} {ly + margin:.12g} {lz + margin:.12g});
        }}
    }}
    {{
        name fluidCandidates;
        type cellSet;
        action delete;
        source cellToCell;
        sourceInfo {{ set surfaceSolidCells; }}
    }}
    {{
        name fluidCells;
        type cellSet;
        action new;
        source regionToCell;
        sourceInfo
        {{
            set fluidCandidates;
            nErode 0;
            insidePoints (({fx:.12g} {fy:.12g} {fz:.12g}));
        }}
    }}
    {{
        name solidCells;
        type cellSet;
        action new;
        source boxToCell;
        sourceInfo
        {{
            box ({-margin:.12g} {-margin:.12g} {-margin:.12g})
                ({lx + margin:.12g} {ly + margin:.12g} {lz + margin:.12g});
        }}
    }}
    {{
        name solidCells;
        type cellSet;
        action delete;
        source cellToCell;
        sourceInfo {{ set fluidCells; }}
    }}
    {{
        name solid;
        type cellZoneSet;
        action new;
        source setToCellZone;
        sourceInfo {{ set solidCells; }}
    }}
    {{
        name fluid;
        type cellZoneSet;
        action new;
        source setToCellZone;
        sourceInfo {{ set fluidCells; }}
    }}
);
// ************************************************************************* //
"""


def topo_set_from_snappy_zone_dict(
    box: np.ndarray,
    dp: float,
    fluid_point: np.ndarray,
) -> str:
    """Retain snappy's solid cell zone and keep only inlet-connected fluid."""
    lx, ly, lz = map(float, box)
    fx, fy, fz = map(float, fluid_point)
    margin = dp
    return foam_header("topoSetDict") + f"""actions
(
    {{
        name solidCells;
        type cellSet;
        action new;
        source zoneToCell;
        sourceInfo {{ zone solid; }}
    }}
    {{
        name fluidCandidate;
        type cellSet;
        action new;
        source boxToCell;
        sourceInfo
        {{
            box ({-margin:.12g} {-margin:.12g} {-margin:.12g})
                ({lx + margin:.12g} {ly + margin:.12g} {lz + margin:.12g});
        }}
    }}
    {{
        name fluidCandidate;
        type cellSet;
        action delete;
        source cellToCell;
        sourceInfo {{ set solidCells; }}
    }}
    {{
        name fluidCells;
        type cellSet;
        action new;
        source regionToCell;
        sourceInfo
        {{
            set fluidCandidate;
            insidePoints (({fx:.12g} {fy:.12g} {fz:.12g}));
            nErode 0;
        }}
    }}
    {{
        name orphanCells;
        type cellSet;
        action new;
        source cellToCell;
        sourceInfo {{ set fluidCandidate; }}
    }}
    {{
        name orphanCells;
        type cellSet;
        action delete;
        source cellToCell;
        sourceInfo {{ set fluidCells; }}
    }}
    {{
        name solidCells;
        type cellSet;
        action add;
        source cellToCell;
        sourceInfo {{ set orphanCells; }}
    }}
    {{
        name solid;
        type cellZoneSet;
        action new;
        source setToCellZone;
        sourceInfo {{ set solidCells; }}
    }}
    {{
        name fluid;
        type cellZoneSet;
        action new;
        source setToCellZone;
        sourceInfo {{ set fluidCells; }}
    }}
);
// ************************************************************************* //
"""


def choose_fluid_point(
    centres: np.ndarray,
    radius: float,
    box: np.ndarray,
    dp: float,
) -> tuple[np.ndarray, float]:
    """Choose an interior point with the largest distance from a sphere surface."""
    axes = [
        np.linspace(0.12 * dp, length - 0.12 * dp, 19) for length in box
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([component.ravel() for component in mesh])
    clearance = np.full(len(points), np.inf)
    for start in range(0, len(centres), 32):
        delta = points[:, None, :] - centres[None, start : start + 32, :]
        distance = np.linalg.norm(delta, axis=2) - radius
        clearance = np.minimum(clearance, distance.min(axis=1))
    index = int(np.argmax(clearance))
    if clearance[index] <= 0.0:
        raise RuntimeError("could not find a fluid point outside the particles")
    return points[index], float(clearance[index])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packing", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--crop-box-dp",
        nargs=6,
        type=float,
        required=True,
        metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
    )
    parser.add_argument("--cells-per-diameter", type=float, default=10.101010101)
    parser.add_argument("--sphere-subdivisions", type=int, default=3)
    parser.add_argument("--surface-refinement", type=int, default=1)
    parser.add_argument("--cells-between-levels", type=int, default=2)
    parser.add_argument(
        "--reuse-surface-case",
        type=Path,
        help=(
            "reuse the already closed solid.obj from a case with the same packing, "
            "crop and sphere discretisation; useful for mesh-density comparisons"
        ),
    )
    parser.add_argument(
        "--include-cut-cells-in-solid",
        action="store_true",
        help=(
            "classify cells intersected by the particle surface as solid; "
            "use this for the cell-volume porosity sensitivity test"
        ),
    )
    parser.add_argument(
        "--solid-cell-source",
        choices=("surface", "snappy-zone"),
        default="surface",
        help=(
            "derive solid cells either by a post-mesh surface query or from "
            "snappyHexMesh's native solid cell zone"
        ),
    )
    parser.add_argument(
        "--minimum-retained-volume-fraction",
        type=float,
        default=1.0e-3,
        help=(
            "omit numerical boundary fragments smaller than this fraction of a "
            "complete meshing sphere"
        ),
    )
    parser.add_argument(
        "--local-refinement-particle-ids",
        nargs="+",
        type=int,
        help=(
            "source particle IDs whose complete meshing-sphere bounding box is "
            "refined one additional level; intended for a diagnosed contact pair"
        ),
    )
    parser.add_argument(
        "--local-refinement-level",
        type=int,
        help="refinement level inside the derived local particle bounding box",
    )
    parser.add_argument(
        "--local-refinement-region",
        choices=("complete-spheres", "contact-gap"),
        default="complete-spheres",
        help=(
            "refine either the complete bounding box of named particles or only "
            "their nearest-surface gap"
        ),
    )
    parser.add_argument(
        "--local-refinement-padding-cells",
        type=float,
        default=4.0,
        help="contact-gap box padding measured in target-level nominal cells",
    )
    args = parser.parse_args()

    if args.cells_per_diameter <= 0.0:
        raise ValueError("cells-per-diameter must be positive")
    if args.sphere_subdivisions not in (1, 2, 3, 4):
        raise ValueError("sphere-subdivisions must be between 1 and 4")
    if args.surface_refinement < 0 or args.cells_between_levels < 1:
        raise ValueError("mesh refinement settings are outside their valid range")
    if not 0.0 <= args.minimum_retained_volume_fraction < 1.0:
        raise ValueError("minimum-retained-volume-fraction must be in [0, 1)")
    if (args.local_refinement_particle_ids is None) != (args.local_refinement_level is None):
        raise ValueError(
            "local refinement particle IDs and local refinement level must be supplied together"
        )
    if args.local_refinement_padding_cells <= 0.0:
        raise ValueError("local refinement padding cells must be positive")

    with np.load(args.packing) as data:
        centres = np.asarray(data["centres_m"], dtype=float)
        particle_ids = np.asarray(data["particle_id"], dtype=np.int64)
        physical_radius = float(data["physical_radius_m"])
        meshing_radius = float(data["meshing_radius_m"])
    dp = 2.0 * physical_radius
    crop = np.asarray(args.crop_box_dp, dtype=float).reshape(3, 2) * dp
    lower, upper = crop[:, 0], crop[:, 1]
    if np.any(upper <= lower):
        raise ValueError("crop upper bounds must exceed lower bounds")
    intersects = np.all(centres + meshing_radius > lower, axis=1) & np.all(
        centres - meshing_radius < upper, axis=1
    )
    if not np.any(intersects):
        raise RuntimeError("crop does not intersect any particles")
    intersecting_indices = np.flatnonzero(intersects)
    sphere_volume = (4.0 / 3.0) * np.pi * meshing_radius**3
    retained_fractions = np.empty(len(intersecting_indices), dtype=float)
    retained_volumes = np.empty(len(intersecting_indices), dtype=float)
    for local_index, source_index in enumerate(intersecting_indices):
        retained_volume, _ = sphere_box_intersection_volume(
            centres[source_index], lower, upper, meshing_radius
        )
        retained_volumes[local_index] = retained_volume
        retained_fractions[local_index] = retained_volume / sphere_volume
    keep = retained_fractions >= args.minimum_retained_volume_fraction
    selected_indices = intersecting_indices[keep]
    omitted_indices = intersecting_indices[~keep]
    local_centres = centres[selected_indices] - lower[None, :]
    local_ids = particle_ids[selected_indices]
    box = upper - lower
    local_refinement_box = None
    local_refinement_details: dict[str, object] | None = None
    local_refinement_ids: list[int] = []
    if args.local_refinement_particle_ids is not None:
        local_refinement_ids = list(dict.fromkeys(args.local_refinement_particle_ids))
        selected: list[int] = []
        for particle_id in local_refinement_ids:
            matches = np.flatnonzero(local_ids == particle_id)
            if len(matches) != 1:
                raise ValueError(
                    f"local refinement particle ID {particle_id} occurs {len(matches)} times in the crop"
                )
            selected.append(int(matches[0]))
        selected_centres = local_centres[np.asarray(selected, dtype=np.int64)]
        if args.local_refinement_region == "complete-spheres":
            local_refinement_box = np.vstack(
                (
                    np.maximum(0.0, selected_centres.min(axis=0) - meshing_radius),
                    np.minimum(box, selected_centres.max(axis=0) + meshing_radius),
                )
            )
        else:
            local_refinement_box, local_refinement_details = contact_gap_refinement_box(
                selected_centres,
                meshing_radius,
                box,
                dp,
                args.cells_per_diameter,
                int(args.local_refinement_level),
                args.local_refinement_padding_cells,
            )
        if np.any(local_refinement_box[1] <= local_refinement_box[0]):
            raise ValueError("derived local refinement box is empty")

    out = args.output_dir
    if out.exists():
        import shutil

        shutil.rmtree(out)
    (out / "constant/triSurface").mkdir(parents=True)
    (out / "system").mkdir(parents=True)
    geometry_dir = out / "geometry"
    geometry_dir.mkdir()
    crop_packing = geometry_dir / "packing_crop.npz"
    np.savez_compressed(
        crop_packing,
        centres_m=local_centres,
        particle_id=local_ids,
        physical_radius_m=np.asarray(physical_radius),
        meshing_radius_m=np.asarray(meshing_radius),
        box_lengths_m=box,
        crop_lo_dp=np.zeros(3),
        crop_hi_dp=box / dp,
        parent_crop_lower_m=lower,
        parent_crop_upper_m=upper,
    )

    solid_obj = out / "constant/triSurface/solid.obj"
    surface_summary = geometry_dir / "solid_surface_summary.json"
    reused_surface_case = None
    if args.reuse_surface_case is not None:
        import shutil

        reused_surface_case = args.reuse_surface_case.resolve()
        reused_manifest_path = reused_surface_case / "case_manifest.json"
        reused_summary_path = reused_surface_case / "geometry/solid_surface_summary.json"
        reused_obj_path = reused_surface_case / "constant/triSurface/solid.obj"
        if not all(path.is_file() for path in (reused_manifest_path, reused_summary_path, reused_obj_path)):
            raise FileNotFoundError("reuse-surface case is missing its manifest, summary or solid.obj")
        reused_manifest = json.loads(reused_manifest_path.read_text(encoding="utf-8"))
        reused_summary = json.loads(reused_summary_path.read_text(encoding="utf-8"))
        controls = reused_manifest.get("numerical_controls", {})
        if reused_manifest.get("source_packing_sha256") != sha256(args.packing):
            raise RuntimeError("reused surface was built from a different packing")
        if not np.allclose(reused_manifest.get("crop_box_dp", []), args.crop_box_dp, rtol=0.0, atol=1.0e-12):
            raise RuntimeError("reused surface has a different crop box")
        if controls.get("sphere_icosphere_subdivisions") != args.sphere_subdivisions:
            raise RuntimeError("reused surface has a different sphere subdivision")
        if not all(reused_summary.get("checks", {}).values()):
            raise RuntimeError("reused particle surface did not pass its geometry checks")
        shutil.copy2(reused_obj_path, solid_obj)
        surface = dict(reused_summary)
        surface["packing"] = str(crop_packing)
        surface["reused_from_case"] = str(reused_surface_case)
        surface["reused_solid_obj_sha256"] = sha256(reused_obj_path)
        surface_summary.write_text(
            json.dumps(surface, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    else:
        # A single global plane-snap tolerance does not close every independent
        # packing: seed101 closes at 2 micrometres, while a seed303 corner cut
        # requires 1 micrometre. Try the same documented sequence for every
        # packing and keep the first closed aggregate surface.
        surface_command = [
            sys.executable,
            str(ROOT / "code/build_clipped_hccb_solid_surface_vtk.py"),
            "--packing",
            str(crop_packing),
            "--output-obj",
            str(solid_obj),
            "--summary",
            str(surface_summary),
            "--subdivisions",
            str(args.sphere_subdivisions),
        ]
        for plane_tolerance in (2.0e-6, 1.0e-6, 5.0e-7, 0.0):
            completed = subprocess.run(
                [
                    *surface_command,
                    "--plane-snap-tolerance-m",
                    str(plane_tolerance),
                ],
                check=False,
            )
            if completed.returncode == 0:
                break
            if completed.returncode != 2:
                raise subprocess.CalledProcessError(
                    completed.returncode, completed.args
                )
        else:
            raise RuntimeError(
                "clipped particle surface stayed open for all snap tolerances"
            )
        surface = json.loads(surface_summary.read_text(encoding="utf-8"))
    if not all(surface["checks"].values()):
        raise RuntimeError("clipped particle surface is not closed and bounded")

    inside, inside_clearance = choose_fluid_point(
        local_centres, meshing_radius, box, dp
    )
    (out / "system/blockMeshDict").write_text(
        block_mesh_dict(box, dp, args.cells_per_diameter), encoding="ascii"
    )
    (out / "system/snappyHexMeshDict").write_text(
        snappy_dict(
            inside,
            args.surface_refinement,
            args.cells_between_levels,
            local_refinement_box,
            args.local_refinement_level,
        ),
        encoding="ascii",
    )
    if args.solid_cell_source == "snappy-zone":
        topo_text = topo_set_from_snappy_zone_dict(box, dp, inside)
    else:
        topo_text = topo_set_dict(
            box, dp, inside, args.include_cut_cells_in_solid
        )
    (out / "system/topoSetDict").write_text(topo_text, encoding="ascii")
    (out / "system/meshQualityDict").write_text(
        '#includeEtc "caseDicts/mesh/generation/meshQualityDict.cfg"\n',
        encoding="ascii",
    )
    (out / "system/controlDict").write_text(
        foam_header("controlDict")
        + "application foamMultiRun;\nstartFrom startTime;\nstartTime 0;\n"
        + "stopAt endTime;\nendTime 1;\ndeltaT 1;\nwriteControl timeStep;\n"
        + "writeInterval 1;\nrunTimeModifiable false;\n",
        encoding="ascii",
    )
    allmesh = out / "Allmesh"
    allmesh.write_text(
        "#!/bin/sh\nset -eu\ncd \"${0%/*}\"\n"
        '. "$WM_PROJECT_DIR/bin/tools/RunFunctions"\n'
        "runApplication blockMesh\n"
        "runApplication snappyHexMesh\n"
        "runApplication topoSet\n"
        "runApplication splitMeshRegions -cellZonesOnly -defaultRegionName fluid\n",
        encoding="ascii",
    )
    allmesh.chmod(0o755)

    manifest = {
        "status": "hccb_dense_snappy_case_built",
        "purpose": "dense pore-scale fluid-solid mesh for subsequent flow and heat transfer",
        "source_packing": str(args.packing),
        "source_packing_sha256": sha256(args.packing),
        "crop_box_dp": list(args.crop_box_dp),
        "crop_lower_m": lower.tolist(),
        "crop_upper_m": upper.tolist(),
        "box_lengths_m": box.tolist(),
        "intersecting_particle_count": int(intersects.sum()),
        "retained_particle_fragment_count": int(keep.sum()),
        "omitted_tiny_fragment_count": int((~keep).sum()),
        "omitted_source_particle_ids": particle_ids[omitted_indices].tolist(),
        "minimum_retained_volume_fraction": args.minimum_retained_volume_fraction,
        "omitted_solid_volume_m3": float(retained_volumes[~keep].sum()),
        "porosity_change_from_omission": float(
            retained_volumes[~keep].sum() / np.prod(box)
        ),
        "physical_particle_diameter_m": dp,
        "meshing_particle_diameter_m": 2.0 * meshing_radius,
        "triangulated_porosity": surface["triangulated_porosity"],
        "fluid_inside_point_m": inside.tolist(),
        "fluid_inside_point_surface_clearance_m": inside_clearance,
        "numerical_controls": {
            "background_cells_per_particle_diameter": args.cells_per_diameter,
            "sphere_icosphere_subdivisions": args.sphere_subdivisions,
            "surface_plane_snap_tolerance_m": surface.get(
                "plane_snap_tolerance_m"
            ),
            "surface_refinement_level": args.surface_refinement,
            "cells_between_levels": args.cells_between_levels,
            "include_cut_cells_in_solid": args.include_cut_cells_in_solid,
            "solid_cell_source": args.solid_cell_source,
            "connected_fluid_region_only": True,
            "local_refinement_particle_ids": local_refinement_ids,
            "local_refinement_level": args.local_refinement_level,
            "local_refinement_region": args.local_refinement_region,
            "local_refinement_padding_cells": args.local_refinement_padding_cells,
            "local_refinement_details": local_refinement_details,
            "local_refinement_box_m": (
                local_refinement_box.tolist() if local_refinement_box is not None else None
            ),
            "local_refinement_box_is_derived_from_complete_meshing_spheres": bool(
                local_refinement_box is not None
                and args.local_refinement_region == "complete-spheres"
            ),
        },
        "solid_surface_summary_sha256": sha256(surface_summary),
        "reused_surface_case": str(reused_surface_case) if reused_surface_case else None,
        "solid_obj_sha256": sha256(solid_obj),
        "new_physical_parameters": [],
    }
    (out / "case_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
