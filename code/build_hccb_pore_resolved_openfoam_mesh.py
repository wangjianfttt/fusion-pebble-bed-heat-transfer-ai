#!/usr/bin/env python3
"""Build a mesh-only OpenFOAM 13 preflight for the HCCB pore-resolved RVE.

Physical geometry is read from the literature parameter manifest and from a
previously audited sphere-packing artifact.  Mesh resolution and triangulation
are numerical controls and are recorded explicitly; this script does not create
or fit physical parameters and does not run a heat-transfer solution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "parameters/literature_parameter_manifest.csv"
P405_TARGET_CELLS = {"G1": 4_870_000, "G2": 6_670_000, "G3": 8_810_000}
P405_SPHERE_SUBDIVISIONS = 3
P405_SURFACE_REFINEMENT = 1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_rows() -> dict[str, dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return {row["parameter_id"]: row for row in csv.DictReader(handle)}


def derive_mesh_controls(
    mesh_level: str,
    requested_cells_per_diameter: float | None,
    requested_sphere_subdivisions: int | None,
    requested_surface_refinement: int | None,
) -> dict[str, float | int | str | None]:
    if mesh_level == "smoke":
        return {
            "mesh_level": mesh_level,
            "cells_per_diameter": requested_cells_per_diameter
            if requested_cells_per_diameter is not None
            else 2.0,
            "sphere_subdivisions": requested_sphere_subdivisions
            if requested_sphere_subdivisions is not None
            else 1,
            "surface_refinement": requested_surface_refinement
            if requested_surface_refinement is not None
            else 1,
            "role": "smoke/preflight controls, not physical parameters and not the P405 publication mesh",
            "target_cell_count": None,
            "derivation": "smoke software preflight",
        }
    if mesh_level not in P405_TARGET_CELLS:
        raise ValueError(f"unknown mesh level {mesh_level!r}")
    g2_cells_per_diameter = 1.0 / 0.099
    target_cell_count = P405_TARGET_CELLS[mesh_level]
    cells_per_diameter = g2_cells_per_diameter * (
        target_cell_count / P405_TARGET_CELLS["G2"]
    ) ** (1.0 / 3.0)
    if requested_cells_per_diameter is not None and not math.isclose(
        requested_cells_per_diameter,
        cells_per_diameter,
        rel_tol=64.0 * np.finfo(np.float64).eps,
        abs_tol=0.0,
    ):
        raise ValueError("P405 mesh levels use the frozen source-derived cells-per-diameter value")
    if (
        requested_sphere_subdivisions is not None
        and requested_sphere_subdivisions != P405_SPHERE_SUBDIVISIONS
    ):
        raise ValueError(
            "P405 production meshes use the frozen ND057 sphere representation; "
            "alternative subdivisions belong in a separately labelled sensitivity case"
        )
    if (
        requested_surface_refinement is not None
        and requested_surface_refinement != P405_SURFACE_REFINEMENT
    ):
        raise ValueError(
            "P405 production meshes use the frozen ND057 surface-refinement control"
        )
    return {
        "mesh_level": mesh_level,
        "cells_per_diameter": cells_per_diameter,
        "sphere_subdivisions": P405_SPHERE_SUBDIVISIONS,
        "surface_refinement": P405_SURFACE_REFINEMENT,
        "role": "P405 mesh-count candidate",
        "target_cell_count": target_cell_count,
        "derivation": (
            "G2=1/0.099dp; G1/G3 scale by cube root of P405 published cell-count ratio; "
            "ND057 freezes the disclosed numerical sphere representation separately from P405"
        ),
    }


def icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = np.asarray(
        [
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
        ],
        dtype=float,
    )
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    faces = np.asarray(
        [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ],
        dtype=np.int64,
    )
    for _ in range(subdivisions):
        cache: dict[tuple[int, int], int] = {}
        new_vertices = vertices.tolist()

        def midpoint(i: int, j: int) -> int:
            edge = (min(i, j), max(i, j))
            if edge not in cache:
                point = 0.5 * (vertices[i] + vertices[j])
                point /= np.linalg.norm(point)
                cache[edge] = len(new_vertices)
                new_vertices.append(point.tolist())
            return cache[edge]

        new_faces: list[tuple[int, int, int]] = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(int(a), int(b)), midpoint(int(b), int(c)), midpoint(int(c), int(a))
            new_faces.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
        vertices = np.asarray(new_vertices, dtype=float)
        faces = np.asarray(new_faces, dtype=np.int64)
    return vertices, faces


def write_obj(path: Path, centres: np.ndarray, radius: float, subdivisions: int) -> tuple[int, int]:
    unit_vertices, unit_faces = icosphere(subdivisions)
    vertex_count = 0
    face_count = 0
    with path.open("w", encoding="ascii") as handle:
        handle.write("# Numerically reconstructed HCCB Li4SiO4 spheres\n")
        for sphere_index, centre in enumerate(centres):
            handle.write(f"o pebble_{sphere_index:05d}\n")
            points = centre[None, :] + radius * unit_vertices
            for x, y, z in points:
                handle.write(f"v {x:.12e} {y:.12e} {z:.12e}\n")
            offset = vertex_count + 1
            for a, b, c in unit_faces:
                handle.write(f"f {offset + int(a)} {offset + int(b)} {offset + int(c)}\n")
            vertex_count += len(unit_vertices)
            face_count += len(unit_faces)
    return vertex_count, face_count


def count_obj(path: Path) -> tuple[int, int]:
    vertex_count = 0
    face_count = 0
    with path.open(encoding="ascii", errors="strict") as handle:
        for line in handle:
            if line.startswith("v "):
                vertex_count += 1
            elif line.startswith("f "):
                face_count += 1
    return vertex_count, face_count


def foam_header(object_name: str, class_name: str = "dictionary") -> str:
    return f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  13
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       {class_name};
    object      {object_name};
}}
// Mesh-only APD006 preflight. No heat-transfer solution is run.

"""


def block_mesh_dict(lx: float, ly: float, dp: float, cells_per_dp: float) -> str:
    z_planes = (-10.0 * dp, 0.0, 10.0 * dp, 20.0 * dp)
    nx = max(2, int(round(lx / dp * cells_per_dp)))
    ny = max(2, int(round(ly / dp * cells_per_dp)))
    nz = max(2, int(round(10.0 * cells_per_dp)))
    vertices = []
    for z in z_planes:
        vertices.extend(((0, 0, z), (lx, 0, z), (lx, ly, z), (0, ly, z)))
    vertex_text = "\n".join(f"    ({x:.12g} {y:.12g} {z:.12g})" for x, y, z in vertices)
    blocks = "\n".join(
        f"    hex ({4*i} {4*i+1} {4*i+2} {4*i+3} {4*i+4} {4*i+5} {4*i+6} {4*i+7}) "
        f"({nx} {ny} {nz}) simpleGrading (1 1 1)"
        for i in range(3)
    )
    return foam_header("blockMeshDict") + f"""convertToMeters 1;

vertices
(
{vertex_text}
);

blocks
(
{blocks}
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
        faces ((12 13 14 15));
    }}
    inletTemperatureWall
    {{
        type wall;
        faces ((0 4 7 3));
    }}
    coolingWall
    {{
        type wall;
        faces ((4 8 11 7) (8 12 15 11));
    }}
    symmetryWalls
    {{
        type symmetry;
        faces
        (
            (1 2 6 5) (5 6 10 9) (9 10 14 13)
            (0 1 5 4) (4 5 9 8) (8 9 13 12)
            (3 7 6 2) (7 11 10 6) (11 15 14 10)
        );
    }}
);

mergePatchPairs ();
// ************************************************************************* //
"""


def snappy_dict(inside: tuple[float, float, float], surface_level: int) -> str:
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
}}

castellatedMeshControls
{{
    features ({{ file "solid.eMesh"; level {surface_level}; }});
    refinementSurfaces
    {{
        solid
        {{
            level ({surface_level} {surface_level});
            faceZone solid;
            cellZone solid;
            mode inside;
        }}
    }}
    refinementRegions {{}}
    insidePoint ({inside[0]:.12g} {inside[1]:.12g} {inside[2]:.12g});
    nCellsBetweenLevels 2;
}}

snapControls
{{
    // The particles are smooth spheres. Explicit edge snapping is unnecessary
    // and can collapse cells where a sphere intersects an outer crop plane.
    explicitFeatureSnap off;
    implicitFeatureSnap off;
    nFeatureSnapIter 0;
}}

addLayersControls {{ layers {{}} }}
writeFlags (scalarLevels layerSets layerFields);
mergeTolerance 1e-6;
// ************************************************************************* //
"""


def topo_set_dict(lx: float, ly: float, dp: float) -> str:
    margin = dp
    return foam_header("topoSetDict") + f"""actions
(
    {{
        name solidCells;
        type cellSet;
        action new;
        source surfaceToCell;
        sourceInfo
        {{
            file "constant/triSurface/solid_full.obj";
            outsidePoints
            (
                ({0.5*lx:.12g} {0.5*ly:.12g} {-5.0*dp:.12g})
                ({0.5*lx:.12g} {0.5*ly:.12g} {15.0*dp:.12g})
            );
            includeCut false;
            includeInside true;
            includeOutside false;
            useSurfaceOrientation false;
            nearDistance -1;
            curvature -1;
        }}
    }}
    {{
        name solid;
        type cellZoneSet;
        action new;
        source setToCellZone;
        sourceInfo {{ set solidCells; }}
    }}
    {{
        name fluidCells;
        type cellSet;
        action new;
        source boxToCell;
        sourceInfo
        {{
            box ({-margin:.12g} {-margin:.12g} {-11.0*dp:.12g})
                ({lx + margin:.12g} {ly + margin:.12g} {21.0*dp:.12g});
        }}
    }}
    {{
        name fluidCells;
        type cellSet;
        action delete;
        source cellToCell;
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


def topo_set_fluid_complement_dict(lx: float, ly: float, dp: float) -> str:
    """Keep the snappy solid zone and name every remaining cell as fluid."""
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
        name fluidCells;
        type cellSet;
        action new;
        source boxToCell;
        sourceInfo
        {{
            box ({-margin:.12g} {-margin:.12g} {-11.0*dp:.12g})
                ({lx + margin:.12g} {ly + margin:.12g} {21.0*dp:.12g});
        }}
    }}
    {{
        name fluidCells;
        type cellSet;
        action delete;
        source cellToCell;
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packing", type=Path, required=True)
    parser.add_argument(
        "--packing-audit",
        type=Path,
        help="independent audit JSON; defaults to independent_audit.json beside the packing",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mesh-level", choices=("smoke", "G1", "G2", "G3"), default="smoke")
    parser.add_argument("--cells-per-diameter", type=float)
    parser.add_argument("--sphere-subdivisions", type=int)
    parser.add_argument("--surface-refinement", type=int)
    parser.add_argument(
        "--solid-obj",
        type=Path,
        help="pre-clipped closed solid surface; required when crop planes intersect particles",
    )
    parser.add_argument("--solid-surface-summary", type=Path)
    parser.add_argument(
        "--project-outer-boundaries",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "project outer-patch points back to the exact box planes; defaults to "
            "on for full spheres and off for a pre-clipped surface"
        ),
    )
    parser.add_argument(
        "--region-classification",
        choices=("snappy", "surface-cell-centres"),
        default=None,
        help=(
            "use the solid cell zone created during surface snapping, or rebuild "
            "solid/fluid zones from cell centres after meshing"
        ),
    )
    args = parser.parse_args()
    controls = derive_mesh_controls(
        args.mesh_level,
        args.cells_per_diameter,
        args.sphere_subdivisions,
        args.surface_refinement,
    )
    cells_per_diameter = float(controls["cells_per_diameter"])
    sphere_subdivisions = int(controls["sphere_subdivisions"])
    surface_refinement = int(controls["surface_refinement"])
    mesh_role = str(controls["role"])
    target_cell_count = controls["target_cell_count"]
    if cells_per_diameter <= 0:
        raise ValueError("cells-per-diameter must be positive")
    if sphere_subdivisions not in (0, 1, 2, 3, 4):
        raise ValueError("sphere-subdivisions must be between 0 and 4")
    if surface_refinement < 0:
        raise ValueError("surface-refinement must be nonnegative")

    rows = manifest_rows()
    with np.load(args.packing) as data:
        centres = np.asarray(data["centres_m"], dtype=float)
        radius = float(data["meshing_radius_m"])
        box = np.asarray(data["box_lengths_m"], dtype=float)
        physical_radius = float(data["physical_radius_m"])
    packing_summary_path = args.packing.with_name("summary.json")
    packing_summary = json.loads(packing_summary_path.read_text(encoding="utf-8"))
    supported_status = {
        "hccb_openmc_packing_generated",
        "hccb_source_sequence_lammps_target_packing_exported",
    }
    if packing_summary.get("status") not in supported_status:
        raise RuntimeError("unsupported packing source")
    if not all(packing_summary.get("checks", {}).values()):
        raise RuntimeError("packing contains a failed geometry check")
    if packing_summary.get("packing_npz_sha256") not in (None, sha256(args.packing)):
        raise RuntimeError("packing NPZ hash differs from its summary")

    packing_audit_path = None
    packing_audit = None
    if packing_summary.get("status") == "hccb_openmc_packing_generated":
        packing_audit_path = args.packing_audit or args.packing.with_name("independent_audit.json")
        packing_audit = json.loads(packing_audit_path.read_text(encoding="utf-8"))
        if packing_audit.get("status") != "hccb_openmc_candidate_audit_passed":
            raise RuntimeError("packing did not pass the independent OpenMC candidate check")
        if not all(packing_audit.get("checks", {}).values()):
            raise RuntimeError("independent packing check contains a failed item")
        if packing_audit.get("packing_npz_sha256") != sha256(args.packing):
            raise RuntimeError("packing NPZ hash differs from the independently checked artifact")

    out = args.output_dir
    if out.exists():
        shutil.rmtree(out)
    (out / "constant/triSurface").mkdir(parents=True)
    (out / "system").mkdir(parents=True)
    obj = out / "constant/triSurface/solid.obj"
    solid_surface_summary = None
    if args.solid_obj is not None:
        shutil.copyfile(args.solid_obj, obj)
        n_vertices, n_faces = count_obj(obj)
        if args.solid_surface_summary is None:
            raise ValueError("--solid-surface-summary is required with --solid-obj")
        solid_surface_summary = json.loads(
            args.solid_surface_summary.read_text(encoding="utf-8")
        )
        if solid_surface_summary.get("status") != "clipped_hccb_solid_surface_generated":
            raise RuntimeError("unsupported clipped-surface summary")
        if not all(solid_surface_summary.get("checks", {}).values()):
            raise RuntimeError("clipped solid surface contains a failed geometry check")
    else:
        n_vertices, n_faces = write_obj(obj, centres, radius, sphere_subdivisions)
    project_outer_boundaries = (
        args.project_outer_boundaries
        if args.project_outer_boundaries is not None
        else args.solid_obj is None
    )
    region_classification = args.region_classification or (
        "snappy" if args.solid_obj is not None else "surface-cell-centres"
    )
    full_obj = out / "constant/triSurface/solid_full.obj"
    full_obj_vertices, full_obj_faces = write_obj(
        full_obj, centres, radius, sphere_subdivisions
    )
    dp = 2.0 * physical_radius
    (out / "system/blockMeshDict").write_text(
        block_mesh_dict(float(box[0]), float(box[1]), dp, cells_per_diameter), encoding="ascii"
    )
    inside = (0.5 * float(box[0]), 0.5 * float(box[1]), -5.0 * dp)
    (out / "system/snappyHexMeshDict").write_text(
        snappy_dict(inside, surface_refinement), encoding="ascii"
    )
    topo_dictionary = (
        topo_set_fluid_complement_dict(float(box[0]), float(box[1]), dp)
        if region_classification == "snappy"
        else topo_set_dict(float(box[0]), float(box[1]), dp)
    )
    (out / "system/topoSetDict").write_text(topo_dictionary, encoding="ascii")
    (out / "system/surfaceFeaturesDict").write_text(
        foam_header("surfaceFeaturesDict")
        + 'surfaces ("solid.obj");\n'
        + "includedAngle 150;\n"
        + "subsetFeatures { nonManifoldEdges yes; openEdges yes; }\n"
        + "trimFeatures { minElem 0; minLen 0; }\n"
        + "writeObj yes;\n\n// ************************************************************************* //\n",
        encoding="ascii",
    )
    (out / "system/meshQualityDict").write_text(
        '#includeEtc "caseDicts/mesh/generation/meshQualityDict.cfg"\n',
        encoding="ascii",
    )
    (out / "system/controlDict").write_text(
        foam_header("controlDict")
        + "application foamMultiRun;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\n"
        + "deltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\nrunTimeModifiable false;\n",
        encoding="ascii",
    )
    if project_outer_boundaries:
        projection_script = out / "project_outer_boundaries.py"
        shutil.copyfile(ROOT / "code/project_hccb_outer_boundary_points.py", projection_script)
        projection_script.chmod(0o755)
    allmesh = out / "Allmesh"
    mesh_commands = (
        "runApplication blockMesh\n"
        "runApplication surfaceFeatures\n"
        "runApplication snappyHexMesh\n"
    )
    if project_outer_boundaries:
        mesh_commands += "python3 project_outer_boundaries.py --case .\n"
    mesh_commands += "runApplication topoSet\n"
    mesh_commands += "runApplication splitMeshRegions -cellZonesOnly -defaultRegionName fluid\n"
    allmesh.write_text(
        "#!/bin/sh\nset -eu\ncd \"${0%/*}\"\n. \"$WM_PROJECT_DIR/bin/tools/RunFunctions\"\n"
        + mesh_commands,
        encoding="ascii",
    )
    allmesh.chmod(0o755)

    payload = {
        "status": "hccb_pore_resolved_openfoam_mesh_case_built",
        "purpose": "mesh-only numerical preflight; no physical CHT result",
        "parameter_ids": ["P048", "P049", "P390", "P392", "P404", "P405"],
        "numerical_design_ids": ["ND056", "ND057"],
        "particle_count": int(len(centres)),
        "packing_seed": packing_summary.get("seed"),
        "packing_reconstruction_mode": packing_summary.get("reconstruction_mode")
        or (packing_audit or {}).get("reconstruction_mode"),
        "physical_particle_diameter_m": 2.0 * physical_radius,
        "meshing_particle_diameter_m": 2.0 * radius,
        "packing_box_m": box.tolist(),
        "full_channel_domain_m": [float(box[0]), float(box[1]), 30.0 * dp],
        "boundary_basis": rows["P392"]["value"],
        "numerical_controls": {
            "mesh_level": args.mesh_level,
            "background_cells_per_particle_diameter": cells_per_diameter,
            "sphere_icosphere_subdivisions": sphere_subdivisions,
            "surface_refinement_level": surface_refinement,
            "role": mesh_role,
            "published_target_cell_count": target_cell_count,
            "derivation": controls["derivation"],
        },
        "published_mesh_target": rows["P405"]["value"],
        "obj_vertices": n_vertices,
        "obj_faces": n_faces,
        "full_sphere_obj_vertices": full_obj_vertices,
        "full_sphere_obj_faces": full_obj_faces,
        "packing_summary_sha256": sha256(packing_summary_path),
        "packing_independent_check_sha256": sha256(packing_audit_path)
        if packing_audit_path is not None
        else None,
        "packing_npz_sha256": sha256(args.packing),
        "solid_obj_sha256": sha256(obj),
        "solid_full_obj_sha256": sha256(full_obj),
        "solid_surface_mode": "preclipped_closed_surface"
        if args.solid_obj is not None
        else "unclipped_full_spheres",
        "solid_surface_summary_sha256": sha256(args.solid_surface_summary)
        if args.solid_surface_summary is not None
        else None,
        "outer_boundary_point_projection": project_outer_boundaries,
        "region_classification": region_classification,
        "new_fitted_physical_parameters": [],
        "allowed_claim": "OpenFOAM mesh construction and region-splitting preflight only",
        "forbidden_claims": ["reproduced pressure drop", "reproduced temperature", "neural training truth"],
    }
    (out / "case_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
