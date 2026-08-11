#!/usr/bin/env python3
"""Build fine-mesh CHT cases for the exact 60-condition P418 matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from build_hccb_gmsh_cht_smoke_case import (
    matrix_condition_id,
    parse_p418_matrix,
    patch_area_m2,
)
from hccb_p418_source_contract import (
    ALL_STEADY_PHYSICAL_PARAMETER_IDS,
    CASE_PHYSICS_PARAMETER_IDS,
    MESH_GEOMETRY_SOURCE_PARAMETER_IDS,
    OPERATING_PARAMETER_IDS,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "parameters/literature_parameter_manifest.csv"
DEFAULT_MESH_CASE = ROOT / "runs/hccb_dense_cht_native_r2"
DEFAULT_MESH_MANIFEST = ROOT / "runs/hccb_dense_snappy_g2_nativezone_r2/case_manifest.json"
DEFAULT_OUTPUT_ROOT = ROOT / "runs/hccb_dense_cht_p418"
CASE_BUILDER = ROOT / "code/build_hccb_gmsh_cht_smoke_case.py"
PILOT_CONDITIONS = [
    "u0p20_T700_q6p85",
    "u0p05_T300_q8p85",
    "u0p05_T900_q8p85",
    "u0p25_T300_q4p85",
    "u0p25_T900_q4p85",
]


def manifest_rows() -> dict[str, dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return {row["parameter_id"]: row for row in csv.DictReader(handle)}


def all_conditions(value: str) -> dict[str, tuple[float, float, float]]:
    velocities, temperatures, sources = parse_p418_matrix(value)
    return {
        matrix_condition_id(velocity, temperature, source): (velocity, temperature, source)
        for velocity in velocities
        for temperature in temperatures
        for source in sources
    }


def copy_mesh_with_detached_boundary(source: Path, destination: Path) -> None:
    """Hard-link immutable mesh arrays and copy boundary files before editing."""
    for region in ("fluid", "solid"):
        src = source / "constant" / region / "polyMesh"
        dst = destination / "constant" / region / "polyMesh"
        if not src.is_dir():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, copy_function=os.link)
        boundary = dst / "boundary"
        contents = boundary.read_bytes()
        boundary.unlink()
        boundary.write_bytes(contents)


def source_solid_volume(mesh_case: Path, mesh_check_summary: Path | None) -> float:
    """Read the solid volume already measured for the shared fine mesh."""
    if mesh_check_summary is not None:
        summary = json.loads(mesh_check_summary.read_text(encoding="utf-8"))
        volume = summary.get("solid", {}).get("volume_m3")
        if volume is None:
            raise ValueError(f"solid volume is absent from {mesh_check_summary}")
        return float(volume)
    candidates = sorted(mesh_case.glob("cht_result_summary*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"no CHT result summary with solid volume is available in {mesh_case}"
        )
    summary = json.loads(candidates[-1].read_text(encoding="utf-8"))
    return float(summary["heat_balance"]["solid_volume_m3"])


def validate_resumable_case(
    *,
    case: Path,
    condition_id: str,
    mesh_source_packing_sha256: str,
) -> dict[str, object]:
    """Validate an already-built unsolved case before reusing it in a matrix."""
    metadata_path = case / "cht_smoke_metadata.json"
    required = (
        metadata_path,
        case / "constant/fluid/polyMesh/boundary",
        case / "constant/solid/polyMesh/boundary",
        case / "system/controlDict",
        case / "system/fvSolution",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            f"existing case {case} is incomplete and cannot be resumed: {missing}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("operating_condition_id") != condition_id:
        raise ValueError(f"existing case {case} has a different operating condition")
    if tuple(metadata.get("parameter_ids", ())) != CASE_PHYSICS_PARAMETER_IDS:
        raise ValueError(f"existing case {case} has different physical inputs")
    if metadata.get("mesh_source_packing_sha256") != mesh_source_packing_sha256:
        raise ValueError(f"existing case {case} comes from a different packing")
    if metadata.get("source_channel_volume_flow_preserved") is not True:
        raise ValueError(
            f"existing case {case} does not preserve the published inlet-channel volume flow"
        )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-case", type=Path, default=DEFAULT_MESH_CASE)
    parser.add_argument("--mesh-manifest", type=Path, default=DEFAULT_MESH_MANIFEST)
    parser.add_argument(
        "--mesh-check-summary",
        type=Path,
        help="checkMesh summary used to obtain the solid volume before the first CHT solve",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mode", choices=("pilot", "all", "selected"), default="pilot")
    parser.add_argument("--condition-id", action="append", default=[])
    parser.add_argument("--parallel-subdomains", type=int, default=32)
    parser.add_argument("--end-time", type=int, default=200)
    parser.add_argument("--write-interval", type=int, default=25)
    parser.add_argument("--energy-correctors", type=int, default=20)
    parser.add_argument(
        "--mesh-resolution-label",
        choices=("coarse", "medium", "fine"),
        default="fine",
        help="Descriptive mesh level recorded in metadata; it changes no physics.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "Reuse an existing case only after checking its condition, literature-input "
            "list, packing hash and required mesh/control files. Missing cases are built."
        ),
    )
    args = parser.parse_args()
    if args.overwrite and args.resume_existing:
        raise ValueError("--overwrite and --resume-existing cannot be used together")

    rows = manifest_rows()
    for parameter_id in ALL_STEADY_PHYSICAL_PARAMETER_IDS:
        if rows[parameter_id]["status"] != "extracted":
            raise RuntimeError(f"{parameter_id} is not an extracted literature value")
    available = all_conditions(rows["P418"]["value"])
    if len(available) != 60:
        raise RuntimeError(f"P418 should define 60 cases, found {len(available)}")

    if args.mode == "pilot":
        selected = PILOT_CONDITIONS
    elif args.mode == "all":
        selected = sorted(available)
    else:
        if not args.condition_id:
            raise ValueError("--mode selected requires at least one --condition-id")
        selected = args.condition_id
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError(f"conditions are not in the published P418 matrix: {unknown}")

    mesh_case = args.mesh_case.resolve()
    mesh_manifest_path = args.mesh_manifest.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    mesh_check_summary = (
        args.mesh_check_summary.resolve() if args.mesh_check_summary else None
    )
    mesh_solid_volume_m3 = source_solid_volume(mesh_case, mesh_check_summary)
    fluid_inlet_area_m2 = patch_area_m2(mesh_case, "fluid", "inlet")
    solid_inlet_area_m2 = patch_area_m2(mesh_case, "solid", "inlet")
    inlet_open_area_fraction = fluid_inlet_area_m2 / (
        fluid_inlet_area_m2 + solid_inlet_area_m2
    )
    mesh_manifest = json.loads(mesh_manifest_path.read_text(encoding="utf-8"))
    mesh_source_hash = str(mesh_manifest["source_packing_sha256"])
    built: list[dict[str, object]] = []
    for condition_id in selected:
        case = output_root / condition_id
        if case.exists():
            if args.resume_existing:
                metadata = validate_resumable_case(
                    case=case,
                    condition_id=condition_id,
                    mesh_source_packing_sha256=mesh_source_hash,
                )
                velocity, temperature, source = available[condition_id]
                built.append(
                    {
                        "condition_id": condition_id,
                        "case_path": str(case.relative_to(ROOT)),
                        "inlet_velocity_m_s": velocity,
                        "pore_opening_boundary_velocity_m_s": metadata[
                            "pore_opening_boundary_velocity_m_s"
                        ],
                        "inlet_open_area_fraction": metadata[
                            "inlet_open_area_fraction"
                        ],
                        "inlet_temperature_K": temperature,
                        "solid_heat_source_MW_m3": source,
                    }
                )
                continue
            if not args.overwrite:
                raise FileExistsError(f"{case} exists; pass --overwrite to replace it")
            shutil.rmtree(case)
        copy_mesh_with_detached_boundary(mesh_case, case)
        command = [
            sys.executable,
            str(CASE_BUILDER),
            "--case",
            str(case),
            "--inlet-patch",
            "inlet",
            "--outlet-patch",
            "outlet",
            "--cooling-wall-patch",
            "coolingWall",
            "--fluid-solid-patch",
            "fluid_to_solid",
            "--solid-fluid-patch",
            "solid_to_fluid",
            "--symmetry-patch",
            "symmetryWalls",
            "--helium-mode",
            "tabulated",
            "--parallel-subdomains",
            str(args.parallel_subdomains),
            "--end-time",
            str(args.end_time),
            "--write-interval",
            str(args.write_interval),
            "--energy-correctors",
            str(args.energy_correctors),
            "--solve-flow-during-energy",
            "--fluid-inlet-area-m2",
            str(fluid_inlet_area_m2),
            "--solid-inlet-area-m2",
            str(solid_inlet_area_m2),
            "--p418-condition-id",
            condition_id,
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        metadata_path = case / "cht_smoke_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if tuple(metadata.get("parameter_ids", ())) != CASE_PHYSICS_PARAMETER_IDS:
            raise ValueError(
                f"{condition_id} records different calculation inputs: "
                f"{metadata.get('parameter_ids')}"
            )
        metadata.update(
            {
                "status": (
                    f"hccb_dense_{args.mesh_resolution_label}_mesh_p418_cht_case_built"
                ),
                "purpose": (
                    f"{args.mesh_resolution_label}-mesh pore-resolved helium-flow and "
                    "fluid-solid heat-transfer calculation over one exact P418 operating point"
                ),
                "mesh_resolution_label": args.mesh_resolution_label,
                "mesh_case": str(mesh_case.relative_to(ROOT)),
                "mesh_source_packing_sha256": mesh_source_hash,
                "mesh_crop_box_dp": mesh_manifest["crop_box_dp"],
                "mesh_triangulated_porosity": mesh_manifest["triangulated_porosity"],
                "mesh_solid_volume_m3": mesh_solid_volume_m3,
                "matrix_role": (
                    f"one new {args.mesh_resolution_label}-mesh three-dimensional CHT "
                    "calculation at an exact published P418 operating point"
                ),
                "sample_use": (
                    f"One {args.mesh_resolution_label}-mesh three-dimensional CHT sample "
                    "for mesh-sensitivity or multi-condition field calculations after "
                    "convergence and conservation checks."
                ),
            }
        )
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        velocity, temperature, source = available[condition_id]
        built.append(
            {
                "condition_id": condition_id,
                "case_path": str(case.relative_to(ROOT)),
                "inlet_velocity_m_s": velocity,
                "pore_opening_boundary_velocity_m_s": metadata[
                    "pore_opening_boundary_velocity_m_s"
                ],
                "inlet_open_area_fraction": metadata["inlet_open_area_fraction"],
                "inlet_temperature_K": temperature,
                "solid_heat_source_MW_m3": source,
            }
        )

    payload = {
        "status": "p418_cht_case_matrix_built",
        "mode": args.mode,
        "source_title": rows["P418"]["source_title"],
        "source_doi": rows["P418"]["source_url_or_doi"],
        "source_matrix_size": len(available),
        "selected_case_count": len(built),
        "published_conditions": [
            {
                "condition_id": condition_id,
                "inlet_velocity_m_s": values[0],
                "inlet_temperature_K": values[1],
                "solid_heat_source_MW_m3": values[2],
            }
            for condition_id, values in sorted(available.items())
        ],
        "parameter_ids": list(ALL_STEADY_PHYSICAL_PARAMETER_IDS),
        "operating_parameter_ids": list(OPERATING_PARAMETER_IDS),
        "case_physics_parameter_ids": list(CASE_PHYSICS_PARAMETER_IDS),
        "mesh_geometry_source_parameter_ids": list(MESH_GEOMETRY_SOURCE_PARAMETER_IDS),
        "mesh_case": str(mesh_case.relative_to(ROOT)),
        "mesh_manifest": str(mesh_manifest_path.relative_to(ROOT)),
        "mesh_check_summary": (
            str(mesh_check_summary.relative_to(ROOT)) if mesh_check_summary else None
        ),
        "mesh_resolution_label": args.mesh_resolution_label,
        "shared_inlet_area_mapping": {
            "fluid_inlet_area_m2": fluid_inlet_area_m2,
            "solid_inlet_area_m2": solid_inlet_area_m2,
            "inlet_open_area_fraction": inlet_open_area_fraction,
            "mapping": "u_pore = u_in_source / inlet_open_area_fraction",
        },
        "existing_case_policy": (
            "validated_and_reused" if args.resume_existing else "new_or_explicit_overwrite"
        ),
        "cases": built,
        "physical_statement": (
            "The operating points, pressure, wall temperature and source-domain boundary types "
            "come from the cited paper. The source inlet-channel velocity is converted to the "
            "pore-opening boundary velocity using the resolved fluid and solid inlet areas so "
            "that the published volume flow is preserved. The local fine mesh is the current "
            "project crop and is not the source paper's full 12.5dp x 12.5dp x 10dp domain."
        ),
    }
    (output_root / "matrix_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
