#!/usr/bin/env python3
"""Export per-face OpenFOAM wall heat flux for the five P418 CHT cases.

The solved cases already contain conservative mass flow and integral energy
checks.  This script re-runs OpenFOAM's own ``wallHeatFlux`` function object in
a temporary hard-linked case so inlet and outlet conductive heat flow are also
written per face.  It does not introduce or fit a physical parameter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np

from compute_hccb_gmsh_boundary_heat_flows import break_hardlink, set_patch_type
from openfoam_ascii_field import read_openfoam_ascii_field


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hardlink_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, copy_function=os.link)


def run_wall_heat_flux(case: Path, region: str, time_name: str) -> str:
    completed = subprocess.run(
        [
            "foamPostProcess",
            "-case",
            str(case),
            "-solver",
            region,
            "-region",
            region,
            "-func",
            "wallHeatFlux",
            "-time",
            time_name,
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout


def field_boundary_array(
    path: Path,
    *,
    cell_count: int,
    patch_names: list[str],
    patch_sizes: dict[str, int],
) -> np.ndarray:
    field = read_openfoam_ascii_field(
        path,
        internal_count=cell_count,
        patch_sizes=patch_sizes,
    )
    values: list[np.ndarray] = []
    for name in patch_names:
        patch_value = field.boundary_value[name]
        if patch_value is None:
            patch_value = np.zeros((patch_sizes[name], 1), dtype=np.float64)
        values.append(patch_value[:, 0])
    return np.concatenate(values)


def logged_patch_integrals(log: str) -> dict[str, float]:
    pattern = re.compile(
        rf"for patch\s+(\S+)\s*=\s*{NUMBER},\s*{NUMBER},\s*({NUMBER}),"
    )
    return {name: float(value) for name, value in pattern.findall(log)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--time", default="300")
    parser.add_argument("--time-from-completion-marker", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset_path = args.dataset_index.resolve()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    expected_case_count = int(dataset.get("case_count", len(dataset["conditions"])))
    if expected_case_count <= 0 or expected_case_count != len(dataset["conditions"]):
        raise ValueError("dataset case_count does not match its condition records")
    dataset_root = dataset_path.parent
    with np.load(dataset_root / dataset["shared_topology_file"], allow_pickle=False) as loaded:
        topology = {name: loaded[name] for name in loaded.files}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fluid_names = list(dataset["boundary_patch_names"]["fluid"])
    solid_names = list(dataset["boundary_patch_names"]["solid"])
    fluid_patch = topology["fluid_boundary_face_patch"].astype(np.int64)
    solid_patch = topology["solid_boundary_face_patch"].astype(np.int64)
    fluid_sizes = {
        name: int(np.count_nonzero(fluid_patch == index))
        for index, name in enumerate(fluid_names)
    }
    solid_sizes = {
        name: int(np.count_nonzero(solid_patch == index))
        for index, name in enumerate(solid_names)
    }
    fluid_area = topology["fluid_boundary_face_area_m2"].astype(np.float64)
    solid_area = topology["solid_boundary_face_area_m2"].astype(np.float64)
    solid_volume = float(np.sum(topology["solid_cell_volume_m3"]))

    fluid_targets: list[np.ndarray] = []
    solid_targets: list[np.ndarray] = []
    reports: list[dict[str, object]] = []
    for record in dataset["conditions"]:
        case_id = str(record["condition_id"])
        case = args.case_root.resolve() / case_id
        if not case.is_dir():
            raise FileNotFoundError(case)
        if args.time_from_completion_marker:
            marker = json.loads((case / "formal_sample_complete.json").read_text(encoding="utf-8"))
            time_name = str(marker["time"])
        else:
            time_name = str(args.time)
        time_dir = case / time_name
        if not time_dir.is_dir():
            raise FileNotFoundError(time_dir)
        staging = Path(tempfile.mkdtemp(prefix=f"p418_heat_{case_id}_", dir=output))
        try:
            hardlink_tree(case / "constant", staging / "constant")
            hardlink_tree(case / "system", staging / "system")
            hardlink_tree(time_dir, staging / time_name)
            for region in ("fluid", "solid"):
                boundary = staging / f"constant/{region}/polyMesh/boundary"
                break_hardlink(boundary)
                boundary_text = boundary.read_text(encoding="utf-8", errors="replace")
                # OpenFOAM's wallHeatFlux function only visits wall patches.
                # The coupled interface must remain mappedWall so the region
                # thermophysical model can be constructed; only the inlet and
                # outlet are temporarily retyped, exactly as in the solved-case
                # heat-balance post-processing route.
                for patch_name in ("inlet", "outlet"):
                    if re.search(rf"(?m)^\s*{re.escape(patch_name)}\s*$", boundary_text):
                        set_patch_type(boundary, patch_name, "wall")
                        boundary_text = boundary.read_text(encoding="utf-8", errors="replace")
            fluid_log = run_wall_heat_flux(staging, "fluid", time_name)
            solid_log = run_wall_heat_flux(staging, "solid", time_name)
            fluid = field_boundary_array(
                staging / time_name / "fluid/wallHeatFlux",
                cell_count=len(topology["fluid_cell_volume_m3"]),
                patch_names=fluid_names,
                patch_sizes=fluid_sizes,
            )
            solid = field_boundary_array(
                staging / time_name / "solid/wallHeatFlux",
                cell_count=len(topology["solid_cell_volume_m3"]),
                patch_names=solid_names,
                patch_sizes=solid_sizes,
            )
            fluid_integral = {
                name: float(np.sum(fluid[fluid_patch == index] * fluid_area[fluid_patch == index]))
                for index, name in enumerate(fluid_names)
            }
            solid_integral = {
                name: float(np.sum(solid[solid_patch == index] * solid_area[solid_patch == index]))
                for index, name in enumerate(solid_names)
            }
            fluid_log_integral = logged_patch_integrals(fluid_log)
            solid_log_integral = logged_patch_integrals(solid_log)
            max_log_difference = max(
                [
                    abs(fluid_integral[name] - fluid_log_integral.get(name, fluid_integral[name]))
                    for name in fluid_names
                ]
                + [
                    abs(solid_integral[name] - solid_log_integral.get(name, solid_integral[name]))
                    for name in solid_names
                ]
            )
            generated = float(record["solid_heat_source_W_m3"]) * solid_volume
            combined_conductive = float(sum(fluid_integral.values()) + sum(solid_integral.values()))
            reports.append(
                {
                    "condition_id": case_id,
                    "source_final_time_s": float(time_name),
                    "generated_power_W": generated,
                    "fluid_boundary_heat_flow_into_region_W": fluid_integral,
                    "solid_boundary_heat_flow_into_region_W": solid_integral,
                    "interface_pair_difference_W": (
                        fluid_integral["fluid_to_solid"]
                        + solid_integral["solid_to_fluid"]
                    ),
                    "solid_balance_relative": abs(
                        generated + sum(solid_integral.values())
                    ) / generated,
                    "maximum_field_vs_log_integral_difference_W": max_log_difference,
                    "combined_conductive_sum_W": combined_conductive,
                }
            )
            fluid_targets.append(fluid)
            solid_targets.append(solid)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    target_path = output / "boundary_heat_flux_targets.npz"
    np.savez_compressed(
        target_path,
        condition_id=np.asarray([record["condition_id"] for record in dataset["conditions"]]),
        fluid_boundary_heat_flux_into_region_W_m2=np.stack(fluid_targets),
        solid_boundary_heat_flux_into_region_W_m2=np.stack(solid_targets),
    )
    checks = {
        "all_dataset_cases_are_present": len(reports) == expected_case_count,
        "all_values_finite": bool(
            np.all(np.isfinite(fluid_targets)) and np.all(np.isfinite(solid_targets))
        ),
        "field_integrals_match_openfoam_log": max(
            float(row["maximum_field_vs_log_integral_difference_W"]) for row in reports
        ) < 1.0e-8,
        "solid_generated_heat_is_balanced": max(
            float(row["solid_balance_relative"]) for row in reports
        ) < 1.0e-5,
    }
    summary = {
        "status": "p418_boundary_heat_flux_targets_ready" if all(checks.values()) else "failed",
        "checks": checks,
        "cases": reports,
        "source_dataset_sha256": sha256(dataset_path),
        "target_file": target_path.name,
        "target_sha256": sha256(target_path),
        "method": "OpenFOAM-13 wallHeatFlux evaluated per boundary face",
        "sign_definition": "positive heat flux enters the selected region",
        "new_physical_parameters": [],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
