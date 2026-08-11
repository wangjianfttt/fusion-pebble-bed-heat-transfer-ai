#!/usr/bin/env python3
"""Compute conductive heat flow on every boundary of a solved Gmsh CHT case."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def set_patch_type(path: Path, patch_name: str, patch_type: str) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        rf"(?ms)(^\s*{re.escape(patch_name)}\s*\n\s*\{{.*?^\s*type\s+)[A-Za-z0-9_]+;"
    )
    updated, count = pattern.subn(rf"\g<1>{patch_type};", text, count=1)
    if count != 1:
        raise ValueError(f"cannot set patch type for {patch_name} in {path}")
    path.write_text(updated, encoding="utf-8")


def hardlink_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, copy_function=os.link)


def break_hardlink(path: Path) -> None:
    replacement = path.with_name(path.name + ".independent")
    shutil.copy2(path, replacement)
    replacement.replace(path)


def run_wall_heat_flux(case: Path, region: str, time_name: str) -> tuple[str, dict[str, float]]:
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
    pattern = re.compile(rf"for patch\s+(\S+)\s*=\s*{NUMBER},\s*{NUMBER},\s*({NUMBER}),")
    values = {patch: float(power) for patch, power in pattern.findall(completed.stdout)}
    if not values:
        raise ValueError(f"no boundary heat flow found for region {region}")
    return completed.stdout, values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--time", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    case = args.case.resolve()
    time_name = str(args.time)
    time_dir = case / time_name
    if not time_dir.is_dir():
        raise FileNotFoundError(time_dir)
    metadata = json.loads((case / "cht_smoke_metadata.json").read_text(encoding="utf-8"))
    inlet_patch = metadata["inlet_patch"]
    outlet_patch = metadata["outlet_patch"]
    fluid_interface_patch = metadata.get("fluid_solid_patch", "fluid_to_solid")
    solid_interface_patch = metadata.get("solid_fluid_patch", "solid_to_fluid")

    temp_root = Path(tempfile.mkdtemp(prefix="hccb_boundary_heat_", dir=case.parent))
    try:
        hardlink_tree(case / "constant", temp_root / "constant")
        hardlink_tree(case / "system", temp_root / "system")
        hardlink_tree(time_dir, temp_root / time_name)

        fluid_boundary = temp_root / "constant/fluid/polyMesh/boundary"
        break_hardlink(fluid_boundary)
        set_patch_type(fluid_boundary, inlet_patch, "wall")
        set_patch_type(fluid_boundary, outlet_patch, "wall")

        solid_boundary = temp_root / "constant/solid/polyMesh/boundary"
        break_hardlink(solid_boundary)
        for patch in (inlet_patch, outlet_patch):
            if re.search(rf"(?m)^\s*{re.escape(patch)}\s*$", solid_boundary.read_text()):
                set_patch_type(solid_boundary, patch, "wall")

        fluid_log, fluid = run_wall_heat_flux(temp_root, "fluid", time_name)
        solid_log, solid = run_wall_heat_flux(temp_root, "solid", time_name)
    finally:
        shutil.rmtree(temp_root)

    fluid_interface = fluid.get(fluid_interface_patch, 0.0)
    solid_interface = solid.get(solid_interface_patch, 0.0)
    result = {
        "case": str(case),
        "time": time_name,
        "sign_definition": "positive heat flow enters the selected region",
        "fluid_boundary_heat_flow_W": fluid,
        "solid_boundary_heat_flow_W": solid,
        "fluid_boundary_heat_flow_sum_W": sum(fluid.values()),
        "solid_boundary_heat_flow_sum_W": sum(solid.values()),
        "interface_pair_difference_W": fluid_interface + solid_interface,
        "fluid_interface_patch": fluid_interface_patch,
        "solid_interface_patch": solid_interface_patch,
        "foamPostProcess_completed": "FOAM FATAL" not in fluid_log + solid_log,
    }

    output = args.output.resolve() if args.output else case / f"boundary_heat_flows_{time_name}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
