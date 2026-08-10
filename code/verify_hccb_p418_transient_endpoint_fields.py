#!/usr/bin/env python3
"""Verify the minimal converged fields needed to initialise P418 thermal steps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_marker(case: Path) -> dict[str, object]:
    marker_path = case / "formal_sample_complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError(marker_path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("solver_finished") is not True:
        raise ValueError(f"steady endpoint is unfinished: {case.name}")
    if marker.get("solver_time_semantics", "steady_iteration_index") != "steady_iteration_index":
        raise ValueError(f"unexpected endpoint time semantics: {case.name}")
    iteration = marker.get(
        "reported_iteration",
        marker.get("steady_iteration_end", marker.get("time")),
    )
    if int(float(iteration)) != 200:
        raise ValueError(f"steady endpoint is not iteration 200: {case.name}")
    for key in ("relative_mass_difference", "relative_energy_difference"):
        value = marker.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{key} is missing or non-finite: {case.name}")
    return marker


def require_files(case: Path, relative_paths: list[str]) -> list[dict[str, object]]:
    rows = []
    for relative in relative_paths:
        path = case / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size <= 0:
            raise ValueError(f"empty endpoint file: {path}")
        rows.append({"path": relative, "bytes": path.stat().st_size})
    return rows


def verify_sequence(matrix_root: Path, sequence: dict[str, object]) -> dict[str, object]:
    source = matrix_root / str(sequence["source_condition_id"])
    target = matrix_root / str(sequence["target_condition_id"])
    source_marker = read_marker(source)
    target_marker = read_marker(target)
    source_time = str(source_marker["time"])
    target_time = str(target_marker["time"])

    source_files = require_files(
        source,
        [
            "cht_smoke_metadata.json",
            f"{source_time}/fluid/T",
            f"{source_time}/solid/T",
        ],
    )
    target_files = require_files(
        target,
        [
            "cht_smoke_metadata.json",
            "0/fluid/U",
            "0/fluid/T",
            "0/fluid/p",
            "0/fluid/p_rgh",
            "0/solid/T",
            "system/controlDict",
            "system/decomposeParDict",
            "system/fvSolution",
            "system/fluid/fvSchemes",
            "system/fluid/fvSolution",
            "system/solid/fvSchemes",
            "system/solid/fvSolution",
            f"{target_time}/fluid/U",
            f"{target_time}/fluid/p",
            f"{target_time}/fluid/p_rgh",
            f"{target_time}/fluid/phi",
        ],
    )
    for region in ("fluid", "solid"):
        poly_mesh = target / "constant" / region / "polyMesh"
        if not poly_mesh.is_dir():
            raise FileNotFoundError(poly_mesh)
        target_files.extend(
            require_files(
                target,
                [
                    f"constant/{region}/polyMesh/points",
                    f"constant/{region}/polyMesh/faces",
                    f"constant/{region}/polyMesh/owner",
                    f"constant/{region}/polyMesh/boundary",
                ],
            )
        )

    return {
        "sequence_id": sequence["sequence_id"],
        "source_condition_id": source.name,
        "target_condition_id": target.name,
        "source_iteration": int(float(source_marker["time"])),
        "target_iteration": int(float(target_marker["time"])),
        "source_required_files": source_files,
        "target_required_files": target_files,
        "new_physical_parameters": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matrix_root = args.matrix_root.resolve()
    plan = json.loads(args.plan.resolve().read_text(encoding="utf-8"))
    rows = [verify_sequence(matrix_root, sequence) for sequence in plan["sequences"]]
    payload = {
        "status": "p418_transient_endpoint_fields_verified",
        "sequence_count": len(rows),
        "sequences": rows,
        "new_physical_parameters": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
