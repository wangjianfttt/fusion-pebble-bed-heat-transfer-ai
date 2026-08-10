#!/usr/bin/env python3
"""Select unfinished steady P418 cases and preserve written iteration fields."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def normalized_points(rows: list[dict[str, object]]) -> np.ndarray:
    values = np.asarray(
        [
            [
                float(row["inlet_velocity_m_s"]),
                float(row["inlet_temperature_K"]),
                float(row["solid_heat_source_MW_m3"]),
            ]
            for row in rows
        ],
        dtype=float,
    )
    span = np.ptp(values, axis=0)
    span[span == 0.0] = 1.0
    return (values - values.min(axis=0)) / span


def maximin_indices(points: np.ndarray, count: int) -> list[int]:
    """Deterministic space-filling selection with a centre-nearest first point."""
    if count >= len(points):
        return list(range(len(points)))
    centre = np.full(points.shape[1], 0.5)
    selected = [int(np.argmin(np.sum((points - centre) ** 2, axis=1)))]
    remaining = set(range(len(points))) - set(selected)
    while len(selected) < count:
        selected_points = points[np.asarray(selected)]
        next_index = max(
            remaining,
            key=lambda i: (
                float(np.min(np.sum((points[i] - selected_points) ** 2, axis=1))),
                -i,
            ),
        )
        selected.append(next_index)
        remaining.remove(next_index)
    return selected


def replace_purge_write(control_dict: Path) -> None:
    text = control_dict.read_text(encoding="utf-8")
    updated, count = re.subn(r"(?m)^\s*purgeWrite\s+\d+\s*;", "purgeWrite 0;", text)
    if count != 1:
        raise ValueError(f"expected one purgeWrite entry in {control_dict}, found {count}")
    control_dict.write_text(updated, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()
    matrix_root = args.matrix_root.resolve()
    manifest = json.loads((matrix_root / "matrix_manifest.json").read_text(encoding="utf-8"))
    rows = [
        row
        for row in manifest["published_conditions"]
        if not (matrix_root / str(row["condition_id"]) / "formal_sample_complete.json").is_file()
    ]
    if len(rows) < args.count:
        raise ValueError(f"only {len(rows)} unfinished cases are available for {args.count} selections")
    indices = maximin_indices(normalized_points(rows), args.count)
    selected = [rows[index] for index in indices]
    plans = []
    for rank, row in enumerate(selected):
        condition_id = str(row["condition_id"])
        case = matrix_root / condition_id
        metadata = json.loads((case / "cht_smoke_metadata.json").read_text(encoding="utf-8"))
        write_interval = int(metadata["write_interval"])
        end_time = int(metadata["end_time"])
        snapshots = list(range(write_interval, end_time + 1, write_interval))
        replace_purge_write(case / "system/controlDict")
        plan = {
            "status": "preserve_p418_3d_steady_iteration_snapshots",
            "selection_rank": rank,
            "selection_method": "deterministic maximin over normalized published u, T_in and q coordinates",
            "condition_id": condition_id,
            "inlet_velocity_m_s": float(row["inlet_velocity_m_s"]),
            "inlet_temperature_K": float(row["inlet_temperature_K"]),
            "solid_heat_source_MW_m3": float(row["solid_heat_source_MW_m3"]),
            "steady_snapshot_iterations": snapshots,
            "solver_time_semantics": "steady_iteration_index",
            "physical_time_s": None,
            "source_title": manifest["source_title"],
            "source_doi": manifest["source_doi"],
            "new_physical_parameters": [],
            "note": (
                "Case selection and snapshot spacing are numerical data-design settings. "
                "The physical operating point is an exact published P418 matrix condition."
            ),
        }
        (case / "transient_snapshot_plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        plans.append(plan)
    summary = {
        "status": "configured_p418_3d_steady_iteration_snapshot_subset",
        "matrix_root": str(matrix_root),
        "requested_count": args.count,
        "selected_count": len(plans),
        "selection_method": "deterministic maximin over normalized published operating coordinates",
        "selected_conditions": [plan["condition_id"] for plan in plans],
        "new_physical_parameters": [],
    }
    output = matrix_root / "transient_snapshot_subset.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
