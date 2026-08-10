#!/usr/bin/env python3
"""Write the completed/pending P418 case list for cloud submission."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build(matrix_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    manifest_path = matrix_root / "matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    conditions = manifest.get("published_conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("matrix_manifest.json has no published_conditions")

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for order, condition in enumerate(conditions, start=1):
        condition_id = str(condition["condition_id"])
        if condition_id in seen:
            raise ValueError(f"duplicate condition_id: {condition_id}")
        seen.add(condition_id)
        case_dir = matrix_root / condition_id
        if not case_dir.is_dir():
            raise FileNotFoundError(case_dir)
        complete = (case_dir / "formal_sample_complete.json").is_file()
        rows.append(
            {
                "order": order,
                "condition_id": condition_id,
                "inlet_velocity_m_s": float(condition["inlet_velocity_m_s"]),
                "inlet_temperature_K": float(condition["inlet_temperature_K"]),
                "solid_heat_source_MW_m3": float(
                    condition["solid_heat_source_MW_m3"]
                ),
                "workstation_status": "completed" if complete else "pending_cloud",
                "submit_to_cloud": "no" if complete else "yes",
            }
        )

    completed = sum(row["workstation_status"] == "completed" for row in rows)
    summary: dict[str, object] = {
        "status": "P418 cloud case list",
        "matrix_root": str(matrix_root.resolve()),
        "total_cases": len(rows),
        "completed_on_workstation": completed,
        "pending_for_cloud": len(rows) - completed,
        "pending_condition_ids": [
            row["condition_id"]
            for row in rows
            if row["workstation_status"] == "pending_cloud"
        ],
    }
    return rows, summary


def write(rows: list[dict[str, object]], summary: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with (output_dir / "cloud_case_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "cloud_case_matrix_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "pending_case_ids.txt").write_text(
        "\n".join(map(str, summary["pending_condition_ids"])) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = build(args.matrix_root.resolve())
    write(rows, summary, args.output_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
