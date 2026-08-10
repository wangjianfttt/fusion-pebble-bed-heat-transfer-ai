#!/usr/bin/env python3
"""Compare exported boundary data with the original OpenFOAM case summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--sample-directory", default="training_sample_300_schema3")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, float | str]] = []
    for case in sorted(args.matrix_root.resolve().glob("u*_T*_q*")):
        solver_path = case / "cht_result_summary_300.json"
        boundary_path = case / args.sample_directory / "boundary_summary.json"
        if not solver_path.exists() or not boundary_path.exists():
            continue
        solver = json.loads(solver_path.read_text(encoding="utf-8"))
        boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "condition_id": case.name,
                "pressure_drop_solver_Pa": float(solver["flow"]["pressure_drop_Pa"]),
                "pressure_drop_export_Pa": float(
                    boundary["flow"]["pressure_drop_area_average_Pa"]
                ),
                "pressure_drop_absolute_difference_Pa": abs(
                    float(solver["flow"]["pressure_drop_Pa"])
                    - float(boundary["flow"]["pressure_drop_area_average_Pa"])
                ),
                "outlet_temperature_solver_K": float(
                    solver["temperature"]["outlet_average_K"]
                ),
                "outlet_temperature_export_K": float(
                    boundary["temperature"]["outlet_area_average_K"]
                ),
                "outlet_temperature_absolute_difference_K": abs(
                    float(solver["temperature"]["outlet_average_K"])
                    - float(boundary["temperature"]["outlet_area_average_K"])
                ),
                "inlet_mass_absolute_difference_kg_s": abs(
                    abs(float(solver["flow"]["inlet_mass_flow_kg_s"]))
                    - float(boundary["flow"]["inlet_mass_flow_kg_s"])
                ),
                "outlet_mass_absolute_difference_kg_s": abs(
                    float(solver["flow"]["outlet_mass_flow_kg_s"])
                    - float(boundary["flow"]["outlet_mass_flow_kg_s"])
                ),
                "outlet_reverse_to_outward_fraction": float(
                    boundary["flow"]["outlet_reverse_to_outward_fraction"]
                ),
            }
        )
    if not rows:
        raise FileNotFoundError("no matching solver and exported-boundary summaries")
    numeric = [key for key in rows[0] if key != "condition_id"]
    payload = {
        "status": "exported_boundaries_compared_with_openfoam_summaries",
        "case_count": len(rows),
        "maximum_absolute_differences": {
            key: max(float(row[key]) for row in rows)
            for key in numeric
            if "difference" in key
        },
        "maximum_outlet_reverse_to_outward_fraction": max(
            float(row["outlet_reverse_to_outward_fraction"]) for row in rows
        ),
        "cases": rows,
    }
    output_json = args.output_json.resolve()
    output_csv = args.output_csv.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
