#!/usr/bin/env python3
"""Build a compact original-cell prediction table from the formal comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


LABELS = {
    "response_surface": "Response surface",
    "pinn_data_only": "Data-only PINN",
    "pinn": "Physics-informed PINN",
    "graph_operator": "Graph operator",
    "transolver": "Physics-Attention",
}


def value(row: dict, key: str) -> float:
    result = float(row[key])
    if not math.isfinite(result):
        raise ValueError(f"non-finite native-cell metric: {key}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    source = args.comparison_summary.resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("status") != "native_cell_model_comparison_ready":
        raise ValueError("native-cell model comparison is incomplete")
    if payload.get("new_physical_parameters") != []:
        raise ValueError("native-cell comparison unexpectedly adds physical parameters")
    by_model = {str(row["model"]): row for row in payload.get("rows", [])}
    missing = [model for model in LABELS if model not in by_model]
    if missing:
        raise ValueError(f"native-cell comparison is missing models: {missing}")

    rendered = [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\small",
        "\\caption{Temperature prediction after reconstruction on the original \\OpenFOAM{} cells for the main complete-condition split. The bounded linear reconstruction is fixed by regional geometry and neighbour extrema and introduces no fitted physical parameter. Values are means over the independent conditions. The total RMSE includes both regional-representation loss and model prediction error.}",
        "\\label{tab:native_cell_prediction}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method & Fluid total RMSE (K) & Solid total RMSE (K) & Solid max-$T$ error (K) & Hotspot shift ($d_p$) \\\\",
        "\\midrule",
    ]
    records = []
    for model, label in LABELS.items():
        row = by_model[model]
        record = {
            "model": model,
            "label": label,
            "fluid_total_rmse_K": value(row, "fluid_limited_native_total_rmse_K_mean"),
            "solid_total_rmse_K": value(row, "solid_limited_native_total_rmse_K_mean"),
            "solid_max_temperature_error_K": value(
                row, "limited_predicted_solid_max_temperature_error_K_mean"
            ),
            "hotspot_shift_dp": value(row, "limited_predicted_hotspot_distance_dp_mean"),
        }
        records.append(record)
        rendered.append(
            f"{label} & {record['fluid_total_rmse_K']:.3g} & "
            f"{record['solid_total_rmse_K']:.3g} & "
            f"{record['solid_max_temperature_error_K']:.3g} & "
            f"{record['hotspot_shift_dp']:.3g} \\\\"
        )
    rendered.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rendered), encoding="utf-8")
    summary = {
        "status": "complete_p418_native_cell_model_table",
        "comparison_summary": str(source),
        "records": records,
        "tex": str(output),
        "new_physical_parameters": [],
    }
    summary_path = args.summary.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
