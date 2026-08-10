#!/usr/bin/env python3
"""Generate the manuscript steady accuracy/cost table from formal results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from plot_hccb_p418_steady_model_comparison import METHODS, SPLITS, read_formal_matrix


REQUIRED_COST_COLUMNS = (
    "model_parameter_count",
    "training_wall_time_s",
    "test_inference_s_per_case",
)


def finite_positive(row: dict[str, str], column: str, key: tuple[str, str]) -> float:
    raw = row.get(column, "")
    if raw == "":
        raise ValueError(f"missing {column} for {key}")
    value = float(raw)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"invalid {column} for {key}: {raw}")
    return value


def format_value(value: float, digits: int = 3) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 0.01 or abs(value) >= 10000.0:
        return f"{value:.2e}"
    return f"{value:.{digits}g}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    source = args.comparison_csv.resolve()
    lookup = read_formal_matrix(source)

    records: list[dict[str, object]] = []
    for method, (label, _, _) in METHODS.items():
        rows = [lookup[(method, split)] for split in SPLITS]
        parameters = [int(round(finite_positive(row, "model_parameter_count", (method, split))))
                      for row, split in zip(rows, SPLITS)]
        if len(set(parameters)) != 1:
            raise ValueError(f"{method} parameter count changes across physical splits: {parameters}")
        training = [finite_positive(row, "training_wall_time_s", (method, split))
                    for row, split in zip(rows, SPLITS)]
        inference = [finite_positive(row, "test_inference_s_per_case", (method, split))
                     for row, split in zip(rows, SPLITS)]
        records.append(
            {
                "architecture": method,
                "label": label,
                "worst_solid_temperature_normalized_rmse": max(
                    float(row["test_solid_temperature_normalized_rmse"]) for row in rows
                ),
                "worst_solid_maximum_temperature_p95_K": max(
                    float(row["test_solid_maximum_temperature_p95_K"]) for row in rows
                ),
                "worst_wall_heat_p95_percent": max(
                    float(row["test_cooling_wall_heat_over_generated_p95_percent"]) for row in rows
                ),
                "worst_regional_energy_difference_percent": 100.0 * max(
                    float(row["test_local_energy_l1_over_two_generated_power_mean"])
                    for row in rows
                ),
                "model_parameter_count": parameters[0],
                "median_training_minutes": float(np.median(training) / 60.0),
                "median_inference_ms_per_case": float(np.median(inference) * 1000.0),
            }
        )

    lines = [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.6pt}",
        "\\caption{Steady prediction accuracy and computational cost. Accuracy entries are the worst values across the five complete-condition splits in Figure~\\ref{fig:steady_model_comparison}. Training and inference times are medians across the same five runs. Times are measured values from the recorded computing device for each method and are not normalized to identical hardware.}",
        "\\label{tab:steady_performance}",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Method & Solid-$T$ nRMSE & Max-$T$ p95 (K) & Wall-heat p95 (\\%) & Energy diff. (\\%) & Params (M) & Train (min) & Infer (ms) \\\\",
        "\\midrule",
    ]
    for record in records:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                record["label"],
                format_value(float(record["worst_solid_temperature_normalized_rmse"])),
                format_value(float(record["worst_solid_maximum_temperature_p95_K"])),
                format_value(float(record["worst_wall_heat_p95_percent"])),
                format_value(float(record["worst_regional_energy_difference_percent"])),
                format_value(float(record["model_parameter_count"]) / 1.0e6),
                format_value(float(record["median_training_minutes"])),
                format_value(float(record["median_inference_ms_per_case"])),
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "status": "complete_formal_p418_steady_performance_table",
        "comparison_csv": str(source),
        "physical_split_count": len(SPLITS),
        "architectures": list(METHODS),
        "records": records,
        "tex": str(output),
        "new_physical_parameter_values_added": [],
    }
    summary_path = args.summary.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
