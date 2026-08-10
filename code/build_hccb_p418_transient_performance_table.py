#!/usr/bin/env python3
"""Build the common transient heat-transfer performance table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


STRICT_SPLIT = "pair_disjoint_stress_test"
ENERGY_METRIC = "projection_aware_volume_weighted_energy_equation_normalized_RMSE"
METHODS = (
    ("dmdc", "Continuous-time DMDc", ""),
    ("graph_transformer_data_only", "Data-only graph--Transformer", ""),
    ("graph_transformer_energy_flux", "Physics graph--Transformer", ""),
    (
        "graph_transformer_factorized_energy_flux",
        "Factorized physics graph--Transformer",
        "",
    ),
    ("low_rank_residual_correction", "Physics graph--Transformer + POD", ""),
    (
        "diffusion_residual_correction",
        "Physics graph--Transformer + diffusion",
        "diffusion_refined_",
    ),
)
TEMPERATURE_METRIC = "solid_temperature_RMSE_K"
MAXIMUM_TEMPERATURE_METRIC = "solid_maximum_temperature_history_RMSE_K"
HOTSPOT_DISTANCE_METRIC = "solid_regional_hotspot_location_p95_error_m"
HOTSPOT_DEFICIT_METRIC = "solid_hotspot_target_temperature_deficit_p95_K"


def format_value(value: float) -> str:
    if value >= 100.0:
        return f"{value:.1f}"
    if value >= 10.0:
        return f"{value:.2f}"
    if value >= 0.1:
        return f"{value:.3f}"
    return f"{value:.2e}"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"transient model metrics are missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("transient model metrics are empty")
    return rows


def metric_value(
    rows: list[dict[str, str]],
    *,
    model: str,
    scope: str,
    metric: str,
    expected_unit: str,
) -> float:
    matches = [
        row
        for row in rows
        if row.get("split_name") == STRICT_SPLIT
        and row.get("model") == model
        and row.get("result_scope") == scope
        and row.get("data_role") == "test"
        and row.get("metric") == metric
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {metric} row for {model} in {STRICT_SPLIT}, found {len(matches)}"
        )
    row = matches[0]
    if row.get("unit") != expected_unit:
        raise ValueError(
            f"unexpected unit for {model} {metric}: {row.get('unit')}, expected {expected_unit}"
        )
    value = float(row["value"])
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"invalid {metric} for {model}: {row.get('value')}")
    return value


def build(metrics_csv: Path) -> tuple[str, list[dict[str, object]]]:
    rows = read_rows(metrics_csv)
    records: list[dict[str, object]] = []
    for model, label, prefix in METHODS:
        field_rmse = metric_value(
            rows,
            model=model,
            scope="regional_temperature_field",
            metric=f"{prefix}{TEMPERATURE_METRIC}",
            expected_unit="K",
        )
        maximum_temperature_rmse = metric_value(
            rows,
            model=model,
            scope="regional_temperature_field",
            metric=f"{prefix}{MAXIMUM_TEMPERATURE_METRIC}",
            expected_unit="K",
        )
        hotspot_p95_m = metric_value(
            rows,
            model=model,
            scope="regional_temperature_field",
            metric=f"{prefix}{HOTSPOT_DISTANCE_METRIC}",
            expected_unit="m",
        )
        hotspot_deficit_k = metric_value(
            rows,
            model=model,
            scope="regional_temperature_field",
            metric=f"{prefix}{HOTSPOT_DEFICIT_METRIC}",
            expected_unit="K",
        )
        energy_difference = metric_value(
            rows,
            model=model,
            scope="transient_energy_balance",
            metric=ENERGY_METRIC,
            expected_unit="dimensionless",
        )
        records.append(
            {
                "model": model,
                "label": label,
                "solid_temperature_field_RMSE_K": field_rmse,
                "solid_maximum_temperature_history_RMSE_K": maximum_temperature_rmse,
                "regional_hotspot_p95_distance_mm": hotspot_p95_m * 1000.0,
                "regional_hotspot_target_temperature_deficit_p95_K": hotspot_deficit_k,
                "projection_aware_energy_equation_normalized_RMSE": energy_difference,
            }
        )

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\caption{Transient heat-transfer prediction on the strict endpoint-pair split. Both steady endpoints and the complete trajectory are excluded from fitting. The hotspot distance is evaluated between regional centroids and does not represent a pebble-internal peak location. The hotspot temperature deficit is the reference maximum minus the reference temperature at the region selected as hottest by the model; it remains small when two nearly equal neighbouring regions exchange rank. The energy column is the projection-aware volume-weighted energy-equation normalized RMSE evaluated on the same independent trajectories.}",
        r"\label{tab:transient_performance}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & Field RMSE (K) & $T_{s,\max}$ RMSE (K) & Hotspot p95 (mm) & Hotspot deficit p95 (K) & Energy difference \\",
        r"\midrule",
    ]
    for record in records:
        lines.append(
            "{} & {} & {} & {} & {} & {} \\\\".format(
                record["label"],
                format_value(float(record["solid_temperature_field_RMSE_K"])),
                format_value(
                    float(record["solid_maximum_temperature_history_RMSE_K"])
                ),
                format_value(float(record["regional_hotspot_p95_distance_mm"])),
                format_value(float(record["regional_hotspot_target_temperature_deficit_p95_K"])),
                format_value(
                    float(
                        record[
                            "projection_aware_energy_equation_normalized_RMSE"
                        ]
                    )
                ),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines), records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    text, records = build(args.metrics_csv.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    payload = {
        "status": "complete_p418_transient_performance_table",
        "split_name": STRICT_SPLIT,
        "metrics_csv": str(args.metrics_csv.resolve()),
        "records": records,
        "tex": str(args.output.resolve()),
        "new_physical_parameters": [],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
