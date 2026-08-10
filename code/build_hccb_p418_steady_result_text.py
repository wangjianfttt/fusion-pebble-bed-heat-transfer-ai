#!/usr/bin/env python3
"""Generate steady-result prose from the five physical condition splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


METHODS = {
    "response_surface": "response surface",
    "pinn_data_only": "data-only PINN",
    "pinn": "physics-informed PINN",
    "graph": "graph operator",
    "transolver": "Physics-Attention operator",
}
SPLITS = (
    "interleaved_all_ranges",
    "temperature_extrapolation",
    "velocity_extrapolation",
    "heat_source_interpolation",
    "heat_source_extrapolation",
)
METRICS = (
    ("test_fluid_temperature_normalized_rmse", "fluid-temperature nRMSE", 1.0, ""),
    ("test_solid_temperature_normalized_rmse", "solid-temperature nRMSE", 1.0, ""),
    ("test_pressure_drop_p95_Pa", "pressure p95 error", 1.0, "Pa"),
    ("test_solid_maximum_temperature_p95_K", "solid maximum-$T$ p95 error", 1.0, "K"),
    ("test_cooling_wall_heat_over_generated_p95_percent", "wall-heat p95 error", 1.0, "\\%"),
    ("test_local_energy_l1_over_two_generated_power_mean", "regional energy difference", 100.0, "\\%"),
)


def finite(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"invalid steady metric {field}: {row[field]}")
    return value


def fmt(value: float) -> str:
    return f"{value:.3g}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-csv", required=True, type=Path)
    parser.add_argument("--thermal-regime-coverage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    source = args.comparison_csv.resolve()
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    lookup = {(row["architecture"], row["split"]): row for row in rows}
    expected = {(method, split) for method in METHODS for split in SPLITS}
    if set(lookup) != expected or len(rows) != len(expected):
        raise ValueError("steady comparison must contain five methods on five physical splits")

    coverage_source = args.thermal_regime_coverage.resolve()
    coverage_payload = json.loads(coverage_source.read_text(encoding="utf-8"))
    if coverage_payload.get("status") != "thermal_regime_split_coverage_complete":
        raise ValueError("thermal-regime coverage is incomplete")
    coverage_rows = coverage_payload.get("rows")
    if not isinstance(coverage_rows, list) or not coverage_rows:
        raise ValueError("thermal-regime coverage has no rows")
    coverage_lookup = {
        (str(row.get("split")), str(row.get("role"))): row
        for row in coverage_rows
    }
    required_coverage = {
        ("temperature_extrapolation", "train"),
        ("temperature_extrapolation", "validation"),
        ("temperature_extrapolation", "test"),
        ("interleaved_all_ranges", "test"),
    }
    if not required_coverage.issubset(coverage_lookup):
        raise ValueError("thermal-regime coverage lacks required split roles")
    for key, row in coverage_lookup.items():
        if not bool(row.get("coverage_complete")) or int(row.get("unknown_case_count", -1)) != 0:
            raise ValueError(f"thermal-regime coverage is incomplete for {key}")

    def regime_text(split: str, role: str, label: str) -> str:
        row = coverage_lookup[(split, role)]
        case_count = int(row["case_count"])
        wall_count = int(row["wall_to_fluid_count"])
        fluid_count = int(row["fluid_to_wall_count"])
        zero_count = int(row.get("zero_wall_heat_count", 0))
        if wall_count + fluid_count + zero_count != case_count:
            raise ValueError(f"wall-heat counts do not cover {split}/{role}")
        zero_text = f", {zero_count} zero-wall-heat" if zero_count else ""
        return (
            f"{label} {case_count} conditions ({wall_count} wall-to-fluid, "
            f"{fluid_count} fluid-to-wall{zero_text})"
        )

    temperature_regime_text = ", ".join(
        (
            regime_text("temperature_extrapolation", "train", "training"),
            regime_text("temperature_extrapolation", "validation", "validation"),
            regime_text("temperature_extrapolation", "test", "independent prediction"),
        )
    )
    interleaved_test_text = regime_text(
        "interleaved_all_ranges", "test", "The interleaved independent set contains"
    )

    worst_by_method: dict[str, dict[str, float]] = {}
    leaders = {}
    for method in METHODS:
        worst_by_method[method] = {
            field: max(finite(lookup[(method, split)], field) * scale for split in SPLITS)
            for field, _, scale, _ in METRICS
        }
    for field, quantity, _, unit in METRICS:
        method = min(METHODS, key=lambda name: worst_by_method[name][field])
        leaders[field] = {
            "method": method,
            "quantity": quantity,
            "value": worst_by_method[method][field],
            "unit": unit,
        }

    leader_text = []
    for field, quantity, _, unit in METRICS:
        record = leaders[field]
        suffix = f"~{unit}" if unit else ""
        leader_text.append(
            f"{quantity} {fmt(record['value'])}{suffix} ({METHODS[record['method']]})"
        )

    data_only = worst_by_method["pinn_data_only"]
    physics = worst_by_method["pinn"]
    comparison_fields = (
        ("test_solid_temperature_normalized_rmse", "solid-temperature nRMSE", ""),
        ("test_cooling_wall_heat_over_generated_p95_percent", "wall-heat p95 error", "\\%"),
        ("test_local_energy_l1_over_two_generated_power_mean", "regional energy difference", "\\%"),
    )
    paired_text = []
    for field, quantity, unit in comparison_fields:
        suffix = f"~{unit}" if unit else ""
        paired_text.append(
            f"{quantity} {fmt(data_only[field])}{suffix} versus {fmt(physics[field])}{suffix}"
        )

    lines = [
        (
            "Across the five complete-condition splits, the lowest worst-case values for the "
            "six separately evaluated quantities are "
            + ", ".join(leader_text)
            + ". Because different physical quantities may favour different architectures, these "
            "results are not combined into a scalar model score."
        ),
        "",
        (
            "With identical coordinate-network form, initialization, observations and optimization "
            "schedule, the data-only and physics-informed PINNs give, respectively, "
            + ", ".join(paired_text)
            + ". This paired comparison isolates the effect of the finite-volume mass, energy and "
            "interface-flux terms; all independent-condition values remain in Figure~\\ref{fig:steady_model_comparison}."
        ),
        "",
        (
            "The temperature-extrapolation split comprises "
            + temperature_regime_text
            + ". It therefore measures transfer across the signed wall-heat change, not only "
            "interpolation between nearby inlet temperatures. "
            + interleaved_test_text
            + "."
        ),
        "",
        (
            "The two heat-source splits contain one training source level, so its zero-variance "
            "input is zero in every role; they test stability at unseen source levels, not learned "
            "source sensitivity."
        ),
        "",
    ]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "status": "complete_p418_steady_manuscript_text",
        "comparison_csv": str(source),
        "thermal_regime_coverage": str(coverage_source),
        "temperature_extrapolation_regime_rows": {
            role: coverage_lookup[("temperature_extrapolation", role)]
            for role in ("train", "validation", "test")
        },
        "interleaved_test_regime_row": coverage_lookup[
            ("interleaved_all_ranges", "test")
        ],
        "split_count": len(SPLITS),
        "leaders_by_metric": leaders,
        "worst_by_method": worst_by_method,
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
