#!/usr/bin/env python3
"""Summarize P418 model errors separately for the two heat-flow regimes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ENGINEERING_METRICS = (
    ("pressure_drop_Pa", "pressure_drop"),
    ("outlet_temperature_K", "outlet_temperature"),
    ("solid_maximum_temperature_K", "solid_maximum_temperature"),
    ("cooling_wall_heat_into_fluid_W", "cooling_wall_heat"),
    ("solid_to_fluid_interphase_net_W", "solid_to_fluid_interphase_heat"),
)


def load_physical_conditions(path: Path) -> dict[str, dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no physical cases in {path}")
    conditions: dict[str, dict[str, object]] = {}
    for row in rows:
        condition_id = str(row["condition_id"])
        solid_minus_wall = float(row["solid_maximum_minus_cooling_wall_K"])
        wall_direction = str(row["cooling_wall_heat_direction"])
        if wall_direction not in {"wall_to_fluid", "fluid_to_wall", "zero"}:
            raise ValueError(
                f"unknown cooling-wall heat direction for {condition_id}: {wall_direction}"
            )
        conditions[condition_id] = {
            "condition_id": condition_id,
            "cooling_wall_heat_direction": wall_direction,
            "solid_temperature_relation": (
                "solid_maximum_above_wall"
                if solid_minus_wall > 0.0
                else "solid_maximum_at_or_below_wall"
            ),
            "solid_maximum_minus_cooling_wall_K": solid_minus_wall,
        }
    return conditions


def finite_values(cases: list[dict[str, object]], key: str) -> np.ndarray:
    values = np.asarray([float(case[key]) for case in cases], dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"missing or non-finite case values for {key}")
    return values


def engineering_values(cases: list[dict[str, object]], key: str) -> np.ndarray:
    values = np.asarray(
        [float(case["engineering_absolute_errors"][key]) for case in cases],
        dtype=float,
    )
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"invalid engineering errors for {key}")
    return values


def statistics(values: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.mean(values)),
        float(np.quantile(values, 0.95)),
        float(np.max(values)),
    )


def summarize_group(
    architecture: str,
    split: str,
    axis: str,
    regime: str,
    cases: list[dict[str, object]],
) -> dict[str, object]:
    row: dict[str, object] = {
        "architecture": architecture,
        "split": split,
        "classification_axis": axis,
        "thermal_regime": regime,
        "case_count": len(cases),
        "condition_ids": ";".join(str(case["condition_id"]) for case in cases),
    }
    for source_name, output_name in ENGINEERING_METRICS:
        values = engineering_values(cases, source_name)
        mean, p95, maximum = statistics(values)
        unit = "W" if source_name.endswith("_W") else "K" if source_name.endswith("_K") else "Pa"
        row[f"{output_name}_mae_{unit}"] = mean
        row[f"{output_name}_p95_{unit}"] = p95
        row[f"{output_name}_maximum_{unit}"] = maximum
    generated = finite_values(cases, "generated_power_W")
    if np.any(generated <= 0.0):
        raise ValueError("generated power must be positive")
    for source_name, output_name in (
        ("cooling_wall_heat_into_fluid_W", "cooling_wall_heat"),
        ("solid_to_fluid_interphase_net_W", "solid_to_fluid_interphase_heat"),
    ):
        relative = 100.0 * engineering_values(cases, source_name) / generated
        mean, p95, maximum = statistics(relative)
        row[f"{output_name}_error_mean_percent_generated"] = mean
        row[f"{output_name}_error_p95_percent_generated"] = p95
        row[f"{output_name}_error_maximum_percent_generated"] = maximum
    for source_name, output_name in (
        ("global_mass_imbalance_over_inlet", "global_mass_imbalance_over_inlet"),
        (
            "global_energy_imbalance_over_generated_power",
            "global_energy_imbalance_over_generated_power",
        ),
    ):
        values = finite_values(cases, source_name)
        mean, p95, maximum = statistics(values)
        row[f"{output_name}_mean"] = mean
        row[f"{output_name}_p95"] = p95
        row[f"{output_name}_maximum"] = maximum
    return row


def summarize_payload(
    payload: dict[str, object],
    physical_conditions: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    architecture = str(payload.get("architecture", ""))
    split = str(payload.get("split_name", ""))
    evaluations = payload.get("evaluations")
    if not architecture or not split or not isinstance(evaluations, dict):
        raise ValueError("model summary lacks architecture, split or evaluations")
    test = evaluations.get("test")
    if not isinstance(test, dict) or not isinstance(test.get("cases"), list):
        raise ValueError("model summary has no test cases")
    cases = test["cases"]
    if not cases:
        raise ValueError("model summary has an empty test set")
    decorated: list[dict[str, object]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("test case must be a mapping")
        condition_id = str(case.get("condition_id", ""))
        if condition_id not in physical_conditions:
            raise ValueError(
                f"physical classification is missing for test case {condition_id}"
            )
        decorated.append({**case, **physical_conditions[condition_id]})
    rows: list[dict[str, object]] = []
    for axis in ("cooling_wall_heat_direction", "solid_temperature_relation"):
        groups: dict[str, list[dict[str, object]]] = {}
        for case in decorated:
            groups.setdefault(str(case[axis]), []).append(case)
        for regime, members in sorted(groups.items()):
            rows.append(
                summarize_group(architecture, split, axis, regime, members)
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no thermal-regime rows to write")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-csv", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--result-prefix", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--architectures", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    physical = load_physical_conditions(args.physical_csv)
    rows: list[dict[str, object]] = []
    source_summaries: list[str] = []
    for split in args.splits:
        for architecture in args.architectures:
            summary = args.results_root / (
                f"{args.result_prefix}_{architecture}_{split}_{args.epochs}epoch/summary.json"
            )
            if not summary.is_file():
                raise FileNotFoundError(summary)
            payload = json.loads(summary.read_text(encoding="utf-8"))
            rows.extend(summarize_payload(payload, physical))
            source_summaries.append(str(summary.resolve()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "thermal_regime_model_errors.csv"
    json_path = args.output_dir / "thermal_regime_model_errors.json"
    write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(
            {
                "status": "thermal_regime_model_errors_complete",
                "physical_classification_source": str(args.physical_csv.resolve()),
                "source_model_summaries": source_summaries,
                "row_count": len(rows),
                "rows": rows,
                "interpretation": (
                    "Errors are reported separately for the wall-to-fluid and "
                    "fluid-to-wall cases and for the solid-maximum temperature "
                    "relative to the wall. Different physical units are not merged."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
