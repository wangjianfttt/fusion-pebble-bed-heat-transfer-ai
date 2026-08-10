#!/usr/bin/env python3
"""Build manuscript text for the 12 fixed-flow/fully-coupled trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


STATUS = "completed_p418_fixed_vs_fully_coupled_step_comparison"
SIGNALS = (
    ("pressure_drop_Pa", "Pressure drop", "Pa"),
    ("outlet_temperature_K", "Outlet temperature", "K"),
    ("maximum_solid_temperature_K", "Maximum solid temperature", "K"),
    (
        "volume_average_fluid_temperature_K",
        "Volume-average fluid temperature",
        "K",
    ),
    (
        "volume_average_solid_temperature_K",
        "Volume-average solid temperature",
        "K",
    ),
    ("cooling_wall_power_W", "Cooling-wall heat rate", "W"),
    ("signed_mass_residual_kg_s", "Signed mass residual", "kg s$^{-1}$"),
    ("net_outward_enthalpy_flow_W", "Net outward enthalpy flow", "W"),
)


def finite(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"invalid {name}: {value}")
    return result


def fmt(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 0.01 or abs(value) >= 1.0e4:
        return f"{value:.2e}"
    if abs(value) < 1.0:
        return f"{value:.3f}"
    if abs(value) < 100.0:
        return f"{value:.2f}"
    return f"{value:.1f}"


def load_and_validate(summary_path: Path, csv_path: Path) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != STATUS:
        raise ValueError(f"unexpected comparison status: {summary.get('status')}")
    if int(summary.get("sequence_count", -1)) != 12:
        raise ValueError("the main comparison must contain 12 trajectories")
    if tuple(summary.get("signals", [])) != tuple(signal for signal, _, _ in SIGNALS):
        raise ValueError("the main comparison does not contain the declared eight signals")
    if summary.get("new_physical_parameters") != []:
        raise ValueError("the comparison introduced a new physical parameter")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 12 * len(SIGNALS):
        raise ValueError("the comparison CSV must contain 12 rows per signal")
    keys = [(row.get("sequence_id"), row.get("signal")) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("the comparison CSV contains duplicate trajectory/signal rows")
    sequence_ids = {str(row.get("sequence_id")) for row in rows}
    if len(sequence_ids) != 12:
        raise ValueError("the comparison CSV does not contain 12 distinct trajectories")

    aggregate = summary.get("aggregate_by_signal")
    if not isinstance(aggregate, dict):
        raise ValueError("the aggregate comparison is missing")
    for signal, _, unit in SIGNALS:
        item = aggregate.get(signal)
        if not isinstance(item, dict):
            raise ValueError(f"aggregate result is missing for {signal}")
        if int(item.get("trajectory_count", -1)) != 12:
            raise ValueError(f"aggregate trajectory count is wrong for {signal}")
        if str(item.get("unit")) != unit.replace("$^{-1}$", "^-1"):
            raise ValueError(f"unit is wrong for {signal}")
        signal_rows = [row for row in rows if row["signal"] == signal]
        expected_median = sorted(finite(row["rmse"], f"{signal} rmse") for row in signal_rows)
        expected_median_value = 0.5 * (expected_median[5] + expected_median[6])
        if not math.isclose(
            finite(item.get("median_rmse"), f"{signal} median rmse"),
            expected_median_value,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise ValueError(f"aggregate median RMSE does not match CSV for {signal}")
        expected_absolute = max(
            finite(row["maximum_absolute_difference"], f"{signal} maximum difference")
            for row in signal_rows
        )
        if not math.isclose(
            finite(
                item.get("largest_absolute_difference"),
                f"{signal} largest absolute difference",
            ),
            expected_absolute,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise ValueError(f"aggregate absolute difference does not match CSV for {signal}")
        expected_normalized = max(
            finite(
                row["maximum_difference_over_fully_coupled_response_span"],
                f"{signal} normalized difference",
            )
            for row in signal_rows
        )
        if not math.isclose(
            finite(
                item.get("largest_difference_over_fully_coupled_response_span"),
                f"{signal} largest normalized difference",
            ),
            expected_normalized,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise ValueError(f"aggregate normalized difference does not match CSV for {signal}")
    return summary


def build_table(summary: dict) -> str:
    aggregate = summary["aggregate_by_signal"]
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{Differences between the fixed-hydrodynamics and fully coupled "
        r"three-dimensional reference histories over the same 12 thermal steps. "
        r"The response-normalized quantity divides the largest absolute difference "
        r"by the span of the corresponding fully coupled trajectory. No acceptance "
        r"percentage is imposed; dimensional values are retained because a small "
        r"response span can amplify the normalized quantity.}",
        r"\label{tab:fixed_fully_coupled_steps}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Quantity & Unit & Median RMSE & Largest absolute difference & "
        r"Largest difference / response span \\",
        r"\midrule",
    ]
    for signal, label, unit in SIGNALS:
        item = aggregate[signal]
        lines.append(
            f"{label} & {unit} & {fmt(finite(item['median_rmse'], signal))} & "
            f"{fmt(finite(item['largest_absolute_difference'], signal))} & "
            f"{fmt(100.0 * finite(item['largest_difference_over_fully_coupled_response_span'], signal))}\\% \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def build_text(summary: dict) -> str:
    aggregate = summary["aggregate_by_signal"]
    tmax = aggregate["maximum_solid_temperature_K"]
    tout = aggregate["outlet_temperature_K"]
    wall = aggregate["cooling_wall_power_W"]
    pressure = aggregate["pressure_drop_Pa"]
    return (
        "Across the same 12 source--target thermal steps, the fixed-hydrodynamics "
        "and fully coupled references gave largest absolute differences of "
        f"{fmt(finite(tmax['largest_absolute_difference'], 'maximum solid temperature'))} K "
        "in maximum solid temperature, "
        f"{fmt(finite(tout['largest_absolute_difference'], 'outlet temperature'))} K "
        "in outlet temperature, "
        f"{fmt(finite(wall['largest_absolute_difference'], 'cooling-wall heat'))} W "
        "in cooling-wall heat rate and "
        f"{fmt(finite(pressure['largest_absolute_difference'], 'pressure drop'))} Pa "
        "in pressure drop. Table~\\ref{tab:fixed_fully_coupled_steps} retains both "
        "the dimensional differences and their normalization by the fully coupled "
        "response span. The comparison therefore quantifies the effect of prescribed "
        "hydrodynamics without introducing a fitted percentage for deciding whether "
        "the two formulations agree."
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--table-output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    args = parser.parse_args()
    summary = load_and_validate(args.summary.resolve(), args.csv.resolve())
    args.table_output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.table_output.resolve().write_text(build_table(summary), encoding="utf-8")
    args.text_output.resolve().write_text(build_text(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "completed_p418_fixed_vs_fully_coupled_manuscript",
                "sequence_count": 12,
                "table": str(args.table_output.resolve()),
                "text": str(args.text_output.resolve()),
                "new_physical_parameters": [],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
