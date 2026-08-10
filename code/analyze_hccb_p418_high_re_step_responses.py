#!/usr/bin/env python3
"""Summarize the six independent high-Re P418 fixed-flow step responses."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SIGNALS = (
    "outlet_temperature_K",
    "maximum_solid_temperature_K",
    "volume_average_fluid_temperature_K",
    "volume_average_solid_temperature_K",
    "cooling_wall_power_W",
)


def finite_float(value: str, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def crossing_time(
    time_s: list[float],
    values: list[float],
    fraction: float,
) -> float:
    initial = values[0]
    change = values[-1] - initial
    if change == 0.0:
        return 0.0
    target = initial + fraction * change
    for time_value, value in zip(time_s, values):
        if (change > 0.0 and value >= target) or (
            change < 0.0 and value <= target
        ):
            return time_value
    raise ValueError(f"response never reaches {fraction:.0%} of its final change")


def read_case(case: Path, expected_points: int) -> dict:
    csv_path = case / "results/hccb_p418_transient_observables_long.csv"
    summary_path = case / "results/summary.json"
    marker_path = case / "cloud_sequence_complete.json"
    if not all(path.is_file() for path in (csv_path, summary_path, marker_path)):
        raise FileNotFoundError(f"incomplete recovered result: {case}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        summary.get("completed_case_count") != 1
        or summary.get("maximum_time_points") != expected_points
        or (
            marker.get("status")
            != "completed_p418_high_re_independent_fixed_flow_sequence"
            and marker.get("solver_finished") is not True
        )
    ):
        raise ValueError(f"completion record is not valid: {case.name}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_points:
        raise ValueError(
            f"{case.name} has {len(rows)} points, expected {expected_points}"
        )
    time_s = [finite_float(row["time_s"], "time_s") for row in rows]
    if time_s[0] != 0.0 or time_s[-1] != 300.0:
        raise ValueError(f"{case.name} does not cover 0--300 s")
    if any(right <= left for left, right in zip(time_s, time_s[1:])):
        raise ValueError(f"{case.name} time axis is not strictly increasing")

    metrics = {}
    for signal in SIGNALS:
        values = [finite_float(row[signal], signal) for row in rows]
        metrics[signal] = {
            "initial": values[0],
            "final": values[-1],
            "change": values[-1] - values[0],
            "t50_s": crossing_time(time_s, values, 0.5),
            "t90_s": crossing_time(time_s, values, 0.9),
        }
    mass_residual = max(
        abs(finite_float(row["signed_mass_residual_kg_s"], "mass residual"))
        for row in rows
    )
    first = rows[0]
    return {
        "sequence_id": case.name,
        "source": {
            "inlet_velocity_m_s": finite_float(
                first["source_inlet_velocity_m_s"], "source velocity"
            ),
            "inlet_temperature_K": finite_float(
                first["source_inlet_temperature_K"], "source temperature"
            ),
            "solid_heat_source_MW_m3": finite_float(
                first["source_solid_heat_source_MW_m3"], "source heat source"
            ),
        },
        "target": {
            "inlet_velocity_m_s": finite_float(
                first["target_inlet_velocity_m_s"], "target velocity"
            ),
            "inlet_temperature_K": finite_float(
                first["target_inlet_temperature_K"], "target temperature"
            ),
            "solid_heat_source_MW_m3": finite_float(
                first["target_solid_heat_source_MW_m3"], "target heat source"
            ),
        },
        "time_point_count": len(rows),
        "maximum_absolute_mass_residual_kg_s": mass_residual,
        "responses": metrics,
    }


def build_summary(input_root: Path, expected_points: int) -> dict:
    cases_root = input_root / "by_sequence"
    if not cases_root.is_dir():
        raise FileNotFoundError(f"missing recovered by-sequence root: {cases_root}")
    cases = [
        read_case(case, expected_points)
        for case in sorted(cases_root.iterdir())
        if case.is_dir()
    ]
    if len(cases) != 6:
        raise ValueError(f"expected six independent curves, found {len(cases)}")

    amplitudes = {
        signal: {
            case["sequence_id"]: abs(case["responses"][signal]["change"])
            for case in cases
        }
        for signal in SIGNALS
    }
    outlet_amplitudes = amplitudes["outlet_temperature_K"]
    ranked_outlet_response = sorted(
        outlet_amplitudes,
        key=outlet_amplitudes.get,
        reverse=True,
    )
    return {
        "status": "completed_p418_high_re_step_response_analysis",
        "scientific_scope": (
            "Independent fixed-flow OpenFOAM step responses at the high-velocity "
            "boundary; these curves are not used for training, normalization, "
            "model selection, or loss-weight selection."
        ),
        "curve_count": len(cases),
        "points_per_curve": expected_points,
        "time_range_s": [0.0, 300.0],
        "cases": cases,
        "outlet_temperature_response_ranking": ranked_outlet_response,
        "new_physical_parameters": [],
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "casewise_response_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "sequence_id",
            "signal",
            "initial",
            "final",
            "change",
            "t50_s",
            "t90_s",
            "maximum_absolute_mass_residual_kg_s",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in summary["cases"]:
            for signal, response in case["responses"].items():
                writer.writerow(
                    {
                        "sequence_id": case["sequence_id"],
                        "signal": signal,
                        **response,
                        "maximum_absolute_mass_residual_kg_s": case[
                            "maximum_absolute_mass_residual_kg_s"
                        ],
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-points", type=int, default=16401)
    args = parser.parse_args()
    summary = build_summary(args.input_root.resolve(), args.expected_points)
    write_outputs(summary, args.output_dir.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
