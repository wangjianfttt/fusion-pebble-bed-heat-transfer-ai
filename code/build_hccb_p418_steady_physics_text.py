#!/usr/bin/env python3
"""Write a manuscript paragraph from the completed 60-case steady matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def values(rows: list[dict[str, str]], field: str) -> list[float]:
    result = [float(row[field]) for row in rows]
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"invalid or missing values for {field}")
    return result


def bounds(items: list[float]) -> tuple[float, float]:
    return min(items), max(items)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed_p418_case_physics_summarized":
        raise ValueError("steady physics summary is incomplete")
    if int(summary.get("completed_case_count", -1)) != 60:
        raise ValueError("the manuscript paragraph requires all 60 steady cases")
    checks = summary["physical_trend_checks"]
    if not checks.get("all_evaluated_checks_passed"):
        raise ValueError("one or more direct physical trend checks failed")

    temperature = read_rows(result_dir / "paired_temperature_response.csv")
    velocity = read_rows(result_dir / "paired_velocity_response.csv")
    source = read_rows(result_dir / "paired_heat_source_response.csv")
    reversal = read_rows(result_dir / "wall_heat_zero_crossings.csv")
    if (len(temperature), len(velocity), len(source), len(reversal)) != (15, 12, 20, 15):
        raise ValueError("paired-response tables are incomplete")

    temperature_steps = values(temperature, "inlet_temperature_step_K")
    outlet_temperature_steps = [
        response * step
        for response, step in zip(
            values(temperature, "outlet_temperature_response_K_per_K"),
            temperature_steps,
            strict=True,
        )
    ]
    solid_temperature_steps = [
        response * step
        for response, step in zip(
            values(temperature, "solid_maximum_response_K_per_K"),
            temperature_steps,
            strict=True,
        )
    ]
    pressure_temperature = [
        abs(value) for value in values(temperature, "pressure_drop_change_percent")
    ]
    pressure_velocity = values(velocity, "pressure_drop_change_percent")
    source_outlet = values(source, "outlet_temperature_response_K_per_MW_m3")
    reversal_temperature = values(
        reversal, "interpolated_zero_wall_heat_inlet_temperature_K"
    )

    reported = {
        "outlet_temperature_step_K": bounds(outlet_temperature_steps),
        "solid_temperature_step_K": bounds(solid_temperature_steps),
        "pressure_drop_temperature_reduction_percent": bounds(pressure_temperature),
        "pressure_drop_velocity_increase_percent": bounds(pressure_velocity),
        "outlet_temperature_source_response_K_per_MW_m3": bounds(source_outlet),
        "wall_heat_reversal_temperature_K": bounds(reversal_temperature),
        "maximum_relative_mass_difference": float(
            summary["maximum_relative_mass_difference"]
        ),
        "maximum_relative_energy_difference": float(
            summary["maximum_relative_energy_difference"]
        ),
    }

    outlet_low, outlet_high = reported["outlet_temperature_step_K"]
    solid_low, solid_high = reported["solid_temperature_step_K"]
    pressure_t_low, pressure_t_high = reported[
        "pressure_drop_temperature_reduction_percent"
    ]
    pressure_u_low, pressure_u_high = reported[
        "pressure_drop_velocity_increase_percent"
    ]
    source_low, source_high = reported[
        "outlet_temperature_source_response_K_per_MW_m3"
    ]
    reversal_low, reversal_high = reported["wall_heat_reversal_temperature_K"]

    text = (
        "All direct pairwise trend checks were satisfied across the complete "
        "$5\\times4\\times3$ matrix. Increasing the inlet temperature from "
        "\\SI{300}{K} to \\SI{900}{K} raised the outlet temperature by "
        f"\\SIrange{{{outlet_low:.1f}}}{{{outlet_high:.1f}}}{{K}} and the maximum "
        f"solid temperature by \\SIrange{{{solid_low:.1f}}}{{{solid_high:.1f}}}{{K}}, "
        "while reducing the pressure drop by "
        f"\\SIrange{{{pressure_t_low:.1f}}}{{{pressure_t_high:.1f}}}{{\\percent}}. "
        "Increasing the inlet velocity from \\SI{0.05}{m.s^{-1}} to "
        "\\SI{0.25}{m.s^{-1}} increased the pressure drop by "
        f"\\SIrange{{{pressure_u_low:.1f}}}{{{pressure_u_high:.1f}}}{{\\percent}}. "
        "At fixed velocity and inlet temperature, each additional "
        "\\SI{1}{MW.m^{-3}} of pebble heating increased the outlet temperature by "
        f"\\SIrange{{{source_low:.2f}}}{{{source_high:.2f}}}{{K}}. "
        "The signed cooling-wall heat changed direction at an interpolated inlet "
        f"temperature of \\SIrange{{{reversal_low:.1f}}}{{{reversal_high:.1f}}}{{K}} "
        "over the sampled velocity and heat-source combinations. The largest "
        "relative mass and energy differences over all fields were "
        f"\\num{{{reported['maximum_relative_mass_difference']:.3g}}} and "
        f"\\num{{{reported['maximum_relative_energy_difference']:.3g}}}, respectively.\n"
    )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")

    summary_output = args.summary_output.resolve()
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "completed_p418_steady_physics_manuscript_text",
        "source_summary": str(summary_path),
        "completed_case_count": 60,
        "reported_values": reported,
        "tex_output": str(output),
        "new_physical_parameters": [],
    }
    summary_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
