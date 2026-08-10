#!/usr/bin/env python3
"""Relate the P418 transient output cadence to literature thermal properties."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


REQUIRED_IDS = ("P048", "P092", "P403", "P418", "P429", "P430")
VALUE_COLUMN = "采用值或关系式"


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {row["parameter_id"]: row for row in csv.DictReader(handle)}
    missing = sorted(set(REQUIRED_IDS).difference(rows))
    if missing:
        raise ValueError(f"missing literature parameters: {missing}")
    return rows


def coefficients(value: str) -> tuple[float, float, float]:
    match = re.fullmatch(
        r"Cp_molar=([0-9.]+)\+([0-9.]+)\*T_K\+([0-9.]+)/T_K\^2", value
    )
    if not match:
        raise ValueError(f"cannot parse P429 heat-capacity relation: {value}")
    return tuple(float(item) for item in match.groups())


def molar_mass_kg_mol(value: str) -> float:
    match = re.search(r"M_Li4SiO4=([0-9.]+)", value)
    if not match:
        raise ValueError(f"cannot parse P430 molar mass: {value}")
    return float(match.group(1)) / 1000.0


def analyze(source: Path, velocity_summary: Path | None = None) -> dict[str, object]:
    rows = load_rows(source)
    diameter_m = float(rows["P048"][VALUE_COLUMN]) / 1000.0
    conductivity = float(rows["P092"][VALUE_COLUMN])
    density = float(rows["P403"][VALUE_COLUMN])
    a, b, c = coefficients(rows["P429"][VALUE_COLUMN])
    molar_mass = molar_mass_kg_mol(rows["P430"][VALUE_COLUMN])
    temperatures = (300.0, 500.0, 700.0, 900.0)
    table = []
    for temperature in temperatures:
        cp_molar = a + b * temperature + c / temperature**2
        cp_mass = cp_molar / molar_mass
        diffusivity = conductivity / (density * cp_mass)
        radial_scale = (0.5 * diameter_m) ** 2 / diffusivity
        table.append(
            {
                "temperature_K": temperature,
                "heat_capacity_J_kg_K": cp_mass,
                "thermal_diffusivity_m2_s": diffusivity,
                "particle_radial_diffusion_scale_s": radial_scale,
            }
        )
    maximum_scale = max(row["particle_radial_diffusion_scale_s"] for row in table)
    minimum_scale = min(row["particle_radial_diffusion_scale_s"] for row in table)
    if velocity_summary is None:
        velocity_summary = (
            Path(__file__).resolve().parents[1]
            / "results/hccb_p418_velocity_step_time_scales/summary.json"
        )
    velocity = json.loads(velocity_summary.resolve().read_text(encoding="utf-8"))
    crossing_min = float(velocity["minimum_domain_crossing_time_s"])
    crossing_max = float(velocity["maximum_domain_crossing_time_s"])
    low_velocity = min(float(value) for value in velocity["published_inlet_velocities_m_s"])
    high_velocity = max(float(value) for value in velocity["published_inlet_velocities_m_s"])
    low_p95_turnover = float(velocity["cell_outflow_turnover_rate_per_s"]["p95"])
    projected_high_p95_turnover = low_p95_turnover * high_velocity / low_velocity
    time_step_schedule = [
        {"start_s": 0.0, "end_s": 0.1, "delta_t_s": 1.0e-5},
        {"start_s": 0.1, "end_s": 1.0, "delta_t_s": 5.0e-4},
        {"start_s": 1.0, "end_s": 25.0, "delta_t_s": 1.0e-2},
        {"start_s": 25.0, "end_s": 300.0, "delta_t_s": 1.25e-1},
    ]
    output_schedule = [
        {"start_s": 0.0, "end_s": 0.1, "interval_s": 5.0e-3},
        {"start_s": 0.1, "end_s": 1.0, "interval_s": 1.0e-1},
        {"start_s": 1.0, "end_s": 5.0, "interval_s": 4.0e-1},
        {"start_s": 5.0, "end_s": 25.0, "interval_s": 4.0},
        {"start_s": 25.0, "end_s": 300.0, "interval_s": 25.0},
    ]
    return {
        "status": "p418_transient_output_resolution_from_literature_properties",
        "parameter_ids": list(REQUIRED_IDS),
        "source_file": str(source),
        "temperatures_K": list(temperatures),
        "values": table,
        "minimum_particle_radial_diffusion_scale_s": minimum_scale,
        "maximum_particle_radial_diffusion_scale_s": maximum_scale,
        "previous_25_s_output_interval_to_maximum_scale_ratio": 25.0 / maximum_scale,
        "velocity_time_scale_source": str(velocity_summary.resolve()),
        "corrected_local_crossing_time_s": {"minimum": crossing_min, "maximum": crossing_max},
        "projected_high_velocity_p95_turnover_rate_per_s": projected_high_p95_turnover,
        "selected_time_step_schedule": time_step_schedule,
        "selected_output_schedule": output_schedule,
        "initial_step_to_fastest_crossing_ratio": time_step_schedule[0]["delta_t_s"] / crossing_min,
        "steps_per_fastest_crossing": crossing_min / time_step_schedule[0]["delta_t_s"],
        "projected_high_velocity_p95_Courant_at_initial_step": (
            projected_high_p95_turnover * time_step_schedule[0]["delta_t_s"]
        ),
        "particle_scale_steps_in_1_to_25_s_stage": {
            "minimum": minimum_scale / time_step_schedule[2]["delta_t_s"],
            "maximum": maximum_scale / time_step_schedule[2]["delta_t_s"],
        },
        "interpretation": (
            "The r^2/alpha value is a particle conduction scale, not a fitted bed response time. "
            "A frozen velocity field does not remove advection from the temperature equation. "
            "The previous 0.25 s finest step would therefore smear the corrected sub-0.1 s local "
            "temperature transport. The selected staged schedule starts at 1e-5 s, then increases "
            "only after the local advective passage and particle-scale response. Three schedules "
            "refine every stage by the same factor of two before the formal calculation is chosen."
        ),
        "new_physical_parameters": [],
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "parameters/hccb_p418_physical_parameter_sources.csv",
    )
    parser.add_argument(
        "--velocity-summary",
        type=Path,
        default=root / "results/hccb_p418_velocity_step_time_scales/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results/hccb_p418_transient_time_resolution",
    )
    args = parser.parse_args()
    result = analyze(args.source.resolve(), args.velocity_summary.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "particle_thermal_scales.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["values"][0]))
        writer.writeheader()
        writer.writerows(result["values"])
    result["table"] = table_path.name
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
