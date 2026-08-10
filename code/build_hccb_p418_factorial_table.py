#!/usr/bin/env python3
"""Build the full-factorial steady-response table for the manuscript."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


OBSERVABLES = (
    ("pressure_drop_Pa", "Pressure drop"),
    ("outlet_temperature_K", "Outlet temperature"),
    ("solid_maximum_temperature_K", "Maximum solid temperature"),
    ("net_outward_enthalpy_flow_W", "Outlet enthalpy flow"),
    ("cooling_wall_heat_into_fluid_W", "Cooling-wall heat"),
)
MAIN_EFFECTS = (
    ("inlet_velocity", "velocity"),
    ("inlet_temperature", "temperature"),
    ("solid_heat_source", "heat source"),
)
INTERACTIONS = (
    "velocity_x_temperature",
    "velocity_x_heat_source",
    "temperature_x_heat_source",
    "velocity_x_temperature_x_heat_source",
)


def finite_fraction(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < -1.0e-10:
        raise ValueError(f"invalid factorial fraction for {name}: {value}")
    return max(0.0, result)


def fmt(value: float) -> str:
    if value < 0.005:
        return "$<0.01$"
    return f"{value:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physics-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    args = parser.parse_args()

    source = args.physics_summary.resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("status") != "completed_p418_case_physics_summarized":
        raise ValueError("steady physics summary is incomplete")
    if int(payload.get("completed_case_count", -1)) != 60:
        raise ValueError("factorial table requires all 60 steady conditions")
    if not bool(payload.get("complete_factorial_decomposition_available")):
        raise ValueError("complete factorial decomposition is unavailable")
    records = payload.get("factorial_variance_decomposition")
    if not isinstance(records, list):
        raise ValueError("factorial decomposition records are missing")

    lookup: dict[tuple[str, str], float] = {}
    for record in records:
        key = (str(record.get("observable")), str(record.get("effect")))
        if key in lookup:
            raise ValueError(f"duplicate factorial record: {key}")
        lookup[key] = finite_fraction(
            record.get("variance_fraction_percent"), f"{key[0]}/{key[1]}"
        )

    expected_effects = {name for name, _ in MAIN_EFFECTS} | set(INTERACTIONS)
    table_rows = []
    for observable, label in OBSERVABLES:
        observed_effects = {
            effect for obs, effect in lookup if obs == observable
        }
        if observed_effects != expected_effects:
            raise ValueError(f"factorial effects are incomplete for {observable}")
        main_values = {
            effect: lookup[(observable, effect)] for effect, _ in MAIN_EFFECTS
        }
        interaction = sum(lookup[(observable, effect)] for effect in INTERACTIONS)
        total = sum(main_values.values()) + interaction
        if not math.isclose(total, 100.0, rel_tol=1.0e-6, abs_tol=1.0e-5):
            raise ValueError(
                f"factorial fractions do not sum to 100% for {observable}: {total}"
            )
        dominant_effect = max(main_values, key=main_values.get)
        table_rows.append(
            {
                "observable": observable,
                "label": label,
                "inlet_velocity_percent": main_values["inlet_velocity"],
                "inlet_temperature_percent": main_values["inlet_temperature"],
                "solid_heat_source_percent": main_values["solid_heat_source"],
                "combined_interactions_percent": interaction,
                "dominant_main_effect": dominant_effect,
            }
        )

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Full-factorial variation of the pore-resolved steady responses. Entries are deterministic sum-of-squares fractions (\%) over the complete $5\times4\times3$ velocity--temperature--heat-source matrix. The interaction column combines all two- and three-factor terms; these fractions describe response variation over the sampled operating matrix and are not statistical uncertainty estimates.}",
        r"\label{tab:steady_factorial_effects}",
        r"\small",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrr@{}}",
        r"\toprule",
        r"Response & Velocity & Temperature & Heat source & Interactions \\",
        r"\midrule",
    ]
    for row in table_rows:
        lines.append(
            f"{row['label']} & {fmt(row['inlet_velocity_percent'])} & "
            f"{fmt(row['inlet_temperature_percent'])} & "
            f"{fmt(row['solid_heat_source_percent'])} & "
            f"{fmt(row['combined_interactions_percent'])} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""))

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")

    summary_output = args.summary_output.resolve()
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "completed_p418_factorial_manuscript_table",
        "physics_summary": str(source),
        "completed_case_count": 60,
        "rows": table_rows,
        "tex_output": str(output),
        "new_physical_parameters": [],
    }
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
