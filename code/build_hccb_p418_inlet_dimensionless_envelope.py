#!/usr/bin/env python3
"""Calculate the source-backed inlet dimensionless envelope for P418 cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


REQUIRED_PARAMETERS = ("P048", "P068", "P070", "P071", "P073", "P388", "P418", "P426")


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = {row["parameter_id"]: row for row in csv.DictReader(stream)}
    missing = [name for name in REQUIRED_PARAMETERS if name not in rows]
    if missing:
        raise ValueError(f"missing source-backed parameters: {missing}")
    return rows


def helium_properties(
    temperature_k: float,
    pressure_pa: float,
    *,
    molar_mass_kg_mol: float,
    gas_constant_j_mol_k: float,
    cp_j_kg_k: float,
) -> dict[str, float]:
    if temperature_k <= 0.0 or pressure_pa <= 0.0:
        raise ValueError("temperature and pressure must be positive")
    density = pressure_pa * molar_mass_kg_mol / (
        gas_constant_j_mol_k * temperature_k
    )
    viscosity = 0.4646 * temperature_k**0.66 * 1.0e-6
    theta = temperature_k / 273.0
    conductivity = 0.1448 * theta**0.68 * (
        1.0
        + 2.5e-3
        * (pressure_pa / 1.0e6) ** 1.17
        * theta ** (-1.85)
    )
    prandtl = cp_j_kg_k * viscosity / conductivity
    return {
        "density_kg_m3": density,
        "dynamic_viscosity_Pa_s": viscosity,
        "thermal_conductivity_W_m_K": conductivity,
        "prandtl": prandtl,
    }


def validate_registered_correlations(parameters: dict[str, dict[str, str]]) -> None:
    expected_tokens = {
        "P070": ("0.4646", "0.66", "1e-6"),
        "P071": ("0.1448", "0.68", "2.5e-3", "1.17", "-1.85"),
    }
    for parameter_id, tokens in expected_tokens.items():
        relation = parameters[parameter_id]["value"].replace(" ", "")
        missing = [token for token in tokens if token not in relation]
        if missing:
            raise ValueError(
                f"registered {parameter_id} relation changed; missing {missing}"
            )


def build_rows(
    cases: list[dict[str, object]], parameters: dict[str, dict[str, str]]
) -> list[dict[str, object]]:
    if len(cases) != 60:
        raise ValueError(f"expected 60 P418 cases, found {len(cases)}")
    if not all(bool(case.get("dictionary_values_match_sources")) for case in cases):
        raise ValueError("one or more OpenFOAM dictionaries differ from source values")
    validate_registered_correlations(parameters)
    diameter_m = float(parameters["P048"]["value"]) * 1.0e-3
    molar_mass = float(parameters["P068"]["value"]) * 1.0e-3
    gas_constant = float(parameters["P073"]["value"])
    cp = float(parameters["P388"]["value"])
    pressure_pa = float(parameters["P426"]["value"]) * 1.0e6
    rows = []
    for case in cases:
        temperature = float(case["inlet_temperature_K"])
        velocity = float(case["inlet_velocity_m_s"])
        properties = helium_properties(
            temperature,
            pressure_pa,
            molar_mass_kg_mol=molar_mass,
            gas_constant_j_mol_k=gas_constant,
            cp_j_kg_k=cp,
        )
        reynolds = (
            properties["density_kg_m3"]
            * velocity
            * diameter_m
            / properties["dynamic_viscosity_Pa_s"]
        )
        rows.append(
            {
                "condition_id": str(case["condition_id"]),
                "inlet_velocity_m_s": velocity,
                "inlet_temperature_K": temperature,
                "solid_heat_source_MW_m3": float(case["solid_heat_source_MW_m3"]),
                **properties,
                "particle_reynolds_inlet": reynolds,
                "particle_peclet_inlet": reynolds * properties["prandtl"],
                "inlet_reynolds_below_1p8": reynolds < 1.8,
            }
        )
    return rows


def bounds(rows: list[dict[str, object]], name: str) -> dict[str, float]:
    values = [float(row[name]) for row in rows]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite {name}")
    return {"minimum": min(values), "maximum": max(values)}


def build_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    unique_thermal_states = {
        (float(row["inlet_velocity_m_s"]), float(row["inlet_temperature_K"]))
        for row in rows
    }
    above = [row for row in rows if float(row["particle_reynolds_inlet"]) >= 1.8]
    return {
        "status": "completed_p418_source_backed_inlet_dimensionless_envelope",
        "case_count": len(rows),
        "unique_velocity_temperature_state_count": len(unique_thermal_states),
        "particle_reynolds_inlet": bounds(rows, "particle_reynolds_inlet"),
        "prandtl_inlet": bounds(rows, "prandtl"),
        "particle_peclet_inlet": bounds(rows, "particle_peclet_inlet"),
        "case_count_with_inlet_reynolds_at_or_above_1p8": len(above),
        "unique_velocity_temperature_states_at_or_above_1p8": sorted(
            {
                f"u={float(row['inlet_velocity_m_s']):.2f} m/s, T={float(row['inlet_temperature_K']):.0f} K"
                for row in above
            }
        ),
        "interpretation": (
            "The inlet particle Reynolds number is not the source paper's Re_p,AVE. "
            "The latter is a packed-bed average whose spatial averaging equation is not published. "
            "Resolved-field comparisons therefore calculate a separate through-flow average and "
            "label values outside the source Re_p,AVE < 1.8 support."
        ),
        "parameter_ids": list(REQUIRED_PARAMETERS),
        "new_physical_parameters": [],
    }


def write_tex(summary: dict[str, object], output: Path) -> None:
    reynolds = summary["particle_reynolds_inlet"]
    prandtl = summary["prandtl_inlet"]
    peclet = summary["particle_peclet_inlet"]
    lines = [
        "% Generated by code/build_hccb_p418_inlet_dimensionless_envelope.py",
        rf"\newcommand{{\InletReMin}}{{{float(reynolds['minimum']):.3f}}}",
        rf"\newcommand{{\InletReMax}}{{{float(reynolds['maximum']):.2f}}}",
        rf"\newcommand{{\InletPrMin}}{{{float(prandtl['minimum']):.3f}}}",
        rf"\newcommand{{\InletPrMax}}{{{float(prandtl['maximum']):.3f}}}",
        rf"\newcommand{{\InletPeMin}}{{{float(peclet['minimum']):.3f}}}",
        rf"\newcommand{{\InletPeMax}}{{{float(peclet['maximum']):.2f}}}",
        rf"\newcommand{{\HighInletReCaseCount}}{{{int(summary['case_count_with_inlet_reynolds_at_or_above_1p8'])}}}",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path, required=True)
    args = parser.parse_args()
    parameters = read_manifest(args.manifest.resolve())
    input_payload = json.loads(args.input_summary.resolve().read_text(encoding="utf-8"))
    cases = input_payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("actual-case input summary has no cases")
    rows = build_rows(cases, parameters)
    summary = build_summary(rows)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "inlet_dimensionless_conditions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_tex(summary, args.tex_output.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
