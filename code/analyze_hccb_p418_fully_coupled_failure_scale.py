#!/usr/bin/env python3
"""Quantify the physical temperature scale of the failed coupled startup."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from hccb_source_backed_thermophysical import load_hccb_thermophysical_parameters


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def cp_mass(temperature_k: float, coefficients: tuple[float, float, float], molar_mass: float) -> float:
    constant, linear, inverse_square = coefficients
    return (
        constant + linear * temperature_k + inverse_square / temperature_k**2
    ) / molar_mass


def minimum_registered_cp(parameters) -> tuple[float, float]:
    low, high = parameters.solid_cp_temperature_range_k
    _, linear, inverse_square = parameters.solid_cp_molar_coefficients
    stationary = (2.0 * inverse_square / linear) ** (1.0 / 3.0)
    candidates = [low, high]
    if low <= stationary <= high:
        candidates.append(stationary)
    values = [
        cp_mass(
            value,
            parameters.solid_cp_molar_coefficients,
            parameters.solid_molar_mass_kg_mol,
        )
        for value in candidates
    ]
    index = min(range(len(values)), key=values.__getitem__)
    return candidates[index], values[index]


def condition_row(path: Path, condition_id: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        matches = [row for row in csv.DictReader(handle) if row["condition_id"] == condition_id]
    if len(matches) != 1:
        raise ValueError(f"expected one {condition_id} row, found {len(matches)}")
    return matches[0]


def analyze(
    failure_path: Path,
    extrema_path: Path,
    steady_summary_path: Path,
) -> dict[str, object]:
    failure_record = load_json(failure_path)
    extrema = load_json(extrema_path)
    if failure_record.get("sequence_id") != "source_up_u0p15_T700":
        raise ValueError("unexpected fully coupled failure sequence")
    if extrema.get("condition_id") != "u0p15_T700_q4p85":
        raise ValueError("temperature extrema do not describe the source endpoint")

    source = condition_row(steady_summary_path, "u0p15_T700_q4p85")
    target = condition_row(steady_summary_path, "u0p15_T700_q8p85")
    if float(source["solid_heat_source_MW_m3"]) != 4.85:
        raise ValueError("source heat generation changed")
    target_source_w_m3 = float(target["solid_heat_source_MW_m3"]) * 1.0e6

    parameters = load_hccb_thermophysical_parameters()
    cp_min_temperature_k, cp_min_j_kg_k = minimum_registered_cp(parameters)
    elapsed_s = float(failure_record["last_logged_physical_time_s"])
    query_temperature_k = float(failure_record["failure"]["query_temperature_K"])
    initial_upper_k = max(
        float(extrema["fluid_temperature_max_K"]),
        float(extrema["solid_temperature_max_K"]),
        float(extrema["maximum_prescribed_boundary_temperature_K"]),
    )
    source_only_rise_k = (
        target_source_w_m3
        * elapsed_s
        / (parameters.solid_density_kg_m3 * cp_min_j_kg_k)
    )
    excursion_k = query_temperature_k - initial_upper_k
    if source_only_rise_k <= 0.0 or excursion_k <= 0.0:
        raise ValueError("failure record does not contain a positive temperature excursion")

    # A pressure excursion can add compressive work.  This ideal-gas estimate is
    # deliberately generous: it assumes the largest logged pressure is reached
    # adiabatically from 120 kPa and instantaneously transferred to the solid.
    pressure_high_pa = float(extrema["generous_pressure_high_Pa"])
    pressure_reference_pa = float(extrema["generous_pressure_reference_Pa"])
    helium_gamma = float(extrema["helium_ideal_gas_gamma"])
    if pressure_high_pa <= pressure_reference_pa or helium_gamma <= 1.0:
        raise ValueError("invalid generous compression-scale inputs")
    adiabatic_temperature_k = initial_upper_k * (
        pressure_high_pa / pressure_reference_pa
    ) ** ((helium_gamma - 1.0) / helium_gamma)
    generous_combined_upper_k = adiabatic_temperature_k + source_only_rise_k

    return {
        "status": "fully_coupled_failure_temperature_scale_quantified",
        "sequence_id": failure_record["sequence_id"],
        "failure_time_s": elapsed_s,
        "initial_temperature_upper_K": initial_upper_k,
        "failed_query_temperature_K": query_temperature_k,
        "observed_excursion_above_initial_upper_K": excursion_k,
        "target_volumetric_source_W_m3": target_source_w_m3,
        "solid_density_kg_m3": parameters.solid_density_kg_m3,
        "minimum_registered_cp_J_kg_K": cp_min_j_kg_k,
        "minimum_cp_temperature_K": cp_min_temperature_k,
        "source_only_temperature_rise_scale_K": source_only_rise_k,
        "observed_excursion_over_source_only_scale": excursion_k / source_only_rise_k,
        "generous_adiabatic_plus_source_upper_K": generous_combined_upper_k,
        "failed_query_above_generous_upper_K": query_temperature_k - generous_combined_upper_k,
        "generous_pressure_reference_Pa": pressure_reference_pa,
        "generous_pressure_high_Pa": pressure_high_pa,
        "helium_ideal_gas_gamma": helium_gamma,
        "pressure_scale_source": extrema["pressure_scale_source"],
        "interpretation": (
            "The 1308.8 K lookup is a numerical startup excursion, not a usable "
            "fully coupled thermal response. The source-only scale uses the largest "
            "target heat generation and the minimum registered heat capacity; the "
            "additional ideal-gas compression estimate is intentionally generous and "
            "does not turn the failed run into physical evidence."
        ),
        "parameter_ids": ["P403", "P429", "P430"],
        "new_physical_parameters": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--extrema", type=Path, required=True)
    parser.add_argument("--steady-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.failure, args.extrema, args.steady_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
