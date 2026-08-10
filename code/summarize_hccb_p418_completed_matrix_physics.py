#!/usr/bin/env python3
"""Summarize physical trends from completed P418 OpenFOAM CHT cases."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hccb_source_backed_thermophysical import load_hccb_thermophysical_parameters


def completed_rows(
    matrix_root: Path,
    time_name: str | None,
    *,
    time_from_completion_marker: bool = False,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for marker in sorted(matrix_root.glob("*/formal_sample_complete.json")):
        case = marker.parent
        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        case_time = (
            str(marker_payload["time"])
            if time_from_completion_marker
            else str(time_name)
        )
        summary_path = case / f"cht_result_summary_{case_time}.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary.get("solver_finished") or not summary.get(
            "all_reported_values_are_finite"
        ):
            continue
        condition = summary["physical_conditions"]
        flow = summary["flow"]
        temperature = summary["temperature"]
        heat = summary["heat_balance"]
        generated = float(heat["solid_generated_power_W"])
        wall_heat = float(heat["cooling_wall_heat_flow_W"])
        inlet_temperature = float(condition["inlet_temperature_K"])
        outlet_temperature = float(temperature["outlet_average_K"])
        outward_enthalpy = float(heat["net_outward_enthalpy_flow_W"])
        external_fluid_raw = heat.get("external_fluid_conductive_heat_flow_W")
        external_solid_raw = heat.get("external_solid_conductive_heat_flow_W")
        boundary_heat_fallback_used = (
            external_fluid_raw is None or external_solid_raw is None
        )
        external_fluid_heat = (
            wall_heat if external_fluid_raw is None else float(external_fluid_raw)
        )
        external_solid_heat = (
            0.0 if external_solid_raw is None else float(external_solid_raw)
        )
        total_external_boundary_heat = external_fluid_heat + external_solid_heat
        solid_maximum = float(temperature["solid_maximum_K"])
        wall_temperature = float(condition["cooling_wall_temperature_K"])
        rows.append(
            {
                "condition_id": case.name,
                "completion_time_s": case_time,
                "inlet_velocity_m_s": float(condition["inlet_velocity_m_s"]),
                "inlet_temperature_K": inlet_temperature,
                "cooling_wall_temperature_K": wall_temperature,
                "solid_heat_source_MW_m3": float(
                    condition["solid_heat_source_W_m3"]
                )
                / 1.0e6,
                "pressure_drop_Pa": float(flow["pressure_drop_Pa"]),
                "outlet_temperature_K": outlet_temperature,
                "outlet_minus_inlet_temperature_K": (
                    outlet_temperature - inlet_temperature
                ),
                "solid_maximum_temperature_K": solid_maximum,
                "solid_maximum_minus_cooling_wall_K": (
                    solid_maximum - wall_temperature
                ),
                "generated_power_W": generated,
                "net_outward_enthalpy_flow_W": outward_enthalpy,
                "net_outward_enthalpy_over_generated": outward_enthalpy / generated,
                "external_fluid_conductive_heat_W": external_fluid_heat,
                "external_solid_conductive_heat_W": external_solid_heat,
                "total_external_boundary_heat_W": total_external_boundary_heat,
                "total_external_boundary_heat_over_generated": (
                    total_external_boundary_heat / generated
                ),
                "reconstructed_energy_difference_W": (
                    generated + total_external_boundary_heat - outward_enthalpy
                ),
                "boundary_heat_fallback_used": boundary_heat_fallback_used,
                "cooling_wall_heat_into_fluid_W": wall_heat,
                "cooling_wall_heat_over_generated": wall_heat / generated,
                "cooling_wall_heat_direction": (
                    "wall_to_fluid"
                    if wall_heat > 0.0
                    else "fluid_to_wall"
                    if wall_heat < 0.0
                    else "zero"
                ),
                "relative_mass_difference": float(flow["relative_mass_difference"]),
                "relative_energy_difference": float(
                    heat["relative_energy_difference"]
                ),
            }
        )
    if not rows:
        raise ValueError(f"no completed finite P418 summaries in {matrix_root}")
    return rows


def temperature_pairs(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    groups: dict[tuple[float, float], list[dict[str, float | str]]] = {}
    for row in rows:
        key = (
            float(row["inlet_velocity_m_s"]),
            float(row["solid_heat_source_MW_m3"]),
        )
        groups.setdefault(key, []).append(row)
    pairs: list[dict[str, float | str]] = []
    for (velocity, source), members in sorted(groups.items()):
        ordered = sorted(members, key=lambda item: float(item["inlet_temperature_K"]))
        if len(ordered) < 2:
            continue
        low, high = ordered[0], ordered[-1]
        temperature_step = float(high["inlet_temperature_K"]) - float(
            low["inlet_temperature_K"]
        )
        if temperature_step <= 0.0:
            continue
        low_pressure = float(low["pressure_drop_Pa"])
        pairs.append(
            {
                "inlet_velocity_m_s": velocity,
                "solid_heat_source_MW_m3": source,
                "low_condition_id": str(low["condition_id"]),
                "high_condition_id": str(high["condition_id"]),
                "inlet_temperature_step_K": temperature_step,
                "outlet_temperature_response_K_per_K": (
                    float(high["outlet_temperature_K"])
                    - float(low["outlet_temperature_K"])
                )
                / temperature_step,
                "solid_maximum_response_K_per_K": (
                    float(high["solid_maximum_temperature_K"])
                    - float(low["solid_maximum_temperature_K"])
                )
                / temperature_step,
                "pressure_drop_change_percent": 100.0
                * (float(high["pressure_drop_Pa"]) - low_pressure)
                / low_pressure,
                "wall_heat_changes_sign": bool(
                    float(low["cooling_wall_heat_into_fluid_W"])
                    * float(high["cooling_wall_heat_into_fluid_W"])
                    < 0.0
                ),
            }
        )
    return pairs


def heat_source_pairs(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    """Compare the lowest and highest available heat source at fixed u and Tin."""
    groups: dict[tuple[float, float], list[dict[str, float | str]]] = {}
    for row in rows:
        key = (
            float(row["inlet_velocity_m_s"]),
            float(row["inlet_temperature_K"]),
        )
        groups.setdefault(key, []).append(row)
    pairs: list[dict[str, float | str]] = []
    for (velocity, temperature), members in sorted(groups.items()):
        ordered = sorted(members, key=lambda item: float(item["solid_heat_source_MW_m3"]))
        if len(ordered) < 2:
            continue
        low, high = ordered[0], ordered[-1]
        source_step = float(high["solid_heat_source_MW_m3"]) - float(
            low["solid_heat_source_MW_m3"]
        )
        if source_step <= 0.0:
            continue
        low_pressure = float(low["pressure_drop_Pa"])
        pairs.append(
            {
                "inlet_velocity_m_s": velocity,
                "inlet_temperature_K": temperature,
                "low_condition_id": str(low["condition_id"]),
                "high_condition_id": str(high["condition_id"]),
                "heat_source_step_MW_m3": source_step,
                "outlet_temperature_response_K_per_MW_m3": (
                    float(high["outlet_temperature_K"])
                    - float(low["outlet_temperature_K"])
                )
                / source_step,
                "solid_maximum_response_K_per_MW_m3": (
                    float(high["solid_maximum_temperature_K"])
                    - float(low["solid_maximum_temperature_K"])
                )
                / source_step,
                "low_solid_maximum_minus_cooling_wall_K": (
                    float(low["solid_maximum_temperature_K"])
                    - float(low["cooling_wall_temperature_K"])
                ),
                "high_solid_maximum_minus_cooling_wall_K": (
                    float(high["solid_maximum_temperature_K"])
                    - float(high["cooling_wall_temperature_K"])
                ),
                "pressure_drop_change_percent": 100.0
                * (float(high["pressure_drop_Pa"]) - low_pressure)
                / low_pressure,
                "wall_heat_response_W_per_MW_m3": (
                    float(high["cooling_wall_heat_into_fluid_W"])
                    - float(low["cooling_wall_heat_into_fluid_W"])
                )
                / source_step,
            }
        )
    return pairs


def velocity_pairs(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    """Compare the lowest and highest available inlet velocity at fixed Tin and q."""
    groups: dict[tuple[float, float], list[dict[str, float | str]]] = {}
    for row in rows:
        key = (
            float(row["inlet_temperature_K"]),
            float(row["solid_heat_source_MW_m3"]),
        )
        groups.setdefault(key, []).append(row)
    pairs: list[dict[str, float | str]] = []
    for (temperature, source), members in sorted(groups.items()):
        ordered = sorted(members, key=lambda item: float(item["inlet_velocity_m_s"]))
        if len(ordered) < 2:
            continue
        low, high = ordered[0], ordered[-1]
        velocity_step = float(high["inlet_velocity_m_s"]) - float(
            low["inlet_velocity_m_s"]
        )
        if velocity_step <= 0.0:
            continue
        low_pressure = float(low["pressure_drop_Pa"])
        pairs.append(
            {
                "inlet_temperature_K": temperature,
                "solid_heat_source_MW_m3": source,
                "low_condition_id": str(low["condition_id"]),
                "high_condition_id": str(high["condition_id"]),
                "velocity_step_m_s": velocity_step,
                "pressure_drop_response_Pa_per_m_s": (
                    float(high["pressure_drop_Pa"]) - low_pressure
                )
                / velocity_step,
                "pressure_drop_change_percent": 100.0
                * (float(high["pressure_drop_Pa"]) - low_pressure)
                / low_pressure,
                "outlet_temperature_response_K_per_m_s": (
                    float(high["outlet_temperature_K"])
                    - float(low["outlet_temperature_K"])
                )
                / velocity_step,
                "solid_maximum_response_K_per_m_s": (
                    float(high["solid_maximum_temperature_K"])
                    - float(low["solid_maximum_temperature_K"])
                )
                / velocity_step,
                "wall_heat_response_W_per_m_s": (
                    float(high["cooling_wall_heat_into_fluid_W"])
                    - float(low["cooling_wall_heat_into_fluid_W"])
                )
                / velocity_step,
            }
        )
    return pairs


def single_factor_linearity(
    rows: list[dict[str, float | str]],
) -> list[dict[str, object]]:
    """Measure interior-point departure from the line joining each slice endpoint."""
    factors = (
        (
            "inlet_velocity_m_s",
            "m s^-1",
            ("inlet_temperature_K", "solid_heat_source_MW_m3"),
        ),
        (
            "inlet_temperature_K",
            "K",
            ("inlet_velocity_m_s", "solid_heat_source_MW_m3"),
        ),
        (
            "solid_heat_source_MW_m3",
            "MW m^-3",
            ("inlet_velocity_m_s", "inlet_temperature_K"),
        ),
    )
    observables = (
        ("pressure_drop_Pa", "Pa"),
        ("outlet_temperature_K", "K"),
        ("solid_maximum_temperature_K", "K"),
        ("net_outward_enthalpy_flow_W", "W"),
        ("cooling_wall_heat_into_fluid_W", "W"),
    )
    output: list[dict[str, object]] = []
    for varied_factor, factor_unit, fixed_factors in factors:
        groups: dict[tuple[float, float], list[dict[str, float | str]]] = {}
        for row in rows:
            key = tuple(float(row[field]) for field in fixed_factors)
            groups.setdefault(key, []).append(row)
        for fixed_values, members in sorted(groups.items()):
            ordered = sorted(members, key=lambda item: float(item[varied_factor]))
            x_values = [float(item[varied_factor]) for item in ordered]
            if len(set(x_values)) < 3:
                continue
            low = ordered[0]
            high = ordered[-1]
            x_low = float(low[varied_factor])
            x_high = float(high[varied_factor])
            if x_high <= x_low:
                continue
            for interior in ordered[1:-1]:
                x = float(interior[varied_factor])
                weight = (x - x_low) / (x_high - x_low)
                for observable, observable_unit in observables:
                    y_low = float(low[observable])
                    y_high = float(high[observable])
                    y = float(interior[observable])
                    y_linear = y_low + weight * (y_high - y_low)
                    deviation = y - y_linear
                    endpoint_span = abs(y_high - y_low)
                    observed_range = max(
                        float(item[observable]) for item in ordered
                    ) - min(float(item[observable]) for item in ordered)
                    output.append(
                        {
                            "varied_factor": varied_factor,
                            "varied_factor_unit": factor_unit,
                            "fixed_factor_1": fixed_factors[0],
                            "fixed_factor_1_value": fixed_values[0],
                            "fixed_factor_2": fixed_factors[1],
                            "fixed_factor_2_value": fixed_values[1],
                            "low_condition_id": str(low["condition_id"]),
                            "interior_condition_id": str(interior["condition_id"]),
                            "high_condition_id": str(high["condition_id"]),
                            "low_factor_value": x_low,
                            "interior_factor_value": x,
                            "high_factor_value": x_high,
                            "observable": observable,
                            "observable_unit": observable_unit,
                            "actual_interior_value": y,
                            "endpoint_linear_prediction": y_linear,
                            "signed_deviation": deviation,
                            "absolute_deviation": abs(deviation),
                            "deviation_over_endpoint_span_percent": (
                                100.0 * abs(deviation) / endpoint_span
                                if endpoint_span > np.finfo(np.float64).tiny
                                else None
                            ),
                            "deviation_over_observed_range_percent": (
                                100.0 * abs(deviation) / observed_range
                                if observed_range > np.finfo(np.float64).tiny
                                else None
                            ),
                        }
                    )
    return output


def summarize_single_factor_linearity(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Report the largest interior-point departure for each factor and observable."""
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in records:
        key = (str(record["varied_factor"]), str(record["observable"]))
        grouped.setdefault(key, []).append(record)
    summary: list[dict[str, object]] = []
    for (factor, observable), members in sorted(grouped.items()):
        worst = max(members, key=lambda item: float(item["absolute_deviation"]))
        relative_values = [
            float(item["deviation_over_observed_range_percent"])
            for item in members
            if item["deviation_over_observed_range_percent"] is not None
        ]
        summary.append(
            {
                "varied_factor": factor,
                "observable": observable,
                "observable_unit": str(worst["observable_unit"]),
                "interior_point_count": len(members),
                "maximum_absolute_deviation": float(worst["absolute_deviation"]),
                "maximum_deviation_over_observed_range_percent": (
                    max(relative_values) if relative_values else None
                ),
                "worst_interior_condition_id": str(
                    worst["interior_condition_id"]
                ),
                "fixed_factor_1": str(worst["fixed_factor_1"]),
                "fixed_factor_1_value": float(worst["fixed_factor_1_value"]),
                "fixed_factor_2": str(worst["fixed_factor_2"]),
                "fixed_factor_2_value": float(worst["fixed_factor_2_value"]),
            }
        )
    return summary


def wall_heat_zero_crossings(
    rows: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    """Linearly locate a wall-heat sign change at fixed inlet velocity and source."""
    groups: dict[tuple[float, float], list[dict[str, float | str]]] = {}
    for row in rows:
        key = (
            float(row["inlet_velocity_m_s"]),
            float(row["solid_heat_source_MW_m3"]),
        )
        groups.setdefault(key, []).append(row)
    crossings: list[dict[str, float | str]] = []
    for (velocity, source), members in sorted(groups.items()):
        ordered = sorted(members, key=lambda item: float(item["inlet_temperature_K"]))
        for low, high in zip(ordered[:-1], ordered[1:]):
            low_temperature = float(low["inlet_temperature_K"])
            high_temperature = float(high["inlet_temperature_K"])
            low_heat = float(low["cooling_wall_heat_into_fluid_W"])
            high_heat = float(high["cooling_wall_heat_into_fluid_W"])
            if high_temperature <= low_temperature or low_heat * high_heat > 0.0:
                continue
            if high_heat == low_heat:
                continue
            crossing_temperature = low_temperature - low_heat * (
                high_temperature - low_temperature
            ) / (high_heat - low_heat)
            crossings.append(
                {
                    "inlet_velocity_m_s": velocity,
                    "solid_heat_source_MW_m3": source,
                    "lower_condition_id": str(low["condition_id"]),
                    "upper_condition_id": str(high["condition_id"]),
                    "lower_inlet_temperature_K": low_temperature,
                    "upper_inlet_temperature_K": high_temperature,
                    "temperature_bracket_width_K": (
                        high_temperature - low_temperature
                    ),
                    "lower_wall_heat_into_fluid_W": low_heat,
                    "upper_wall_heat_into_fluid_W": high_heat,
                    "interpolated_zero_wall_heat_inlet_temperature_K": (
                        crossing_temperature
                    ),
                    "cooling_wall_temperature_K": float(
                        low["cooling_wall_temperature_K"]
                    ),
                }
            )
    return crossings


def wall_heat_zero_crossing_responses(
    crossings: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    """Compare how the interpolated zero-wall-heat temperature moves with u and q."""
    factor_pairs = (
        (
            "solid_heat_source_MW_m3",
            "MW m^-3",
            "inlet_velocity_m_s",
            "K per MW m^-3",
        ),
        (
            "inlet_velocity_m_s",
            "m s^-1",
            "solid_heat_source_MW_m3",
            "K per m s^-1",
        ),
    )
    responses: list[dict[str, float | str]] = []
    for varied_factor, varied_unit, fixed_factor, response_unit in factor_pairs:
        groups: dict[float, list[dict[str, float | str]]] = {}
        for row in crossings:
            groups.setdefault(float(row[fixed_factor]), []).append(row)
        for fixed_value, members in sorted(groups.items()):
            ordered = sorted(members, key=lambda item: float(item[varied_factor]))
            if len(ordered) < 2:
                continue
            low, high = ordered[0], ordered[-1]
            input_step = float(high[varied_factor]) - float(low[varied_factor])
            if input_step <= 0.0:
                continue
            low_temperature = float(
                low["interpolated_zero_wall_heat_inlet_temperature_K"]
            )
            high_temperature = float(
                high["interpolated_zero_wall_heat_inlet_temperature_K"]
            )
            responses.append(
                {
                    "varied_factor": varied_factor,
                    "varied_factor_unit": varied_unit,
                    "fixed_factor": fixed_factor,
                    "fixed_factor_value": fixed_value,
                    "low_factor_value": float(low[varied_factor]),
                    "high_factor_value": float(high[varied_factor]),
                    "factor_step": input_step,
                    "low_zero_wall_heat_temperature_K": low_temperature,
                    "high_zero_wall_heat_temperature_K": high_temperature,
                    "zero_wall_heat_temperature_change_K": (
                        high_temperature - low_temperature
                    ),
                    "zero_wall_heat_temperature_response": (
                        high_temperature - low_temperature
                    )
                    / input_step,
                    "response_unit": response_unit,
                    "source_crossing_count": len(ordered),
                    "maximum_temperature_bracket_width_K": max(
                        float(item["temperature_bracket_width_K"])
                        for item in ordered
                    ),
                }
            )
    return responses


def thermal_regime_summary(
    rows: list[dict[str, float | str]],
) -> dict[str, object]:
    """Count wall-heat directions and report continuous hotspot indicators."""
    direction_counts = Counter(str(row["cooling_wall_heat_direction"]) for row in rows)
    hotspot_offsets = np.asarray(
        [float(row["solid_maximum_minus_cooling_wall_K"]) for row in rows],
        dtype=np.float64,
    )
    solid_maxima = np.asarray(
        [float(row["solid_maximum_temperature_K"]) for row in rows],
        dtype=np.float64,
    )
    outlet_rises = np.asarray(
        [float(row["outlet_minus_inlet_temperature_K"]) for row in rows],
        dtype=np.float64,
    )
    energy_over_generated = np.asarray(
        [
            abs(float(row["reconstructed_energy_difference_W"]))
            / max(abs(float(row["generated_power_W"])), np.finfo(np.float64).tiny)
            for row in rows
        ],
        dtype=np.float64,
    )
    return {
        "cooling_wall_heat_direction_counts": dict(sorted(direction_counts.items())),
        "solid_maximum_above_cooling_wall_count": int(np.count_nonzero(hotspot_offsets > 0.0)),
        "solid_maximum_at_or_below_cooling_wall_count": int(np.count_nonzero(hotspot_offsets <= 0.0)),
        "solid_maximum_minus_cooling_wall_range_K": [
            float(np.min(hotspot_offsets)),
            float(np.max(hotspot_offsets)),
        ],
        "solid_maximum_temperature_range_K": [
            float(np.min(solid_maxima)),
            float(np.max(solid_maxima)),
        ],
        "outlet_minus_inlet_temperature_range_K": [
            float(np.min(outlet_rises)),
            float(np.max(outlet_rises)),
        ],
        "maximum_reconstructed_energy_difference_over_generated": float(
            np.max(energy_over_generated)
        ),
        "definition": (
            "Signs are taken directly from the OpenFOAM cooling-wall heat flow and "
            "the computed solid maximum minus the prescribed cooling-wall temperature; "
            "no fitted transition threshold is introduced."
        ),
    }


def steady_transition_proximity(
    rows: list[dict[str, float | str]],
    transition_temperatures_k: tuple[float, ...],
) -> list[dict[str, float | str | bool]]:
    """Compare each computed steady solid maximum with published transitions."""
    records: list[dict[str, float | str | bool]] = []
    for row in rows:
        maximum = float(row["solid_maximum_temperature_K"])
        record: dict[str, float | str | bool] = {
            "condition_id": str(row["condition_id"]),
            "inlet_velocity_m_s": float(row["inlet_velocity_m_s"]),
            "inlet_temperature_K": float(row["inlet_temperature_K"]),
            "solid_heat_source_MW_m3": float(row["solid_heat_source_MW_m3"]),
            "solid_maximum_temperature_K": maximum,
        }
        for index, transition in enumerate(transition_temperatures_k, start=1):
            transition = float(transition)
            record[f"transition_{index}_temperature_K"] = transition
            record[f"transition_{index}_minus_solid_maximum_K"] = (
                transition - maximum
            )
            record[f"transition_{index}_reached_by_solid_maximum"] = bool(
                maximum >= transition
            )
        records.append(record)
    return records


def physical_trend_checks(
    rows: list[dict[str, float | str]],
    temperature_response: list[dict[str, float | str]],
    source_response: list[dict[str, float | str]],
    velocity_response: list[dict[str, float | str]],
) -> dict[str, object]:
    """Check sign relations that follow directly from the declared problem."""

    def pair_check(
        records: list[dict[str, float | str]],
        field: str,
        relation: str,
    ) -> dict[str, object]:
        if relation == "positive":
            failures = [record for record in records if float(record[field]) <= 0.0]
        elif relation == "nonnegative":
            failures = [record for record in records if float(record[field]) < 0.0]
        elif relation == "negative":
            failures = [record for record in records if float(record[field]) >= 0.0]
        else:
            raise ValueError(f"unknown relation {relation}")
        return {
            "evaluated": bool(records),
            "pair_count": len(records),
            "expected_relation": relation,
            "passed": not failures if records else None,
            "failing_condition_pairs": [
                [str(record["low_condition_id"]), str(record["high_condition_id"])]
                for record in failures
            ],
        }

    checks = {
        "positive_pressure_drop": {
            "evaluated": bool(rows),
            "case_count": len(rows),
            "passed": all(float(row["pressure_drop_Pa"]) > 0.0 for row in rows),
            "failing_condition_ids": [
                str(row["condition_id"])
                for row in rows
                if float(row["pressure_drop_Pa"]) <= 0.0
            ],
        },
        "pressure_drop_decreases_with_inlet_temperature_at_fixed_u_q": pair_check(
            temperature_response, "pressure_drop_change_percent", "negative"
        ),
        "outlet_temperature_increases_with_inlet_temperature_at_fixed_u_q": pair_check(
            temperature_response, "outlet_temperature_response_K_per_K", "positive"
        ),
        "solid_maximum_does_not_decrease_with_inlet_temperature_at_fixed_u_q": pair_check(
            temperature_response, "solid_maximum_response_K_per_K", "nonnegative"
        ),
        "outlet_temperature_increases_with_heat_source_at_fixed_u_T": pair_check(
            source_response, "outlet_temperature_response_K_per_MW_m3", "positive"
        ),
        "solid_maximum_does_not_decrease_with_heat_source_at_fixed_u_T": pair_check(
            source_response, "solid_maximum_response_K_per_MW_m3", "nonnegative"
        ),
        "pressure_drop_increases_with_velocity_at_fixed_T_q": pair_check(
            velocity_response, "pressure_drop_response_Pa_per_m_s", "positive"
        ),
    }
    evaluated = [item for item in checks.values() if item["evaluated"]]
    return {
        "all_evaluated_checks_passed": all(item["passed"] for item in evaluated),
        "checks": checks,
        "physical_basis": (
            "Only direct sign relations are checked. Wall heat and outlet-temperature "
            "responses to velocity are not assigned a fixed sign because the 635 K wall, "
            "inlet enthalpy and internal heating can compete."
        ),
    }


def factorial_variance_decomposition(
    rows: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    """Decompose a complete balanced u-T-q matrix into main and interaction effects."""
    velocities = sorted({float(row["inlet_velocity_m_s"]) for row in rows})
    temperatures = sorted({float(row["inlet_temperature_K"]) for row in rows})
    sources = sorted({float(row["solid_heat_source_MW_m3"]) for row in rows})
    if min(len(velocities), len(temperatures), len(sources)) < 2:
        return []
    expected = len(velocities) * len(temperatures) * len(sources)
    keys = {
        (
            float(row["inlet_velocity_m_s"]),
            float(row["inlet_temperature_K"]),
            float(row["solid_heat_source_MW_m3"]),
        )
        for row in rows
    }
    if len(rows) != expected or len(keys) != expected:
        return []
    lookup = {
        (
            float(row["inlet_velocity_m_s"]),
            float(row["inlet_temperature_K"]),
            float(row["solid_heat_source_MW_m3"]),
        ): row
        for row in rows
    }
    observables = (
        ("pressure_drop_Pa", "Pa"),
        ("outlet_temperature_K", "K"),
        ("solid_maximum_temperature_K", "K"),
        ("net_outward_enthalpy_flow_W", "W"),
        ("cooling_wall_heat_into_fluid_W", "W"),
    )
    output: list[dict[str, float | str]] = []
    for observable, unit in observables:
        field = np.asarray(
            [
                [
                    [float(lookup[(u, temperature, source)][observable]) for source in sources]
                    for temperature in temperatures
                ]
                for u in velocities
            ],
            dtype=np.float64,
        )
        grand = float(np.mean(field))
        mean_u = np.mean(field, axis=(1, 2))
        mean_t = np.mean(field, axis=(0, 2))
        mean_q = np.mean(field, axis=(0, 1))
        mean_ut = np.mean(field, axis=2)
        mean_uq = np.mean(field, axis=1)
        mean_tq = np.mean(field, axis=0)
        ss_total = float(np.sum((field - grand) ** 2))
        if ss_total <= np.finfo(np.float64).tiny:
            continue
        ss_u = len(temperatures) * len(sources) * float(np.sum((mean_u - grand) ** 2))
        ss_t = len(velocities) * len(sources) * float(np.sum((mean_t - grand) ** 2))
        ss_q = len(velocities) * len(temperatures) * float(np.sum((mean_q - grand) ** 2))
        interaction_ut = mean_ut - mean_u[:, None] - mean_t[None, :] + grand
        interaction_uq = mean_uq - mean_u[:, None] - mean_q[None, :] + grand
        interaction_tq = mean_tq - mean_t[:, None] - mean_q[None, :] + grand
        ss_ut = len(sources) * float(np.sum(interaction_ut**2))
        ss_uq = len(temperatures) * float(np.sum(interaction_uq**2))
        ss_tq = len(velocities) * float(np.sum(interaction_tq**2))
        interaction_utq = (
            field
            - mean_ut[:, :, None]
            - mean_uq[:, None, :]
            - mean_tq[None, :, :]
            + mean_u[:, None, None]
            + mean_t[None, :, None]
            + mean_q[None, None, :]
            - grand
        )
        ss_utq = float(np.sum(interaction_utq**2))
        effects = (
            ("inlet_velocity", ss_u),
            ("inlet_temperature", ss_t),
            ("solid_heat_source", ss_q),
            ("velocity_x_temperature", ss_ut),
            ("velocity_x_heat_source", ss_uq),
            ("temperature_x_heat_source", ss_tq),
            ("velocity_x_temperature_x_heat_source", ss_utq),
        )
        for effect, sum_of_squares in effects:
            output.append(
                {
                    "observable": observable,
                    "observable_unit": unit,
                    "effect": effect,
                    "sum_of_squares": sum_of_squares,
                    "variance_fraction_percent": 100.0
                    * sum_of_squares
                    / ss_total,
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_optional_csv(
    path: Path, rows: list[dict[str, float | str]]
) -> None:
    """Write a conditional result or remove a stale file from an earlier run."""
    if rows:
        write_csv(path, rows)
    else:
        path.unlink(missing_ok=True)


def plot_rows(rows: list[dict[str, float | str]], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 5.3), constrained_layout=True)
    temperature = np.asarray([float(row["inlet_temperature_K"]) for row in rows])
    velocity = np.asarray([float(row["inlet_velocity_m_s"]) for row in rows])
    pressure = np.asarray([float(row["pressure_drop_Pa"]) for row in rows])
    scatter = axes[0, 0].scatter(
        velocity,
        pressure,
        c=temperature,
        cmap="cividis",
        s=36,
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )
    colorbar = fig.colorbar(scatter, ax=axes[0, 0], pad=0.02)
    colorbar.set_label(r"$T_{\mathrm{in}}$ (K)")
    axes[0, 0].set_xlabel(r"$u_{\mathrm{in}}$ (m s$^{-1}$)")
    axes[0, 0].set_ylabel(r"$\Delta p$ (Pa)")

    colors = ("#0072B2", "#D55E00", "#000000")
    markers = ("o", "s", "D")
    groups: dict[tuple[float, float], list[dict[str, float | str]]] = {}
    for row in rows:
        key = (
            float(row["inlet_velocity_m_s"]),
            float(row["solid_heat_source_MW_m3"]),
        )
        groups.setdefault(key, []).append(row)
    for index, (key, members) in enumerate(sorted(groups.items())):
        members = sorted(members, key=lambda item: float(item["inlet_temperature_K"]))
        x = [float(item["inlet_temperature_K"]) for item in members]
        label = rf"$u={key[0]:.2f}$ m s$^{{-1}}$, $q={key[1]:.2f}$ MW m$^{{-3}}$"
        style = {
            "color": colors[index % len(colors)],
            "marker": markers[index % len(markers)],
            "markersize": 5,
            "linewidth": 1.1 if len(members) > 1 else 0.0,
            "label": label,
        }
        axes[0, 1].plot(
            x, [float(item["outlet_temperature_K"]) for item in members], **style
        )
        axes[1, 0].plot(
            x,
            [float(item["solid_maximum_temperature_K"]) for item in members],
            **style,
        )
        axes[1, 1].plot(
            x,
            [float(item["cooling_wall_heat_over_generated"]) for item in members],
            **style,
        )
    axes[0, 1].set_xlabel(r"$T_{\mathrm{in}}$ (K)")
    axes[0, 1].set_ylabel(r"$T_{\mathrm{out}}$ (K)")
    axes[1, 0].set_xlabel(r"$T_{\mathrm{in}}$ (K)")
    axes[1, 0].set_ylabel(r"$T_{\mathrm{s,max}}$ (K)")
    axes[1, 1].set_xlabel(r"$T_{\mathrm{in}}$ (K)")
    axes[1, 1].set_ylabel(r"$Q_{\mathrm{wall\rightarrow fluid}}/Q_{\mathrm{gen}}$")
    axes[1, 1].axhline(0.0, color="0.35", linestyle="--", linewidth=0.8)
    axes[0, 1].legend(frameon=False, loc="best")
    for label, axis in zip(("(a)", "(b)", "(c)", "(d)"), axes.ravel()):
        axis.text(-0.18, 1.05, label, transform=axis.transAxes, fontweight="bold")
        axis.minorticks_on()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--time", default="300")
    parser.add_argument("--time-from-completion-marker", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = completed_rows(
        args.matrix_root.resolve(),
        None if args.time_from_completion_marker else str(args.time),
        time_from_completion_marker=args.time_from_completion_marker,
    )
    pairs = temperature_pairs(rows)
    source_pairs = heat_source_pairs(rows)
    flow_pairs = velocity_pairs(rows)
    wall_crossings = wall_heat_zero_crossings(rows)
    wall_crossing_responses = wall_heat_zero_crossing_responses(wall_crossings)
    regimes = thermal_regime_summary(rows)
    parameters = load_hccb_thermophysical_parameters()
    transition_records = steady_transition_proximity(
        rows, parameters.solid_transition_temperatures_k
    )
    trend_checks = physical_trend_checks(rows, pairs, source_pairs, flow_pairs)
    linearity_records = single_factor_linearity(rows)
    linearity_summary = summarize_single_factor_linearity(linearity_records)
    factorial_effects = factorial_variance_decomposition(rows)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "completed_case_physics.csv", rows)
    write_optional_csv(output / "paired_temperature_response.csv", pairs)
    write_optional_csv(output / "paired_heat_source_response.csv", source_pairs)
    write_optional_csv(output / "paired_velocity_response.csv", flow_pairs)
    write_optional_csv(output / "single_factor_linearity.csv", linearity_records)
    write_optional_csv(
        output / "single_factor_linearity_summary.csv", linearity_summary
    )
    write_optional_csv(output / "wall_heat_zero_crossings.csv", wall_crossings)
    write_optional_csv(
        output / "wall_heat_zero_crossing_responses.csv",
        wall_crossing_responses,
    )
    write_optional_csv(
        output / "factorial_variance_decomposition.csv", factorial_effects
    )
    write_optional_csv(
        output / "steady_transition_proximity.csv", transition_records
    )
    closest_transition_records = []
    for index, transition in enumerate(
        parameters.solid_transition_temperatures_k, start=1
    ):
        closest = min(
            transition_records,
            key=lambda item: abs(
                float(item[f"transition_{index}_minus_solid_maximum_K"])
            ),
        )
        closest_transition_records.append(
            {
                "transition_temperature_K": float(transition),
                "closest_condition_id": str(closest["condition_id"]),
                "closest_solid_maximum_temperature_K": float(
                    closest["solid_maximum_temperature_K"]
                ),
                "transition_minus_solid_maximum_K": float(
                    closest[f"transition_{index}_minus_solid_maximum_K"]
                ),
                "completed_case_count_reaching_transition": sum(
                    bool(item[f"transition_{index}_reached_by_solid_maximum"])
                    for item in transition_records
                ),
            }
        )
    summary = {
        "status": "completed_p418_case_physics_summarized",
        "completed_case_count": len(rows),
        "temperature_pair_count": len(pairs),
        "heat_source_pair_count": len(source_pairs),
        "velocity_pair_count": len(flow_pairs),
        "wall_heat_zero_crossing_count": len(wall_crossings),
        "wall_heat_zero_crossing_response_count": len(wall_crossing_responses),
        "single_factor_linearity_interior_point_count": len(linearity_records),
        "complete_factorial_decomposition_available": bool(factorial_effects),
        "maximum_relative_mass_difference": max(
            float(row["relative_mass_difference"]) for row in rows
        ),
        "maximum_relative_energy_difference": max(
            float(row["relative_energy_difference"]) for row in rows
        ),
        "temperature_pairs": pairs,
        "heat_source_pairs": source_pairs,
        "velocity_pairs": flow_pairs,
        "wall_heat_zero_crossings": wall_crossings,
        "wall_heat_zero_crossing_responses": wall_crossing_responses,
        "thermal_regime_summary": regimes,
        "steady_transition_proximity": {
            "transition_temperatures_K": list(
                parameters.solid_transition_temperatures_k
            ),
            "closest_completed_cases": closest_transition_records,
            "definition": (
                "Each published P431 transition temperature is compared directly "
                "with the computed steady solid maximum. No user-defined transition "
                "band is introduced. The smoothed P428-P429 heat-capacity relation "
                "does not resolve the sharp anomalies at these temperatures."
            ),
        },
        "physical_trend_checks": trend_checks,
        "single_factor_linearity_summary": linearity_summary,
        "factorial_variance_decomposition": factorial_effects,
        "interpretation_boundary": (
            "Only cases that hold the other two operating variables fixed are joined. "
            "Each response uses the lowest and highest currently available value of the varied input. "
            "For slices with at least three input levels, interior-point departure from endpoint linear interpolation is reported without imposing a pass threshold. "
            "Zero-wall-heat temperatures are linearly located only inside adjacent computed inlet-temperature brackets; their responses to velocity and heat source compare computed crossing extrema and are not material constants. "
            "A partial completed-case result is preliminary and is not the 60-case model comparison."
        ),
        "new_physical_parameters": [],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_rows(rows, output / "completed_case_physics")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
