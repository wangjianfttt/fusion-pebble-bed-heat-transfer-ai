#!/usr/bin/env python3
"""Compare a cloud P418 result with the workstation reference summary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


INPUT_QUANTITIES = {
    "inlet_temperature_K": "physical_conditions.inlet_temperature_K",
    "inlet_velocity_m_s": "physical_conditions.inlet_velocity_m_s",
    "cooling_wall_temperature_K": "physical_conditions.cooling_wall_temperature_K",
    "solid_heat_source_W_m3": "physical_conditions.solid_heat_source_W_m3",
}

OUTPUT_QUANTITIES = {
    "pressure_drop_Pa": "flow.pressure_drop_Pa",
    "outlet_average_temperature_K": "temperature.outlet_average_K",
    "maximum_solid_temperature_K": "temperature.solid_maximum_K",
    "cooling_wall_heat_flow_W": "heat_balance.cooling_wall_heat_flow_W",
    "solid_generated_power_W": "heat_balance.solid_generated_power_W",
    "relative_mass_difference": "flow.relative_mass_difference",
    "relative_energy_difference": "heat_balance.relative_energy_difference",
}


def nested(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(path)
        value = value[key]
    return value


def finite_number(payload: dict[str, Any], path: str) -> float:
    value = nested(payload, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} is not finite: {result}")
    return result


def compare(reference: dict[str, Any], cloud: dict[str, Any]) -> dict[str, Any]:
    input_rows = []
    for name, path in INPUT_QUANTITIES.items():
        reference_value = finite_number(reference, path)
        cloud_value = finite_number(cloud, path)
        same = math.isclose(
            reference_value,
            cloud_value,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        input_rows.append(
            {
                "quantity": name,
                "reference": reference_value,
                "cloud": cloud_value,
                "same_input": same,
            }
        )

    output_rows = []
    for name, path in OUTPUT_QUANTITIES.items():
        reference_value = finite_number(reference, path)
        cloud_value = finite_number(cloud, path)
        signed = cloud_value - reference_value
        scale = max(abs(reference_value), abs(cloud_value), 1.0e-300)
        output_rows.append(
            {
                "quantity": name,
                "reference": reference_value,
                "cloud": cloud_value,
                "signed_difference": signed,
                "absolute_difference": abs(signed),
                "relative_difference": abs(signed) / scale,
            }
        )

    same_inputs = all(bool(row["same_input"]) for row in input_rows)
    solver_finished = bool(reference.get("solver_finished")) and bool(
        cloud.get("solver_finished")
    )
    complete = same_inputs and solver_finished
    largest = max(output_rows, key=lambda row: float(row["relative_difference"]))
    return {
        "status": (
            "cloud_reference_result_comparison_complete"
            if complete
            else "cloud_reference_result_incomplete_or_input_mismatch"
        ),
        "same_physical_inputs": same_inputs,
        "both_solvers_finished": solver_finished,
        "physical_input_comparison": input_rows,
        "result_comparison": output_rows,
        "largest_relative_result_difference": {
            "quantity": largest["quantity"],
            "value": largest["relative_difference"],
        },
        "interpretation": (
            "The listed differences compare two numerical runs with the same declared "
            "physical inputs. No new acceptance percentage is introduced; pressure, "
            "temperature, heat-flow and conservation differences are reported directly."
        ),
        "new_physical_parameters": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cloud", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference_path = args.reference.resolve()
    cloud_path = args.cloud.resolve()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    cloud = json.loads(cloud_path.read_text(encoding="utf-8"))
    result = compare(reference, cloud)
    result["reference_summary"] = str(reference_path)
    result["cloud_summary"] = str(cloud_path)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "cloud_reference_result_comparison_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
