#!/usr/bin/env python3
"""Compare the source-defined full HCCB domain with the fine local crop."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMETERS = ROOT / "parameters/literature_parameter_manifest.csv"


def parameter_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["parameter_id"]: row for row in csv.DictReader(handle)}


def relative_difference(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-300)


def require_close(name: str, value: float, reference: float) -> None:
    if not math.isclose(value, reference, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError(f"{name} differs between full and local calculations: {value} != {reference}")


def compare(
    full_result_path: Path,
    local_result_path: Path,
    local_mesh_manifest_path: Path,
    parameters_path: Path,
) -> dict:
    full = json.loads(full_result_path.read_text(encoding="utf-8"))
    local = json.loads(local_result_path.read_text(encoding="utf-8"))
    local_mesh = json.loads(local_mesh_manifest_path.read_text(encoding="utf-8"))
    rows = parameter_rows(parameters_path)

    if full.get("status") != "steady_CHT_result_computed_pending_mesh_and_seed_sensitivity":
        raise ValueError("full-domain result has not completed its physical and conservation checks")
    if not local.get("solver_finished") or not local.get("all_reported_values_are_finite"):
        raise ValueError("local-domain reference result is incomplete")

    full_manifest = full["observables"]
    local_condition = local["physical_conditions"]
    require_close("inlet velocity", local_condition["inlet_velocity_m_s"], 0.20)
    require_close("inlet temperature", local_condition["inlet_temperature_K"], 700.0)
    require_close("solid heat source", local_condition["solid_heat_source_W_m3"], 6.85e6)
    require_close("full inlet temperature", full_manifest["inlet_temperature_K"], 700.0)

    physical_dp_m = float(rows["P048"]["value"]) * 1.0e-3
    solid_k = float(rows["P092"]["value"])
    heat_source = local_condition["solid_heat_source_W_m3"]
    temperature_scale = heat_source * physical_dp_m**2 / solid_k
    cooling_wall_temperature = float(rows["P055"]["value"])

    crop = [float(value) for value in local_mesh["crop_box_dp"]]
    local_flow_length_dp = crop[5] - crop[4]
    local_flow_length_m = local_flow_length_dp * physical_dp_m
    full_total_flow_length_m = 30.0 * physical_dp_m
    full_packed_length_m = 10.0 * physical_dp_m

    full_pressure_drop = float(full_manifest["pressure_drop_Pa"])
    local_pressure_drop = float(local["flow"]["pressure_drop_Pa"])
    full_tmax = float(full_manifest["maximum_solid_temperature_K"])
    local_tmax = float(local["temperature"]["solid_maximum_K"])
    full_tout = float(full_manifest["outlet_temperature_K"])
    local_tout = float(local["temperature"]["outlet_average_K"])
    full_generated_power = float(full["conservation"]["generated_power_W"])
    local_generated_power = float(local["heat_balance"]["solid_generated_power_W"])
    full_cooling_power = float(
        full["conservation"]["fluid_wall_heat_flux_integrals_W"].get("coolingWall", 0.0)
    )
    local_cooling_power = float(local["heat_balance"]["cooling_wall_heat_flow_W"])

    result = {
        "status": "hccb_p418_full_and_local_domain_compared",
        "same_operating_condition": {
            "inlet_velocity_m_s": 0.20,
            "inlet_temperature_K": 700.0,
            "solid_heat_source_MW_m3": 6.85,
            "cooling_wall_temperature_K": cooling_wall_temperature,
        },
        "geometry": {
            "physical_particle_diameter_m": physical_dp_m,
            "full_packed_region_dp": [12.5, 12.5, 10.0],
            "full_total_flow_length_dp": 30.0,
            "local_crop_box_dp": crop,
            "local_flow_length_dp": local_flow_length_dp,
        },
        "full_domain": {
            "pressure_drop_Pa": full_pressure_drop,
            "domain_average_pressure_gradient_Pa_m": full_pressure_drop / full_total_flow_length_m,
            "published_pressure_drop_Pa": float(full_manifest["reference_pressure_drop_Pa"]),
            "published_pressure_drop_relative_difference": float(
                full_manifest["pressure_drop_relative_error"]
            ),
            "outlet_temperature_K": full_tout,
            "maximum_solid_temperature_K": full_tmax,
            "published_maximum_temperature_K": float(
                full_manifest["reference_maximum_temperature_K"]
            ),
            "published_maximum_temperature_relative_difference": float(
                full_manifest["maximum_temperature_relative_error"]
            ),
            "maximum_temperature_above_wall_K": full_tmax - cooling_wall_temperature,
            "dimensionless_maximum_temperature": (
                full_tmax - cooling_wall_temperature
            ) / temperature_scale,
            "generated_power_W": full_generated_power,
            "cooling_wall_heat_into_fluid_W": full_cooling_power,
            "cooling_wall_heat_over_generated_power": full_cooling_power / full_generated_power,
            "relative_mass_imbalance": float(full_manifest["relative_mass_imbalance"]),
            "relative_energy_imbalance": float(
                full["conservation"]["combined_energy_residual_relative"]
            ),
        },
        "local_domain": {
            "pressure_drop_Pa": local_pressure_drop,
            "domain_average_pressure_gradient_Pa_m": local_pressure_drop / local_flow_length_m,
            "outlet_temperature_K": local_tout,
            "maximum_solid_temperature_K": local_tmax,
            "maximum_temperature_above_wall_K": local_tmax - cooling_wall_temperature,
            "dimensionless_maximum_temperature": (
                local_tmax - cooling_wall_temperature
            ) / temperature_scale,
            "generated_power_W": local_generated_power,
            "cooling_wall_heat_into_fluid_W": local_cooling_power,
            "cooling_wall_heat_over_generated_power": local_cooling_power / local_generated_power,
            "relative_mass_imbalance": float(local["flow"]["relative_mass_difference"]),
            "relative_energy_imbalance": float(local["heat_balance"]["relative_energy_difference"]),
        },
        "local_relative_to_full": {
            "domain_average_pressure_gradient_relative_difference": relative_difference(
                local_pressure_drop / local_flow_length_m,
                full_pressure_drop / full_total_flow_length_m,
            ),
            "outlet_temperature_rise_relative_difference": relative_difference(
                local_tout - 700.0,
                full_tout - 700.0,
            ),
            "dimensionless_maximum_temperature_relative_difference": relative_difference(
                (local_tmax - cooling_wall_temperature) / temperature_scale,
                (full_tmax - cooling_wall_temperature) / temperature_scale,
            ),
            "cooling_wall_heat_fraction_relative_difference": relative_difference(
                local_cooling_power / local_generated_power,
                full_cooling_power / full_generated_power,
            ),
        },
        "length_interpretation": {
            "full_total_flow_length_m": full_total_flow_length_m,
            "full_packed_length_m": full_packed_length_m,
            "local_flow_length_m": local_flow_length_m,
            "pressure_gradient_note": (
                "The reported full-domain average includes the published 10dp inlet and "
                "outlet extensions. It is not called a packed-bed-only pressure gradient."
            ),
        },
        "physical_parameter_ids": ["P048", "P055", "P092", "P390", "P391", "P418"],
        "new_physical_parameters": [],
        "claim_boundary": (
            "This is a one-condition domain-size and published-reference comparison. "
            "It does not replace the 60-condition local-domain training set or prove "
            "full-domain accuracy over the complete operating matrix."
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-result", type=Path, required=True)
    parser.add_argument("--local-result", type=Path, required=True)
    parser.add_argument("--local-mesh-manifest", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, default=DEFAULT_PARAMETERS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        args.full_result.resolve(),
        args.local_result.resolve(),
        args.local_mesh_manifest.resolve(),
        args.parameters.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
