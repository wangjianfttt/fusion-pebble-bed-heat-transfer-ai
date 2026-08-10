#!/usr/bin/env python3
"""Compare completed P418 fields with the source low-Re heat-transfer relation.

The source paper names ``Re_p,AVE`` but does not publish its spatial averaging
equation.  This script therefore keeps two transparent quantities separate:

* a volume average of the local Reynolds number based on ``|U|``; and
* a through-flow Reynolds number based on the net axial mass flux and the
  volume-averaged viscosity.

Only the latter is used as the coordinate for the source-correlation reference.
The former remains a useful measure of transverse pore flow and channeling.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PARAMETER_IDS = ("P048", "P070", "P071", "P388", "P417", "P419")


def steady_iteration_record(marker: dict[str, object]) -> tuple[int, str, object]:
    semantics = str(
        marker.get("solver_time_semantics", "steady_iteration_index")
    )
    if semantics != "steady_iteration_index":
        raise ValueError(
            "dimensionless steady analysis requires a steady-iteration result"
        )
    raw_iteration = marker.get(
        "reported_iteration",
        marker.get("steady_iteration_end", marker.get("time")),
    )
    try:
        iteration = int(float(raw_iteration))
    except (TypeError, ValueError) as exc:
        raise ValueError("steady iteration is missing or invalid") from exc
    return iteration, semantics, marker.get("physical_time_s")


def read_parameters(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = {row["parameter_id"]: row for row in csv.DictReader(stream)}
    missing = [item for item in PARAMETER_IDS if item not in rows]
    if missing:
        raise ValueError(f"missing literature parameters: {missing}")
    return rows


def helium_viscosity(temperature_k: np.ndarray) -> np.ndarray:
    """P070 helium viscosity correlation."""
    return 0.4646 * np.asarray(temperature_k, dtype=np.float64) ** 0.66 * 1.0e-6


def helium_conductivity(
    temperature_k: np.ndarray, pressure_pa: np.ndarray
) -> np.ndarray:
    """P071 helium conductivity correlation."""
    temperature_k = np.asarray(temperature_k, dtype=np.float64)
    pressure_mpa = np.asarray(pressure_pa, dtype=np.float64) / 1.0e6
    theta = temperature_k / 273.0
    return 0.1448 * theta**0.68 * (
        1.0 + 2.5e-3 * pressure_mpa**1.17 * theta**-1.85
    )


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.sum(values * weights) / np.sum(weights))


def analyze_arrays(
    arrays: dict[str, np.ndarray],
    *,
    particle_diameter_m: float,
    fluid_cp_j_kg_k: float,
    solid_heat_source_w_m3: float,
    flow_axis: int = 2,
    interphase_heat_into_fluid_w: float | None = None,
    solid_wall_heat_into_solid_w: float | None = None,
) -> dict[str, float]:
    fluid_volume = np.asarray(arrays["fluid_cell_volume_m3"], dtype=np.float64)
    solid_volume = np.asarray(arrays["solid_cell_volume_m3"], dtype=np.float64)
    velocity_vector = np.asarray(arrays["fluid_velocity_m_s"], dtype=np.float64)
    if flow_axis < 0 or flow_axis >= velocity_vector.shape[1]:
        raise ValueError(f"flow_axis {flow_axis} is outside the velocity vector")
    velocity_magnitude = np.linalg.norm(velocity_vector, axis=1)
    axial_velocity = velocity_vector[:, flow_axis]
    pressure = np.asarray(arrays["fluid_pressure_Pa"], dtype=np.float64)
    fluid_temperature = np.asarray(arrays["fluid_temperature_K"], dtype=np.float64)
    solid_temperature = np.asarray(arrays["solid_temperature_K"], dtype=np.float64)
    density = np.asarray(arrays["fluid_density_kg_m3"], dtype=np.float64)
    interface_area = float(
        np.sum(np.asarray(arrays["interface_face_area_m2"], dtype=np.float64))
    )
    viscosity = helium_viscosity(fluid_temperature)
    conductivity = helium_conductivity(fluid_temperature, pressure)
    local_reynolds_magnitude = (
        density * velocity_magnitude * particle_diameter_m / viscosity
    )
    local_prandtl = fluid_cp_j_kg_k * viscosity / conductivity
    reynolds_local_magnitude_average = weighted_mean(
        local_reynolds_magnitude, fluid_volume
    )
    viscosity_average = weighted_mean(viscosity, fluid_volume)
    conductivity_average = weighted_mean(conductivity, fluid_volume)
    density_average = weighted_mean(density, fluid_volume)
    axial_velocity_average = weighted_mean(axial_velocity, fluid_volume)
    axial_velocity_magnitude_average = weighted_mean(
        np.abs(axial_velocity), fluid_volume
    )
    velocity_magnitude_average = weighted_mean(velocity_magnitude, fluid_volume)
    axial_mass_flux_average = weighted_mean(
        density * axial_velocity, fluid_volume
    )
    reynolds_axial_throughflow = (
        abs(axial_mass_flux_average) * particle_diameter_m / viscosity_average
    )
    prandtl_local_average = weighted_mean(local_prandtl, fluid_volume)
    prandtl_mean_properties = (
        fluid_cp_j_kg_k * viscosity_average / conductivity_average
    )
    peclet_local_magnitude_average = weighted_mean(
        local_reynolds_magnitude * local_prandtl, fluid_volume
    )
    fluid_temperature_average = weighted_mean(fluid_temperature, fluid_volume)
    solid_temperature_average = weighted_mean(solid_temperature, solid_volume)
    phase_temperature_difference = (
        solid_temperature_average - fluid_temperature_average
    )
    solid_total_volume = float(np.sum(solid_volume))
    generated_power = solid_total_volume * solid_heat_source_w_m3
    nusselt_p417_throughflow = (
        9.405
        * reynolds_axial_throughflow**2.322
        * prandtl_mean_properties**7.427
        + 0.264
    )
    nusselt_p417_local_magnitude_sensitivity = (
        9.405
        * reynolds_local_magnitude_average**2.322
        * prandtl_local_average**7.427
        + 0.264
    )
    p419_applicable = phase_temperature_difference > 0.0
    p417_reynolds_applicable = reynolds_axial_throughflow < 1.8
    if p419_applicable:
        interphase_htc = generated_power / (
            interface_area * phase_temperature_difference
        )
        nusselt_field = interphase_htc * particle_diameter_m / conductivity_average
        correlation_difference = (
            100.0
            * (nusselt_p417_throughflow - nusselt_field)
            / nusselt_field
        )
        local_magnitude_correlation_difference = (
            100.0
            * (nusselt_p417_local_magnitude_sensitivity - nusselt_field)
            / nusselt_field
        )
    else:
        interphase_htc = float("nan")
        nusselt_field = float("nan")
        correlation_difference = float("nan")
        local_magnitude_correlation_difference = float("nan")
    resolved_interphase_htc = float("nan")
    resolved_interphase_nusselt = float("nan")
    interphase_heat_over_generated = float("nan")
    solid_wall_heat_over_generated = float("nan")
    solid_energy_partition_error = float("nan")
    interface_flux_and_temperature_sign_agree = False
    if interphase_heat_into_fluid_w is not None:
        interphase_heat_over_generated = interphase_heat_into_fluid_w / generated_power
        if abs(phase_temperature_difference) > np.finfo(np.float64).tiny:
            resolved_interphase_htc = interphase_heat_into_fluid_w / (
                interface_area * phase_temperature_difference
            )
            resolved_interphase_nusselt = (
                resolved_interphase_htc * particle_diameter_m / conductivity_average
            )
            interface_flux_and_temperature_sign_agree = bool(
                resolved_interphase_htc > 0.0
            )
    if solid_wall_heat_into_solid_w is not None:
        solid_wall_heat_over_generated = solid_wall_heat_into_solid_w / generated_power
    if (
        np.isfinite(interphase_heat_over_generated)
        and np.isfinite(solid_wall_heat_over_generated)
    ):
        # Steady solid balance with positive heat entering each region:
        # Qgen + Qwall,into-solid - Qinterface,into-fluid = 0.
        solid_energy_partition_error = interphase_heat_over_generated - (
            1.0 + solid_wall_heat_over_generated
        )
    return {
        "reynolds_particle_axial_throughflow": reynolds_axial_throughflow,
        "reynolds_particle_local_magnitude_volume_average": (
            reynolds_local_magnitude_average
        ),
        "prandtl_from_volume_averaged_properties": prandtl_mean_properties,
        "prandtl_local_volume_average": prandtl_local_average,
        "peclet_local_magnitude_volume_average": (
            peclet_local_magnitude_average
        ),
        "fluid_density_volume_average_kg_m3": density_average,
        "fluid_viscosity_volume_average_Pa_s": viscosity_average,
        "fluid_axial_velocity_volume_average_m_s": axial_velocity_average,
        "fluid_abs_axial_velocity_volume_average_m_s": (
            axial_velocity_magnitude_average
        ),
        "fluid_velocity_magnitude_volume_average_m_s": (
            velocity_magnitude_average
        ),
        "fluid_temperature_volume_average_K": fluid_temperature_average,
        "solid_temperature_volume_average_K": solid_temperature_average,
        "phase_temperature_difference_K": phase_temperature_difference,
        "fluid_conductivity_volume_average_W_m_K": conductivity_average,
        "solid_total_volume_m3": solid_total_volume,
        "interface_area_m2": interface_area,
        "generated_power_from_volume_W": generated_power,
        "p419_positive_phase_temperature_difference": p419_applicable,
        "p417_reynolds_below_1p8": p417_reynolds_applicable,
        "p417_p419_comparable": p419_applicable and p417_reynolds_applicable,
        "interphase_htc_W_m2_K": interphase_htc,
        "nusselt_from_resolved_field_P419": nusselt_field,
        "nusselt_from_source_correlation_P417_throughflow_reference": (
            nusselt_p417_throughflow
        ),
        "nusselt_from_source_correlation_P417_local_magnitude_sensitivity": (
            nusselt_p417_local_magnitude_sensitivity
        ),
        "throughflow_correlation_relative_difference_percent": (
            correlation_difference
        ),
        "local_magnitude_correlation_relative_difference_percent": (
            local_magnitude_correlation_difference
        ),
        "openfoam_interphase_heat_into_fluid_W": (
            float(interphase_heat_into_fluid_w)
            if interphase_heat_into_fluid_w is not None
            else float("nan")
        ),
        "openfoam_interphase_heat_over_generated_power": interphase_heat_over_generated,
        "openfoam_solid_wall_heat_into_solid_W": (
            float(solid_wall_heat_into_solid_w)
            if solid_wall_heat_into_solid_w is not None
            else float("nan")
        ),
        "openfoam_solid_wall_heat_over_generated_power": solid_wall_heat_over_generated,
        "openfoam_solid_energy_partition_error_over_generated": (
            solid_energy_partition_error
        ),
        "openfoam_interface_flux_and_phase_temperature_sign_agree": (
            interface_flux_and_temperature_sign_agree
        ),
        "nusselt_from_openfoam_interphase_flux": resolved_interphase_nusselt,
        "htc_from_openfoam_interphase_flux_W_m2_K": resolved_interphase_htc,
    }


def boundary_heat_by_condition(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("cases", [])
    by_condition = {str(row["condition_id"]): row for row in rows}
    if len(by_condition) != len(rows):
        raise ValueError("boundary heat summary contains repeated condition ids")
    return by_condition


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--parameter-manifest", type=Path, required=True)
    parser.add_argument("--boundary-heat-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    parameters = read_parameters(args.parameter_manifest.resolve())
    particle_diameter_m = float(parameters["P048"]["value"]) * 1.0e-3
    fluid_cp = float(parameters["P388"]["value"])
    boundary_heat = boundary_heat_by_condition(
        args.boundary_heat_summary.resolve() if args.boundary_heat_summary else None
    )
    rows: list[dict[str, float | str]] = []
    completion_iterations: set[int] = set()
    for marker_path in sorted(args.matrix_root.resolve().glob("*/formal_sample_complete.json")):
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        sample_path = Path(str(marker["training_sample"]))
        if not sample_path.is_absolute():
            sample_path = marker_path.parent / sample_path
        metadata = json.loads(
            (marker_path.parent / "cht_smoke_metadata.json").read_text(encoding="utf-8")
        )
        condition_id = str(marker["condition_id"])
        heat_row = boundary_heat.get(condition_id)
        if boundary_heat and heat_row is None:
            raise ValueError(f"boundary heat summary lacks {condition_id}")
        fluid_interface_heat = None
        solid_wall_heat = None
        if heat_row is not None:
            fluid_interface_heat = float(
                heat_row["fluid_boundary_heat_flow_into_region_W"]["fluid_to_solid"]
            )
            solid_wall_heat = float(
                heat_row["solid_boundary_heat_flow_into_region_W"]["coolingWall"]
            )
        with np.load(sample_path, allow_pickle=False) as loaded:
            metrics = analyze_arrays(
                {key: loaded[key] for key in loaded.files},
                particle_diameter_m=particle_diameter_m,
                fluid_cp_j_kg_k=fluid_cp,
                solid_heat_source_w_m3=float(metadata["solid_heat_source_W_m3"]),
                interphase_heat_into_fluid_w=fluid_interface_heat,
                solid_wall_heat_into_solid_w=solid_wall_heat,
            )
        completion_iteration, time_semantics, physical_time_s = steady_iteration_record(
            marker
        )
        completion_iterations.add(completion_iteration)
        rows.append(
            {
                "condition_id": condition_id,
                "completion_iteration": completion_iteration,
                "solver_time_semantics": time_semantics,
                "physical_time_s": physical_time_s,
                "inlet_velocity_m_s": float(metadata["inlet_velocity_m_s"]),
                "inlet_temperature_K": float(metadata["inlet_temperature_K"]),
                "solid_heat_source_W_m3": float(metadata["solid_heat_source_W_m3"]),
                **metrics,
            }
        )
    if not rows:
        raise ValueError("no completed P418 samples were found")
    p419_rows = [
        row
        for row in rows
        if bool(row["p419_positive_phase_temperature_difference"])
        and np.isfinite(float(row["nusselt_from_resolved_field_P419"]))
    ]
    comparable_rows = [row for row in p419_rows if bool(row["p417_p419_comparable"])]
    comparable_within_source_30_percent = [
        row
        for row in comparable_rows
        if abs(
            float(row["throughflow_correlation_relative_difference_percent"])
        )
        <= 30.0
    ]
    p417_out_of_range_rows = [
        row for row in rows if not bool(row["p417_reynolds_below_1p8"])
    ]
    resolved_flux_rows = [
        row
        for row in rows
        if np.isfinite(float(row["nusselt_from_openfoam_interphase_flux"]))
    ]

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "dimensionless_heat_transfer.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "p418_dimensionless_heat_transfer_comparison_complete",
        "case_count": len(rows),
        "p419_positive_phase_difference_case_count": len(p419_rows),
        "p419_nonpositive_phase_difference_case_count": len(rows) - len(p419_rows),
        "p417_p419_in_range_comparable_case_count": len(comparable_rows),
        "p417_reference_within_source_30_percent_case_count": len(
            comparable_within_source_30_percent
        ),
        "p417_reference_within_source_30_percent_fraction": (
            len(comparable_within_source_30_percent) / len(comparable_rows)
            if comparable_rows
            else None
        ),
        "p417_reynolds_out_of_range_case_count": len(p417_out_of_range_rows),
        "completion_iterations": sorted(completion_iterations),
        "solver_time_semantics": "steady_iteration_index",
        "physical_time_s": None,
        "parameter_ids": list(PARAMETER_IDS),
        "particle_diameter_m": particle_diameter_m,
        "fluid_cp_j_kg_k": fluid_cp,
        "reynolds_axial_throughflow_range": [
            min(float(row["reynolds_particle_axial_throughflow"]) for row in rows),
            max(float(row["reynolds_particle_axial_throughflow"]) for row in rows),
        ],
        "reynolds_local_magnitude_volume_average_range": [
            min(
                float(row["reynolds_particle_local_magnitude_volume_average"])
                for row in rows
            ),
            max(
                float(row["reynolds_particle_local_magnitude_volume_average"])
                for row in rows
            ),
        ],
        "prandtl_mean_properties_range": [
            min(
                float(row["prandtl_from_volume_averaged_properties"])
                for row in rows
            ),
            max(
                float(row["prandtl_from_volume_averaged_properties"])
                for row in rows
            ),
        ],
        "resolved_nusselt_range_for_positive_phase_difference": [
            min(float(row["nusselt_from_resolved_field_P419"]) for row in p419_rows),
            max(float(row["nusselt_from_resolved_field_P419"]) for row in p419_rows),
        ] if p419_rows else None,
        "maximum_absolute_in_range_correlation_difference_percent": (
            max(
                abs(
                    float(
                        row[
                            "throughflow_correlation_relative_difference_percent"
                        ]
                    )
                )
                for row in comparable_rows
            )
            if comparable_rows
            else None
        ),
        "maximum_absolute_out_of_re_range_difference_percent": (
            max(
                abs(
                    float(
                        row[
                            "throughflow_correlation_relative_difference_percent"
                        ]
                    )
                )
                for row in p417_out_of_range_rows
                if np.isfinite(
                    float(
                        row[
                            "throughflow_correlation_relative_difference_percent"
                        ]
                    )
                )
            )
            if any(
                np.isfinite(
                    float(
                        row[
                            "throughflow_correlation_relative_difference_percent"
                        ]
                    )
                )
                for row in p417_out_of_range_rows
            )
            else None
        ),
        "openfoam_boundary_heat_summary": (
            str(args.boundary_heat_summary.resolve())
            if args.boundary_heat_summary
            else None
        ),
        "openfoam_interphase_heat_over_generated_power_range": (
            [
                min(float(row["openfoam_interphase_heat_over_generated_power"]) for row in resolved_flux_rows),
                max(float(row["openfoam_interphase_heat_over_generated_power"]) for row in resolved_flux_rows),
            ]
            if resolved_flux_rows
            else None
        ),
        "openfoam_interface_flux_nusselt_range": (
            [
                min(
                    float(row["nusselt_from_openfoam_interphase_flux"])
                    for row in resolved_flux_rows
                ),
                max(
                    float(row["nusselt_from_openfoam_interphase_flux"])
                    for row in resolved_flux_rows
                ),
            ]
            if resolved_flux_rows
            else None
        ),
        "openfoam_interface_flux_case_count": len(resolved_flux_rows),
        "openfoam_solid_wall_heat_over_generated_power_range": (
            [
                min(float(row["openfoam_solid_wall_heat_over_generated_power"]) for row in resolved_flux_rows),
                max(float(row["openfoam_solid_wall_heat_over_generated_power"]) for row in resolved_flux_rows),
            ]
            if resolved_flux_rows
            else None
        ),
        "openfoam_interface_flux_sign_consistent_case_count": sum(
            bool(row["openfoam_interface_flux_and_phase_temperature_sign_agree"])
            for row in resolved_flux_rows
        ),
        "maximum_absolute_openfoam_solid_energy_partition_error_over_generated": (
            max(
                abs(float(row["openfoam_solid_energy_partition_error_over_generated"]))
                for row in resolved_flux_rows
            )
            if resolved_flux_rows
            else None
        ),
        "scope": (
            "P417 is a same-source aggregate correlation. The comparison tests whether "
            "the current smaller resolved crop follows that aggregate relation; it is not "
            "experimental validation and it is not used as a neural-network label. The "
            "source paper names Re_p,AVE and Pr_AVE but does not publish their spatial "
            "averaging equations. The P417 reference here therefore uses net axial mass "
            "flux and volume-averaged viscosity, while the volume average based on local "
            "velocity magnitude is retained separately as a three-dimensional channeling "
            "measure and sensitivity calculation. No coefficient is fitted to force either "
            "definition onto the source figure. Cases "
            "with non-positive phase-average Ts-Tf are reported outside the positive P419 "
            "interphase-heat-transfer definition rather than forced into the correlation. "
            "Cases with Re >= 1.8 are reported separately from the declared P417 range. "
            "P419 uses total solid generation in its published aggregate definition. "
            "The optional OpenFOAM interface-flux quantities are reported separately because "
            "the present local crop also exchanges heat directly with a 635 K wall and with "
            "the inlet stream; they are not substituted into the P417/P419 fit."
        ),
        "source_reynolds_averaging_equation_published": False,
        "p417_reference_reynolds_definition": (
            "abs(volume_average(rho*U_axial))*d_p/volume_average(mu)"
        ),
        "local_3d_reynolds_definition": (
            "volume_average(rho*|U|*d_p/mu)"
        ),
        "new_physical_parameters": [],
        "table": str(csv_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
