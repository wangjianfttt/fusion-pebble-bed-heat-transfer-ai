#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from analyze_hccb_p418_dimensionless_heat_transfer import (  # noqa: E402
    analyze_arrays,
    steady_iteration_record,
)


class P418DimensionlessHeatTransferTest(unittest.TestCase):
    def test_steady_iteration_is_not_reported_as_physical_time(self) -> None:
        iteration, semantics, physical_time = steady_iteration_record(
            {
                "time": "200",
                "solver_time_semantics": "steady_iteration_index",
                "physical_time_s": None,
            }
        )
        self.assertEqual(iteration, 200)
        self.assertEqual(semantics, "steady_iteration_index")
        self.assertIsNone(physical_time)

    def test_transient_marker_is_rejected_by_steady_analysis(self) -> None:
        with self.assertRaisesRegex(ValueError, "steady-iteration result"):
            steady_iteration_record(
                {"time": "25", "solver_time_semantics": "physical_time_s"}
            )

    def test_field_definition_and_source_correlation_are_finite(self) -> None:
        arrays = {
            "fluid_cell_volume_m3": np.array([1.0e-9, 2.0e-9]),
            "solid_cell_volume_m3": np.array([1.5e-9, 1.5e-9]),
            "fluid_velocity_m_s": np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]),
            "fluid_pressure_Pa": np.array([120000.0, 119999.0]),
            "fluid_temperature_K": np.array([500.0, 520.0]),
            "solid_temperature_K": np.array([550.0, 560.0]),
            "fluid_density_kg_m3": np.array([0.115, 0.111]),
            "interface_face_area_m2": np.array([1.0e-6, 2.0e-6]),
        }
        result = analyze_arrays(
            arrays,
            particle_diameter_m=1.0e-3,
            fluid_cp_j_kg_k=5200.0,
            solid_heat_source_w_m3=4.85e6,
            flow_axis=0,
            interphase_heat_into_fluid_w=2.0e-3,
            solid_wall_heat_into_solid_w=-1.0e-3,
        )
        self.assertGreater(result["reynolds_particle_axial_throughflow"], 0.0)
        self.assertGreater(
            result["reynolds_particle_local_magnitude_volume_average"], 0.0
        )
        self.assertGreater(result["prandtl_from_volume_averaged_properties"], 0.0)
        self.assertGreater(result["nusselt_from_resolved_field_P419"], 0.0)
        self.assertTrue(result["p417_reynolds_below_1p8"])
        self.assertTrue(result["p417_p419_comparable"])
        self.assertTrue(
            np.isfinite(
                result[
                    "nusselt_from_source_correlation_P417_throughflow_reference"
                ]
            )
        )
        self.assertAlmostEqual(result["solid_total_volume_m3"], 3.0e-9)
        self.assertGreater(result["nusselt_from_openfoam_interphase_flux"], 0.0)
        self.assertTrue(
            result["openfoam_interface_flux_and_phase_temperature_sign_agree"]
        )
        self.assertGreater(
            result["openfoam_interphase_heat_over_generated_power"], 0.0
        )
        self.assertLess(result["openfoam_solid_wall_heat_over_generated_power"], 0.0)

    def test_solid_energy_partition_identity(self) -> None:
        arrays = {
            "fluid_cell_volume_m3": np.array([1.0]),
            "solid_cell_volume_m3": np.array([2.0]),
            "fluid_velocity_m_s": np.array([[0.1, 0.0, 0.0]]),
            "fluid_pressure_Pa": np.array([120000.0]),
            "fluid_temperature_K": np.array([500.0]),
            "solid_temperature_K": np.array([550.0]),
            "fluid_density_kg_m3": np.array([0.1]),
            "interface_face_area_m2": np.array([1.0]),
        }
        generated = 2.0 * 4.85e6
        wall_into_solid = 0.4 * generated
        interface_into_fluid = generated + wall_into_solid
        result = analyze_arrays(
            arrays,
            particle_diameter_m=1.0e-3,
            fluid_cp_j_kg_k=5200.0,
            solid_heat_source_w_m3=4.85e6,
            interphase_heat_into_fluid_w=interface_into_fluid,
            solid_wall_heat_into_solid_w=wall_into_solid,
        )
        self.assertAlmostEqual(
            result["openfoam_solid_energy_partition_error_over_generated"], 0.0
        )

    def test_nonpositive_phase_temperature_difference_is_outside_p419(self) -> None:
        arrays = {
            "fluid_cell_volume_m3": np.array([1.0]),
            "solid_cell_volume_m3": np.array([1.0]),
            "fluid_velocity_m_s": np.array([[0.1, 0.0, 0.0]]),
            "fluid_pressure_Pa": np.array([120000.0]),
            "fluid_temperature_K": np.array([600.0]),
            "solid_temperature_K": np.array([590.0]),
            "fluid_density_kg_m3": np.array([0.1]),
            "interface_face_area_m2": np.array([1.0]),
        }
        result = analyze_arrays(
            arrays,
            particle_diameter_m=1.0e-3,
            fluid_cp_j_kg_k=5200.0,
            solid_heat_source_w_m3=4.85e6,
        )
        self.assertFalse(result["p419_positive_phase_temperature_difference"])
        self.assertTrue(np.isnan(result["nusselt_from_resolved_field_P419"]))
        self.assertGreater(
            result[
                "nusselt_from_source_correlation_P417_throughflow_reference"
            ],
            0.0,
        )

    def test_transverse_pore_flow_does_not_inflate_throughflow_reynolds(self) -> None:
        arrays = {
            "fluid_cell_volume_m3": np.array([1.0, 1.0]),
            "solid_cell_volume_m3": np.array([1.0]),
            "fluid_velocity_m_s": np.array(
                [[4.0, 0.0, 0.1], [-4.0, 0.0, 0.1]]
            ),
            "fluid_pressure_Pa": np.array([120000.0, 120000.0]),
            "fluid_temperature_K": np.array([500.0, 500.0]),
            "solid_temperature_K": np.array([550.0]),
            "fluid_density_kg_m3": np.array([0.1, 0.1]),
            "interface_face_area_m2": np.array([1.0]),
        }
        result = analyze_arrays(
            arrays,
            particle_diameter_m=1.0e-3,
            fluid_cp_j_kg_k=5200.0,
            solid_heat_source_w_m3=4.85e6,
        )
        self.assertGreater(
            result["reynolds_particle_local_magnitude_volume_average"],
            30.0 * result["reynolds_particle_axial_throughflow"],
        )

    def test_signed_interface_flux_gives_positive_coefficient_when_heat_follows_temperature(self) -> None:
        arrays = {
            "fluid_cell_volume_m3": np.array([1.0]),
            "solid_cell_volume_m3": np.array([1.0]),
            "fluid_velocity_m_s": np.array([[0.1, 0.0, 0.0]]),
            "fluid_pressure_Pa": np.array([120000.0]),
            "fluid_temperature_K": np.array([600.0]),
            "solid_temperature_K": np.array([590.0]),
            "fluid_density_kg_m3": np.array([0.1]),
            "interface_face_area_m2": np.array([1.0]),
        }
        result = analyze_arrays(
            arrays,
            particle_diameter_m=1.0e-3,
            fluid_cp_j_kg_k=5200.0,
            solid_heat_source_w_m3=4.85e6,
            interphase_heat_into_fluid_w=-10.0,
        )
        self.assertGreater(result["htc_from_openfoam_interphase_flux_W_m2_K"], 0.0)
        self.assertGreater(result["nusselt_from_openfoam_interphase_flux"], 0.0)
        self.assertTrue(
            result["openfoam_interface_flux_and_phase_temperature_sign_agree"]
        )


if __name__ == "__main__":
    unittest.main()
