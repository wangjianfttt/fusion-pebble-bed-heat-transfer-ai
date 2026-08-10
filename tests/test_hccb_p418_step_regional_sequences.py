#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from export_hccb_p418_step_regional_sequences import (
    aggregate_regional_mass_flux,
    aggregate_state,
    condition_vector,
    matching_regional_graph,
    numeric_time_directory,
    preserve_openfoam_subface_mass_flux,
    regional_boundary_features,
    validate_sequence_arrays,
)


class P418StepRegionalSequenceTest(unittest.TestCase):
    def test_solved_phi_is_preserved_on_subfaces_with_orientation(self) -> None:
        internal, boundary = preserve_openfoam_subface_mass_flux(
            internal_fine=np.array([1.0, 2.0, 3.0]),
            boundary_fine=np.array([-4.0, 5.0]),
            internal_openfoam_face=np.array([2, 0]),
            internal_orientation=np.array([-1.0, 1.0]),
            boundary_openfoam_face=np.array([1, -1, 0]),
        )
        np.testing.assert_allclose(internal, [-3.0, 1.0])
        np.testing.assert_allclose(boundary, [5.0, 0.0, -4.0])

    def test_solved_phi_is_summed_with_regional_face_orientation(self) -> None:
        internal, boundary = aggregate_regional_mass_flux(
            internal_fine=np.asarray([2.0, 3.0, 99.0]),
            boundary_fine=np.asarray([-4.0, -1.0, 5.0]),
            internal_inverse=np.asarray([0, 0]),
            internal_crossing=np.asarray([True, True, False]),
            internal_orientation=np.asarray([1.0, -1.0]),
            internal_count=1,
            boundary_inverse=np.asarray([0, 0, 1]),
            boundary_count=2,
        )
        np.testing.assert_allclose(internal, [-1.0])
        np.testing.assert_allclose(boundary, [-5.0, 5.0])

    def test_volume_weighted_fluid_and_solid_state(self) -> None:
        state = aggregate_state(
            fluid_velocity=np.asarray([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            fluid_pressure=np.asarray([10.0, 30.0]),
            fluid_temperature=np.asarray([300.0, 500.0]),
            solid_temperature=np.asarray([700.0, 900.0]),
            fluid_volume=np.asarray([1.0, 3.0]),
            solid_volume=np.asarray([1.0, 1.0]),
            fluid_parent=np.asarray([0, 0]),
            solid_parent=np.asarray([1, 1]),
            fluid_global=np.asarray([0]),
            solid_global=np.asarray([1]),
        )
        np.testing.assert_allclose(state[0], [2.5, 0.0, 0.0, 25.0, 450.0])
        np.testing.assert_allclose(state[1], [0.0, 0.0, 0.0, 0.0, 800.0])

    def test_numeric_time_directory_accepts_openfoam_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "25").mkdir()
            self.assertEqual(numeric_time_directory(root, 25.0), "25")

    def test_condition_vector_contains_source_target_and_fixed_boundaries(self) -> None:
        metadata = {
            "source_parameters": {
                "inlet_velocity_m_s": 0.05,
                "inlet_temperature_K": 300.0,
                "solid_heat_source_MW_m3": 4.85,
            },
            "target_parameters": {
                "inlet_velocity_m_s": 0.25,
                "inlet_temperature_K": 900.0,
                "solid_heat_source_MW_m3": 8.85,
            },
        }
        boundary = {
            "outlet_pressure_Pa": 120000.0,
            "cooling_wall_temperature_K": 635.0,
        }
        np.testing.assert_allclose(
            condition_vector(metadata, boundary),
            [0.05, 300.0, 4.85, 0.25, 900.0, 8.85, 120000.0, 635.0],
        )

    def test_regional_graph_level_is_selected_by_exact_node_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "regional.npz"
            np.savez_compressed(
                path,
                level_0_node_type=np.asarray([0, 1, 1]),
                level_0_edge_source=np.asarray([0, 1]),
                level_0_edge_target=np.asarray([1, 2]),
                level_0_edge_kind=np.asarray([2, 1]),
                level_0_edge_area_m2=np.ones(2),
                level_0_edge_area_vector_m2=np.ones((2, 3)),
                level_0_edge_centroid_m=np.ones((2, 3)),
            )
            level, graph = matching_regional_graph(path, np.asarray([0, 1, 1]))
            self.assertEqual(level, 0)
            self.assertEqual(len(graph["edge_source"]), 2)

    def test_boundary_features_match_selected_regional_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model_geometry.npz"
            np.savez_compressed(
                path,
                boundary_role_names=np.asarray(
                    ["inlet", "cooling_wall", "fluid_solid_interface"]
                ),
                level_2_boundary_volume_fraction=np.asarray(
                    [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]], dtype=np.float32
                ),
            )
            boundary, names = regional_boundary_features(path, 2, 2)
            self.assertEqual(boundary.shape, (2, 3))
            self.assertEqual(names.tolist(), ["inlet", "cooling_wall", "fluid_solid_interface"])
            np.testing.assert_allclose(boundary[:, 2], 0.5)

    def test_fully_coupled_sequence_retains_time_dependent_state_and_flux(self) -> None:
        result = validate_sequence_arrays(
            times=np.asarray([0.0, 0.1, 1.0]),
            state=np.zeros((3, 4, 5)),
            internal_mass_flux=np.zeros((3, 6)),
            boundary_mass_flux=np.zeros((3, 2)),
            history_mode="fully_coupled_flow_heat",
        )
        self.assertTrue(result["mass_flux_time_dependent"])
        self.assertEqual(result["state_channel_count"], 5)

    def test_fixed_hydrodynamics_sequence_keeps_one_flux_field(self) -> None:
        result = validate_sequence_arrays(
            times=np.asarray([0.0, 1.0]),
            state=np.zeros((2, 4, 5)),
            internal_mass_flux=np.zeros(6),
            boundary_mass_flux=np.zeros(2),
            history_mode="fixed_hydrodynamics_thermal",
        )
        self.assertFalse(result["mass_flux_time_dependent"])

    def test_sequence_rejects_nonincreasing_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "increase strictly"):
            validate_sequence_arrays(
                times=np.asarray([0.0, 0.0]),
                state=np.zeros((2, 4, 5)),
                internal_mass_flux=np.zeros((2, 6)),
                boundary_mass_flux=np.zeros((2, 2)),
                history_mode="fully_coupled_flow_heat",
            )


if __name__ == "__main__":
    unittest.main()
