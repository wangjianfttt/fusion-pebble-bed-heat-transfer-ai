#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required")
class P418FullyCoupledTransientPhysicsTest(unittest.TestCase):
    @staticmethod
    def geometry():
        import torch

        from hccb_multiregion_steady_cht_residual import CoupledInterfaceMap, RegionMesh
        from hccb_p418_regional_cht_adapter import P418SubfaceGeometry

        dtype = torch.float64
        fluid_mesh = RegionMesh(
            cell_centroid=torch.tensor(
                [[0.0, 0.0, 0.5], [0.0, 0.0, 1.5]], dtype=dtype
            ),
            cell_volume=torch.ones(2, dtype=dtype),
            internal_face_centroid=torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype),
            internal_area_vector=torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype),
            internal_owner=torch.tensor([0]),
            internal_neighbour=torch.tensor([1]),
            boundary_face_centroid=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0],
                    [-0.5, 0.0, 0.5],
                    [0.5, 0.0, 1.5],
                    [0.0, 0.5, 0.5],
                ],
                dtype=dtype,
            ),
            boundary_area_vector=torch.tensor(
                [
                    [0.0, 0.0, -1.0],
                    [0.0, 0.0, 1.0],
                    [-1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=dtype,
            ),
            boundary_owner=torch.tensor([0, 1, 0, 1, 0]),
        )
        solid_mesh = RegionMesh(
            cell_centroid=torch.tensor([[0.0, 1.0, 0.5]], dtype=dtype),
            cell_volume=torch.ones(1, dtype=dtype),
            internal_face_centroid=torch.empty((0, 3), dtype=dtype),
            internal_area_vector=torch.empty((0, 3), dtype=dtype),
            internal_owner=torch.empty(0, dtype=torch.long),
            internal_neighbour=torch.empty(0, dtype=torch.long),
            boundary_face_centroid=torch.tensor(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0],
                    [0.5, 1.0, 0.5],
                    [-0.5, 1.0, 0.5],
                    [0.0, 0.5, 0.5],
                ],
                dtype=dtype,
            ),
            boundary_area_vector=torch.tensor(
                [
                    [0.0, 0.0, -1.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                ],
                dtype=dtype,
            ),
            boundary_owner=torch.zeros(5, dtype=torch.long),
        )
        return P418SubfaceGeometry(
            fluid_mesh=fluid_mesh,
            solid_mesh=solid_mesh,
            interface=CoupledInterfaceMap(
                fluid_boundary_face=torch.tensor([4]),
                solid_boundary_face=torch.tensor([4]),
            ),
            fluid_boundary_patch=torch.arange(5),
            solid_boundary_patch=torch.arange(5),
            fluid_patch_names=(
                "inlet",
                "outlet",
                "coolingWall",
                "symmetryWalls",
                "fluid_to_solid",
            ),
            solid_patch_names=(
                "inlet",
                "outlet",
                "coolingWall",
                "symmetryWalls",
                "solid_to_fluid",
            ),
            fine_to_regional_global=np.array([0, 1, 2]),
            fluid_global_region=np.array([0, 1]),
            solid_global_region=np.array([2]),
        )

    def reference_history(self):
        import torch

        from hccb_p418_fully_coupled_transient_physics import (
            assemble_p418_fully_coupled_transient_residual,
        )
        from hccb_source_backed_thermophysical import helium_density

        geometry = self.geometry()
        time = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
        state = torch.zeros((1, 3, 3, 5), dtype=torch.float64)
        state[:, :, :2, 2] = (0.2 + 0.01 * time)[None, :, None]
        state[:, :, :2, 3] = (120000.0 + 100.0 * time)[None, :, None]
        state[:, :, :2, 4] = (700.0 + 5.0 * time)[None, :, None]
        state[:, :, 2, 4] = (650.0 + 3.0 * time)[None, :]
        step = torch.tensor(
            [[0.2, 700.0, 4.85, 0.25, 900.0, 8.85, 120000.0, 635.0]],
            dtype=torch.float64,
        )
        density = helium_density(state[:, :, 0, 3], state[:, :, 0, 4])
        mass_flow = density * state[:, :, 0, 2]
        internal_flux = mass_flow[..., None]
        boundary_flux = torch.zeros((1, 3, 5), dtype=torch.float64)
        boundary_flux[..., 0] = -mass_flow
        boundary_flux[..., 1] = mass_flow
        residual = assemble_p418_fully_coupled_transient_residual(
            geometry=geometry,
            step_condition=step,
            state_physical=state,
            time_s=time,
            fluid_internal_mass_flux_kg_s=internal_flux,
            fluid_boundary_mass_flux_kg_s=boundary_flux,
        )
        return geometry, time, step, state, internal_flux, boundary_flux, residual

    def test_density_and_momentum_storage_follow_conservative_variables(self) -> None:
        import torch

        from hccb_p418_fully_coupled_transient_physics import (
            density_and_momentum_storage,
        )
        from hccb_source_backed_thermophysical import helium_density

        time = torch.tensor([0.0, 1.5, 4.0], dtype=torch.float64)
        state = torch.zeros((1, 3, 1, 5), dtype=torch.float64)
        state[..., 0] = (0.2 + 0.03 * time)[None, :, None]
        state[..., 3] = 120000.0
        state[..., 4] = 700.0
        density_storage, momentum_storage = density_and_momentum_storage(state, time)
        rho = helium_density(
            torch.tensor(120000.0, dtype=torch.float64),
            torch.tensor(700.0, dtype=torch.float64),
        )
        torch.testing.assert_close(density_storage, torch.zeros_like(density_storage))
        expected = torch.zeros_like(momentum_storage)
        expected[..., 0] = rho * 0.03
        torch.testing.assert_close(momentum_storage, expected, atol=1.0e-12, rtol=1.0e-12)

    def test_pressure_history_produces_exact_density_storage(self) -> None:
        import torch

        from hccb_p418_fully_coupled_transient_physics import (
            density_and_momentum_storage,
        )

        time = torch.tensor([0.0, 2.0, 7.0], dtype=torch.float64)
        state = torch.zeros((1, 3, 1, 5), dtype=torch.float64)
        state[..., 3] = (120000.0 + 1000.0 * time)[None, :, None]
        state[..., 4] = 700.0
        density_storage, momentum_storage = density_and_momentum_storage(state, time)
        expected = torch.full_like(density_storage, 480.19 / (1.0e6 * 700.0) * 1000.0)
        torch.testing.assert_close(density_storage, expected, atol=1.0e-14, rtol=1.0e-12)
        torch.testing.assert_close(momentum_storage, torch.zeros_like(momentum_storage))

    def test_constant_history_reduces_to_existing_steady_equations(self) -> None:
        import torch

        from hccb_p418_fully_coupled_transient_physics import (
            P418FullyCoupledEquationScales,
            assemble_p418_fully_coupled_transient_residual,
            dimensionless_fully_coupled_equation_terms,
        )
        from hccb_source_backed_thermophysical import helium_density

        geometry = self.geometry()
        time = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
        state = torch.zeros((1, 3, 3, 5), dtype=torch.float64)
        state[:, :, :2, 2] = 0.2
        state[:, :, :2, 3] = 120000.0
        state[..., 4] = 700.0
        step = torch.tensor(
            [[0.2, 700.0, 0.0, 0.2, 700.0, 0.0, 120000.0, 700.0]],
            dtype=torch.float64,
        )
        rho = helium_density(
            torch.tensor(120000.0, dtype=torch.float64),
            torch.tensor(700.0, dtype=torch.float64),
        )
        mass_flow = rho * 0.2
        internal_flux = mass_flow.expand(1, 3, 1).clone()
        boundary_flux = torch.zeros((1, 3, 5), dtype=torch.float64)
        boundary_flux[..., 0] = -mass_flow
        boundary_flux[..., 1] = mass_flow
        residual = assemble_p418_fully_coupled_transient_residual(
            geometry=geometry,
            step_condition=step,
            state_physical=state,
            time_s=time,
            fluid_internal_mass_flux_kg_s=internal_flux,
            fluid_boundary_mass_flux_kg_s=boundary_flux,
        )
        torch.testing.assert_close(
            residual.density_storage_kg_m3_s,
            torch.zeros_like(residual.density_storage_kg_m3_s),
        )
        torch.testing.assert_close(
            residual.momentum_storage_n_m3,
            torch.zeros_like(residual.momentum_storage_n_m3),
        )
        torch.testing.assert_close(
            residual.continuity_kg_m3_s,
            torch.zeros_like(residual.continuity_kg_m3_s),
            atol=1.0e-14,
            rtol=0.0,
        )
        torch.testing.assert_close(
            residual.momentum_n_m3, residual.steady_momentum_n_m3
        )
        torch.testing.assert_close(
            residual.fluid_energy_w_m3,
            residual.steady_fluid_energy_w_m3,
            atol=1.0e-10,
            rtol=0.0,
        )
        torch.testing.assert_close(
            residual.solid_energy_w_m3,
            residual.steady_solid_energy_w_m3,
            atol=1.0e-10,
            rtol=0.0,
        )
        torch.testing.assert_close(
            residual.internal_mass_flux_consistency_kg_s,
            torch.zeros_like(residual.internal_mass_flux_consistency_kg_s),
            atol=1.0e-14,
            rtol=0.0,
        )
        torch.testing.assert_close(
            residual.boundary_mass_flux_consistency_kg_s,
            torch.zeros_like(residual.boundary_mass_flux_consistency_kg_s),
            atol=1.0e-14,
            rtol=0.0,
        )
        one = torch.tensor(1.0, dtype=torch.float64)
        scales = P418FullyCoupledEquationScales(*(one for _ in range(8)))
        terms = dimensionless_fully_coupled_equation_terms(
            residual=residual,
            scales=scales,
            fluid_volume_m3=geometry.fluid_mesh.cell_volume,
            solid_volume_m3=geometry.solid_mesh.cell_volume,
        )
        self.assertEqual(
            set(terms),
            {
                "continuity",
                "momentum",
                "fluid_energy",
                "solid_energy",
                "interface_flux",
                "interface_temperature",
                "internal_mass_flux",
                "boundary_mass_flux",
            },
        )
        self.assertTrue(all(torch.isfinite(value) for value in terms.values()))

    def test_fully_coupled_assembly_rejects_time_fixed_face_flux(self) -> None:
        import torch

        from hccb_p418_fully_coupled_transient_physics import (
            assemble_p418_fully_coupled_transient_residual,
        )

        with self.assertRaisesRegex(ValueError, "fixed flux is not fully coupled"):
            assemble_p418_fully_coupled_transient_residual(
                geometry=self.geometry(),
                step_condition=torch.tensor(
                    [[0.2, 700.0, 0.0, 0.2, 700.0, 0.0, 120000.0, 700.0]],
                    dtype=torch.float64,
                ),
                state_physical=torch.tensor(
                    [[
                        [[0.0, 0.0, 0.2, 120000.0, 700.0]] * 2
                        + [[0.0, 0.0, 0.0, 0.0, 700.0]],
                        [[0.0, 0.0, 0.2, 120000.0, 700.0]] * 2
                        + [[0.0, 0.0, 0.0, 0.0, 700.0]],
                    ]],
                    dtype=torch.float64,
                ),
                time_s=torch.tensor([0.0, 1.0], dtype=torch.float64),
                fluid_internal_mass_flux_kg_s=torch.zeros((1, 1), dtype=torch.float64),
                fluid_boundary_mass_flux_kg_s=torch.zeros((1, 2, 5), dtype=torch.float64),
            )

    def test_training_scales_and_projection_difference_use_reference_curve(self) -> None:
        import torch

        from hccb_p418_fully_coupled_training import (
            PHYSICS_TERM_NAMES,
            projection_aware_physics_terms,
            training_equation_scales,
        )

        geometry, _, _, state, _, _, residual = self.reference_history()
        scales = training_equation_scales([residual], [state])
        for scale in scales.__dict__.values():
            self.assertTrue(bool(torch.isfinite(scale) & (scale > 0.0)))
        terms = projection_aware_physics_terms(
            prediction=residual,
            reference=residual,
            scales=scales,
            fluid_volume_m3=geometry.fluid_mesh.cell_volume,
            solid_volume_m3=geometry.solid_mesh.cell_volume,
        )
        self.assertEqual(set(terms), set(PHYSICS_TERM_NAMES))
        for value in terms.values():
            torch.testing.assert_close(value, torch.zeros_like(value))

    def test_one_full_state_parameter_update_reaches_all_loss_groups(self) -> None:
        import torch

        from hccb_p418_fully_coupled_spatiotemporal_operator import (
            HCCBP418FullyCoupledRegionalOperator,
            build_p418_fully_coupled_flux_graph,
        )
        from hccb_p418_fully_coupled_training import (
            combine_fully_coupled_loss_groups,
            projection_aware_physics_terms,
            supervised_fully_coupled_terms,
            training_equation_scales,
        )
        from hccb_p418_fully_coupled_transient_physics import (
            assemble_p418_fully_coupled_transient_residual,
        )
        from hccb_p418_spatiotemporal_regional_operator import (
            P418ThermalStepRegionalGraph,
        )

        torch.manual_seed(19)
        geometry, time, step, state, internal, boundary, reference = (
            self.reference_history()
        )
        graph = P418ThermalStepRegionalGraph.from_tensors(
            centroid_m=torch.tensor(
                [[0.0, 0.0, 0.5], [0.0, 0.0, 1.5], [0.0, 1.0, 0.5]],
                dtype=torch.float64,
            ),
            volume_m3=torch.ones(3, dtype=torch.float64),
            node_type=torch.tensor([0, 0, 1]),
            edge_source=torch.tensor([0, 1, 0, 2]),
            edge_target=torch.tensor([1, 0, 2, 0]),
            edge_kind=torch.tensor([0, 0, 2, 2]),
            edge_area_m2=torch.ones(4, dtype=torch.float64),
            edge_area_vector_m2=torch.ones((4, 3), dtype=torch.float64),
        )
        flux_graph = build_p418_fully_coupled_flux_graph(
            geometry=geometry,
            graph=graph,
        )
        self.assertEqual(flux_graph.internal_features.shape, (1, 10))
        self.assertEqual(flux_graph.boundary_features.shape, (5, 12))
        torch.testing.assert_close(
            flux_graph.boundary_active,
            torch.tensor([True, True, False, False, False]),
        )
        model = HCCBP418FullyCoupledRegionalOperator(
            hidden_dim=8,
            local_pre_iterations=1,
            physics_attention_blocks=1,
            local_post_iterations=1,
            physics_attention_heads=1,
            physics_slices=2,
            temporal_layers=1,
            temporal_heads=1,
            temporal_node_chunk_size=None,
        ).to(dtype=torch.float64)
        prediction = model(
            state[:, 0],
            internal[:, 0],
            boundary[:, 0],
            torch.zeros((1, 8), dtype=torch.float64),
            time / time.max(),
            graph,
            flux_graph,
        )
        prediction_residual = assemble_p418_fully_coupled_transient_residual(
            geometry=geometry,
            step_condition=step,
            state_physical=prediction.state,
            time_s=time,
            fluid_internal_mass_flux_kg_s=prediction.internal_mass_flux,
            fluid_boundary_mass_flux_kg_s=prediction.boundary_mass_flux,
        )
        scales = training_equation_scales([reference], [state])
        physics = projection_aware_physics_terms(
            prediction=prediction_residual,
            reference=reference,
            scales=scales,
            fluid_volume_m3=geometry.fluid_mesh.cell_volume,
            solid_volume_m3=geometry.solid_mesh.cell_volume,
        )
        supervised = supervised_fully_coupled_terms(
            predicted_state=prediction.state,
            reference_state=state,
            predicted_internal_mass_flux=prediction.internal_mass_flux,
            reference_internal_mass_flux=internal,
            predicted_boundary_mass_flux=prediction.boundary_mass_flux,
            reference_boundary_mass_flux=boundary,
            node_type=graph.node_type,
            state_scale_by_node=torch.tensor(
                [[0.1, 0.1, 0.1, 1000.0, 100.0]] * 3,
                dtype=torch.float64,
            ),
            internal_mass_flux_scale_kg_s=scales.internal_mass_flux_kg_s,
            boundary_mass_flux_scale_kg_s=scales.boundary_mass_flux_kg_s,
        )
        total, groups = combine_fully_coupled_loss_groups(
            supervised_terms=supervised,
            physics_terms=physics,
            state_weight=1.0,
            face_flux_weight=1.0,
            physics_weight=1.0,
        )
        self.assertEqual(set(groups), {"state_data", "face_flux_data", "physics"})
        self.assertTrue(bool(torch.isfinite(total)))
        before = model.state_change.layers[-1].weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-6)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.assertTrue(
            all(
                parameter.grad is None or torch.all(torch.isfinite(parameter.grad))
                for parameter in model.parameters()
            )
        )
        optimizer.step()
        self.assertFalse(torch.equal(before, model.state_change.layers[-1].weight.detach()))


if __name__ == "__main__":
    unittest.main()
