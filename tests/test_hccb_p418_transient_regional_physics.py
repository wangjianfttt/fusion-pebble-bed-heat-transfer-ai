#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is tested on the remote machine")
class P418TransientRegionalPhysicsTest(unittest.TestCase):
    def test_fluid_enthalpy_storage_matches_constant_density_heating(self) -> None:
        import torch

        from hccb_p418_transient_regional_physics import (
            conservative_fluid_storage_terms,
        )
        from hccb_source_backed_thermophysical import (
            load_hccb_thermophysical_parameters,
        )

        time = torch.tensor([0.0, 2.0, 5.0], dtype=torch.float64)
        temperature = (400.0 + 3.0 * time)[None, :, None]
        density = 0.08
        pressure = density * temperature * 1.0e6 / 480.19
        state = torch.zeros((1, 3, 1, 5), dtype=torch.float64)
        state[..., 3] = pressure
        state[..., 4] = temperature
        terms = conservative_fluid_storage_terms(state, time)
        cp = load_hccb_thermophysical_parameters().helium_cp_j_kg_k
        torch.testing.assert_close(
            terms.enthalpy_w_m3,
            torch.full_like(terms.enthalpy_w_m3, density * cp * 3.0),
            rtol=1.0e-12,
            atol=1.0e-10,
        )

    def test_fluid_kinetic_storage_matches_accelerating_uniform_gas(self) -> None:
        import torch

        from hccb_p418_transient_regional_physics import (
            conservative_fluid_storage_terms,
        )
        from hccb_source_backed_thermophysical import helium_density

        time = torch.tensor([0.0, 1.0, 3.0, 7.0], dtype=torch.float64)
        velocity = 0.2 + 0.04 * time
        state = torch.zeros((1, 4, 1, 5), dtype=torch.float64)
        state[0, :, 0, 0] = velocity
        state[..., 3] = 120000.0
        state[..., 4] = 700.0
        terms = conservative_fluid_storage_terms(state, time)
        rho = helium_density(
            torch.tensor(120000.0, dtype=torch.float64),
            torch.tensor(700.0, dtype=torch.float64),
        )
        expected = (rho * 0.04 * velocity)[None, :, None]
        torch.testing.assert_close(
            terms.kinetic_w_m3, expected, rtol=1.0e-12, atol=1.0e-12
        )
        torch.testing.assert_close(
            terms.total_w_m3,
            terms.kinetic_w_m3,
            rtol=1.0e-12,
            atol=1.0e-10,
        )

    def test_pressure_work_has_openfoam_sign_at_reference_temperature(self) -> None:
        import torch

        from hccb_p418_transient_regional_physics import (
            conservative_fluid_storage_terms,
        )

        time = torch.tensor([0.0, 2.0, 8.0], dtype=torch.float64)
        state = torch.zeros((1, 3, 1, 5), dtype=torch.float64)
        state[..., 3] = (120000.0 + 25.0 * time)[None, :, None]
        state[..., 4] = 298.15
        terms = conservative_fluid_storage_terms(state, time)
        expected = torch.full_like(terms.total_w_m3, -25.0)
        torch.testing.assert_close(terms.pressure_work_w_m3, expected)
        torch.testing.assert_close(terms.total_w_m3, expected, atol=1.0e-10, rtol=0.0)

    def test_fixed_mass_flux_expands_only_over_time(self) -> None:
        import torch

        from hccb_p418_transient_regional_physics import _expanded_fixed_face_flux

        face_flux = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        expanded = _expanded_fixed_face_flux(
            face_flux, batch=2, time_count=3, face_count=2, name="test flux"
        )
        self.assertEqual(tuple(expanded.shape), (6, 2))
        torch.testing.assert_close(expanded[0], face_flux[0])
        torch.testing.assert_close(expanded[2], face_flux[0])
        torch.testing.assert_close(expanded[3], face_flux[1])

    def test_irregular_time_derivative_is_exact_for_linear_history(self) -> None:
        import torch

        from hccb_p418_transient_regional_physics import time_derivative

        time = torch.tensor([0.0, 1.0, 4.0, 10.0])
        values = 3.0 * time[None, :, None] + torch.tensor([[[2.0]]])
        torch.testing.assert_close(time_derivative(values, time), torch.full_like(values, 3.0))

    def test_target_condition_uses_published_target_and_fixed_boundaries(self) -> None:
        import torch

        from hccb_p418_transient_regional_physics import target_physical_conditions

        step = torch.tensor(
            [[0.05, 300.0, 4.85, 0.25, 900.0, 8.85, 120000.0, 635.0]]
        )
        torch.testing.assert_close(
            target_physical_conditions(step),
            torch.tensor([[0.25, 900.0, 8.85e6, 120000.0, 635.0]]),
        )

    def test_time_derivative_accepts_multiple_curves_and_nodes(self) -> None:
        import torch

        from hccb_p418_transient_regional_physics import time_derivative

        time = torch.tensor([[0.0, 1.0, 4.0], [0.0, 2.0, 5.0]])
        slope = torch.tensor([2.0, 5.0])[:, None, None]
        values = slope * time[:, :, None] + torch.tensor([1.0, 3.0])[:, None, None]
        expected = slope.expand_as(values)
        torch.testing.assert_close(time_derivative(values, time), expected)

    def test_irregular_second_order_derivative_is_exact_for_quadratic_history(self) -> None:
        import torch

        from hccb_p418_transient_regional_physics import time_derivative

        time = torch.tensor([0.0, 1.0, 2.0, 5.0, 30.0, 55.0])
        values = (2.5 * time.square() - 4.0 * time + 7.0)[None, :, None]
        expected = (5.0 * time - 4.0)[None, :, None]
        torch.testing.assert_close(time_derivative(values, time), expected)

    def test_two_times_use_the_available_secant_at_both_endpoints(self) -> None:
        import torch

        from hccb_p418_transient_regional_physics import time_derivative

        time = torch.tensor([2.0, 7.0])
        values = torch.tensor([[[3.0], [18.0]]])
        torch.testing.assert_close(
            time_derivative(values, time), torch.full_like(values, 3.0)
        )


if __name__ == "__main__":
    unittest.main()
