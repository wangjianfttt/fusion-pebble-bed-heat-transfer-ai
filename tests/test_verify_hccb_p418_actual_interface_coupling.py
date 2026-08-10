#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from openfoam13_face_flux_reconstruction import coupled_temperature_interface  # noqa: E402
from verify_hccb_p418_actual_interface_coupling import interface_metrics  # noqa: E402


class ActualInterfaceCouplingTest(unittest.TestCase):
    def test_known_two_resistance_interface_is_common_and_reciprocal(self) -> None:
        interface_temperature, fluid_heat, solid_heat = coupled_temperature_interface(
            fluid_cell_temperature=torch.tensor([[500.0]], dtype=torch.float64),
            solid_cell_temperature=torch.tensor([[700.0]], dtype=torch.float64),
            fluid_conductivity=torch.tensor([[2.0]], dtype=torch.float64),
            solid_conductivity=torch.tensor([[4.0]], dtype=torch.float64),
            fluid_cell_centroid=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64),
            solid_cell_centroid=torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64),
            face_centroid=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            fluid_outward_area_vector=torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float64),
        )
        self.assertAlmostEqual(float(interface_temperature[0, 0]), 600.0)
        self.assertAlmostEqual(float(fluid_heat[0, 0]), -600.0)
        self.assertAlmostEqual(float(solid_heat[0, 0]), 600.0)
        metrics = interface_metrics(
            fluid_owner_temperature_k=torch.tensor([[500.0]], dtype=torch.float64),
            solid_owner_temperature_k=torch.tensor([[700.0]], dtype=torch.float64),
            interface_temperature_k=interface_temperature,
            fluid_outward_heat_w=fluid_heat,
            solid_outward_heat_w=solid_heat,
        )
        self.assertEqual(metrics["maximum_absolute_flux_sum_W"], 0.0)
        self.assertEqual(metrics["maximum_flux_sum_over_global_interface_flux"], 0.0)
        self.assertEqual(metrics["maximum_interface_temperature_interval_violation_K"], 0.0)
        self.assertGreaterEqual(metrics["minimum_fluid_heat_direction_product_W_K"], 0.0)
        self.assertGreaterEqual(metrics["minimum_solid_heat_direction_product_W_K"], 0.0)

    def test_metrics_reject_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            interface_metrics(
                fluid_owner_temperature_k=torch.zeros((1, 2)),
                solid_owner_temperature_k=torch.zeros((1, 1)),
                interface_temperature_k=torch.zeros((1, 1)),
                fluid_outward_heat_w=torch.zeros((1, 1)),
                solid_outward_heat_w=torch.zeros((1, 1)),
            )


if __name__ == "__main__":
    unittest.main()
