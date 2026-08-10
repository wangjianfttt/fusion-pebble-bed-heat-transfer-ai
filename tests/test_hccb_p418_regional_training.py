#!/usr/bin/env python3

from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is tested on the remote compute machine")
class P418RegionalTrainingTest(unittest.TestCase):
    def test_constant_condition_scale_survives_loading(self) -> None:
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "code"))
        from train_hccb_p418_regional_operator import load_scales

        statistics = {
            "splits": {
                "heat_source_extrapolation": {
                    "condition_input": {
                        "mean": [0.15, 600.0, 4.85e6, 120000.0, 635.0],
                        "standard_deviation": [0.1, 100.0, 0.0, 0.0, 0.0],
                    },
                    "targets": {
                        "fluid_velocity_m_s": {
                            "mean": [0.0, 0.0, 0.0],
                            "standard_deviation": [1.0, 1.0, 1.0],
                        },
                        "fluid_gauge_pressure_Pa": {
                            "mean": [0.0],
                            "standard_deviation": [1.0],
                        },
                        "fluid_temperature_K": {
                            "mean": [300.0],
                            "standard_deviation": [1.0],
                        },
                        "solid_temperature_K": {
                            "mean": [500.0],
                            "standard_deviation": [1.0],
                        },
                    },
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "statistics.json"
            path.write_text(json.dumps(statistics), encoding="utf-8")
            scales = load_scales(path, "heat_source_extrapolation")
        self.assertTrue(
            np.array_equal(scales.condition_std, [0.1, 100.0, 0.0, 0.0, 0.0])
        )

    def test_target_scaling_loss_and_source_schedule(self) -> None:
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "code"))
        from train_hccb_p418_regional_operator import (
            EngineeringGeometry,
            FieldScales,
            cell_adjacent_engineering_metrics,
            chunk_loss,
            exact_boundary_reference_metrics,
            normalized_target_chunk,
            normalized_condition,
            source_learning_rate,
        )

        scales = FieldScales(
            condition_mean=np.zeros(5),
            condition_std=np.ones(5),
            velocity_mean=np.zeros(3),
            velocity_std=np.ones(3),
            pressure_mean=0.0,
            pressure_std=2.0,
            fluid_temperature_mean=300.0,
            fluid_temperature_std=10.0,
            solid_temperature_mean=500.0,
            solid_temperature_std=20.0,
        )
        field = {
            "fluid_velocity_m_s": np.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
            "fluid_pressure_Pa": np.asarray([120002.0, 120004.0]),
            "fluid_temperature_K": np.asarray([310.0, 320.0]),
            "solid_temperature_K": np.asarray([520.0, 540.0]),
        }
        target_np, valid_np = normalized_target_chunk(
            field,
            fluid_count=2,
            start=0,
            stop=4,
            outlet_pressure_pa=120000.0,
            scales=scales,
        )
        self.assertTrue(np.allclose(target_np[0], [1.0, 2.0, 3.0, 1.0, 1.0]))
        self.assertEqual(int(valid_np.sum()), 12)
        target = torch.as_tensor(target_np).unsqueeze(0)
        valid = torch.as_tensor(valid_np).unsqueeze(0)
        node_type = torch.tensor([0, 0, 1, 1])
        volume = torch.ones(4)
        denominator = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        loss, channels = chunk_loss(
            target.clone(), target, valid, node_type, volume, denominator
        )
        self.assertEqual(float(loss), 0.0)
        self.assertTrue(torch.equal(channels, torch.zeros(6)))
        self.assertAlmostEqual(source_learning_rate(0, 1001), 1.0e-5)
        self.assertAlmostEqual(source_learning_rate(1000, 1001), 1.0e-6)

        geometry = EngineeringGeometry(
            fluid_boundary_owner=np.array([0, 1]),
            fluid_boundary_patch=np.array([0, 1]),
            fluid_boundary_area_m2=np.array([2.0, 2.0]),
            fluid_boundary_area_vector_m2=np.array(
                [[0.0, 0.0, -2.0], [0.0, 0.0, 2.0]]
            ),
            inlet_patch=0,
            outlet_patch=1,
        )
        engineering = cell_adjacent_engineering_metrics(
            velocity_m_s=np.array(
                [[0.0, 0.0, 0.1], [0.0, 0.0, 0.1]]
            ),
            pressure_pa=np.array([120010.0, 120000.0]),
            fluid_temperature_k=np.array([600.0, 600.0]),
            solid_temperature_k=np.array([610.0, 620.0]),
            geometry=geometry,
        )
        self.assertAlmostEqual(
            engineering["pressure_drop_boundary_adjacent_cells_Pa"], 10.0
        )
        self.assertAlmostEqual(
            engineering["outlet_temperature_boundary_adjacent_cells_K"], 600.0
        )
        self.assertAlmostEqual(
            engineering["solid_maximum_temperature_K"], 620.0
        )
        exact = exact_boundary_reference_metrics(
            {
                "fluid_boundary_pressure_Pa": np.array([120011.0, 120000.0]),
                "fluid_boundary_temperature_K": np.array([600.0, 605.0]),
                "fluid_boundary_face_mass_flow_kg_s": np.array([-2.0, 2.0]),
                "solid_temperature_K": np.array([610.0, 620.0]),
            },
            geometry,
        )
        self.assertAlmostEqual(exact["pressure_drop_boundary_faces_Pa"], 11.0)
        self.assertAlmostEqual(
            exact["outlet_temperature_boundary_faces_K"], 605.0
        )

        constant_input_scales = FieldScales(
            condition_mean=np.array([0.15, 600.0, 4.85e6, 120000.0, 635.0]),
            condition_std=np.array([0.1, 100.0, 0.0, 0.0, 0.0]),
            velocity_mean=np.zeros(3),
            velocity_std=np.ones(3),
            pressure_mean=0.0,
            pressure_std=1.0,
            fluid_temperature_mean=300.0,
            fluid_temperature_std=1.0,
            solid_temperature_mean=500.0,
            solid_temperature_std=1.0,
        )
        condition = normalized_condition(
            {
                "inlet_velocity_m_s": 0.2,
                "inlet_temperature_K": 700.0,
                "solid_heat_source_W_m3": 8.85e6,
                "outlet_pressure_Pa": 120000.0,
                "cooling_wall_temperature_K": 635.0,
            },
            constant_input_scales,
        )
        self.assertTrue(np.allclose(condition, [0.5, 1.0, 0.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
