#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is tested on the remote machine")
class P418SpatiotemporalRegionalOperatorTest(unittest.TestCase):
    def test_temperature_bounds_accept_named_and_legacy_records(self) -> None:
        import numpy as np

        from hccb_p418_spatiotemporal_regional_operator import (
            temperature_output_bounds_by_node_type,
        )

        expected = np.asarray(
            [[300.0, 1000.0], [298.0, 1300.0]], dtype=np.float32
        )
        np.testing.assert_array_equal(
            temperature_output_bounds_by_node_type(
                {
                    "fluid": [300.0, 1000.0],
                    "solid": [298.0, 1300.0],
                }
            ),
            expected,
        )
        np.testing.assert_array_equal(
            temperature_output_bounds_by_node_type(expected.tolist()),
            expected,
        )
        with self.assertRaisesRegex(ValueError, "exactly fluid and solid"):
            temperature_output_bounds_by_node_type(
                {"fluid": [300.0, 1000.0]}
            )

    def test_hard_initial_state_and_fixed_hydrodynamics(self) -> None:
        import torch

        from hccb_p418_spatiotemporal_regional_operator import (
            HCCBP418SpatiotemporalRegionalOperator,
            P418ThermalStepRegionalGraph,
        )

        graph = P418ThermalStepRegionalGraph.from_tensors(
            centroid_m=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
            ),
            volume_m3=torch.ones(4),
            node_type=torch.tensor([0, 0, 1, 1]),
            edge_source=torch.tensor([0, 1, 2, 3, 0, 2]),
            edge_target=torch.tensor([1, 0, 3, 2, 2, 0]),
            edge_kind=torch.tensor([0, 0, 1, 1, 2, 2]),
            edge_area_m2=torch.ones(6),
            edge_area_vector_m2=torch.tensor(
                [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                 [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
            ),
            boundary_fraction=torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        )
        initial = torch.tensor(
            [[[1.0, 0.0, 0.0, 10.0, 300.0],
              [2.0, 0.0, 0.0, 20.0, 310.0],
              [0.0, 0.0, 0.0, 0.0, 350.0],
              [0.0, 0.0, 0.0, 0.0, 360.0]]]
        )
        condition = torch.zeros((1, 8))
        time = torch.tensor([0.0, 0.5, 1.0])
        for mode in ("repeated_query_spatial", "factorized_static_spatial"):
            with self.subTest(mode=mode):
                model = HCCBP418SpatiotemporalRegionalOperator(
                    hidden_dim=8,
                    local_pre_iterations=1,
                    physics_attention_blocks=1,
                    local_post_iterations=1,
                    physics_attention_heads=2,
                    physics_slices=4,
                    temporal_layers=1,
                    temporal_heads=1,
                    temporal_node_chunk_size=2,
                    spatial_temporal_mode=mode,
                    boundary_role_count=3,
                )
                output = model(initial, condition, time, graph)
                self.assertEqual(tuple(output.shape), (1, 3, 4, 5))
                torch.testing.assert_close(output[:, 0], initial)
                torch.testing.assert_close(
                    output[..., :4], initial[:, None, :, :4].expand(-1, 3, -1, -1)
                )
                self.assertTrue(torch.isfinite(output).all())
                output[..., 4].sum().backward()
                trainable = [
                    parameter for parameter in model.parameters() if parameter.requires_grad
                ]
                self.assertTrue(all(parameter.grad is not None for parameter in trainable))
                self.assertTrue(
                    all(torch.isfinite(parameter.grad).all() for parameter in trainable)
                )

    def test_boundary_roles_are_part_of_structural_features(self) -> None:
        import torch

        from hccb_p418_spatiotemporal_regional_operator import (
            P418ThermalStepRegionalGraph,
        )

        graph = P418ThermalStepRegionalGraph.from_tensors(
            centroid_m=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            volume_m3=torch.ones(2),
            node_type=torch.tensor([0, 1]),
            edge_source=torch.tensor([0, 1]),
            edge_target=torch.tensor([1, 0]),
            edge_kind=torch.tensor([2, 2]),
            edge_area_m2=torch.ones(2),
            edge_area_vector_m2=torch.ones((2, 3)),
            boundary_fraction=torch.tensor([[1.0, 0.0], [0.0, 0.75]]),
        )
        self.assertEqual(graph.boundary_role_count, 2)
        self.assertEqual(graph.structural_feature_dim, 8)
        torch.testing.assert_close(
            graph.structural_features()[:, -2:],
            graph.boundary_fraction,
        )

    def test_literature_bounded_temperature_output_is_exact_at_initial_time(self) -> None:
        import torch

        from hccb_p418_spatiotemporal_regional_operator import (
            HCCBP418SpatiotemporalRegionalOperator,
            P418ThermalStepRegionalGraph,
        )

        graph = P418ThermalStepRegionalGraph.from_tensors(
            centroid_m=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            volume_m3=torch.ones(2),
            node_type=torch.tensor([0, 1]),
            edge_source=torch.tensor([0, 1]),
            edge_target=torch.tensor([1, 0]),
            edge_kind=torch.tensor([2, 2]),
            edge_area_m2=torch.ones(2),
            edge_area_vector_m2=torch.tensor(
                [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
            ),
        )
        mean = torch.tensor([600.0, 700.0])
        std = torch.tensor([100.0, 200.0])
        bounds = torch.tensor([[300.0, 1000.0], [298.0, 1300.0]])
        initial = torch.zeros((1, 2, 5))
        initial[..., 3] = torch.tensor([120000.0, 0.0])
        initial[..., 4] = (torch.tensor([500.0, 650.0]) - mean) / std
        model = HCCBP418SpatiotemporalRegionalOperator(
            hidden_dim=8,
            local_pre_iterations=1,
            physics_attention_blocks=1,
            local_post_iterations=1,
            physics_attention_heads=2,
            physics_slices=4,
            temporal_layers=1,
            temporal_heads=1,
            temporal_node_chunk_size=2,
            temperature_output_mode="literature_bounded_logit",
            temperature_mean_k_by_node_type=mean,
            temperature_std_k_by_node_type=std,
            temperature_bounds_k_by_node_type=bounds,
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.temperature_change.layers[-1].bias.fill_(-20.0)
        time = torch.tensor([0.0, 0.5, 1.0])
        output = model(initial, torch.zeros((1, 8)), time, graph)
        physical_temperature = (
            output[..., 4] * std[graph.node_type] + mean[graph.node_type]
        )
        torch.testing.assert_close(output[:, 0], initial)
        self.assertTrue(
            torch.all(physical_temperature[:, 1:] >= bounds[graph.node_type, 0])
        )
        self.assertTrue(
            torch.all(physical_temperature[:, 1:] <= bounds[graph.node_type, 1])
        )
        torch.testing.assert_close(
            output[..., :4], initial[:, None, :, :4].expand(-1, 3, -1, -1)
        )

    def test_literature_bounded_residual_can_heat_from_lower_bound(self) -> None:
        import torch

        from hccb_p418_spatiotemporal_regional_operator import (
            HCCBP418SpatiotemporalRegionalOperator,
            P418ThermalStepRegionalGraph,
        )

        graph = P418ThermalStepRegionalGraph.from_tensors(
            centroid_m=torch.tensor([[0.0, 0.0, 0.0]]),
            volume_m3=torch.ones(1),
            node_type=torch.tensor([0]),
            edge_source=torch.tensor([0]),
            edge_target=torch.tensor([0]),
            edge_kind=torch.tensor([0]),
            edge_area_m2=torch.ones(1),
            edge_area_vector_m2=torch.tensor([[1.0, 0.0, 0.0]]),
        )
        mean = torch.tensor([600.0, 700.0])
        std = torch.tensor([100.0, 200.0])
        bounds = torch.tensor([[300.0, 1000.0], [298.0, 1300.0]])
        initial = torch.zeros((1, 1, 5))
        initial[..., 4] = (300.0 - mean[0]) / std[0]
        model = HCCBP418SpatiotemporalRegionalOperator(
            hidden_dim=8,
            local_pre_iterations=1,
            physics_attention_blocks=1,
            local_post_iterations=1,
            physics_attention_heads=2,
            physics_slices=4,
            temporal_layers=1,
            temporal_heads=1,
            temporal_node_chunk_size=1,
            temperature_output_mode="literature_bounded_residual",
            temperature_mean_k_by_node_type=mean,
            temperature_std_k_by_node_type=std,
            temperature_bounds_k_by_node_type=bounds,
        )
        for parameter in model.parameters():
            parameter.data.zero_()
        model.temperature_change.layers[-1].bias.data.fill_(1.0)
        output = model(
            initial,
            torch.zeros((1, 8)),
            torch.tensor([0.0, 1.0]),
            graph,
        )
        physical_temperature = output[..., 4] * std[0] + mean[0]
        torch.testing.assert_close(physical_temperature[:, 0], torch.tensor([[300.0]]))
        predicted_temperature = float(physical_temperature[0, 1, 0].detach())
        self.assertGreater(predicted_temperature, 390.0)
        self.assertLess(predicted_temperature, 1000.0)
        physical_temperature[0, 1, 0].backward()
        self.assertGreater(
            float(model.temperature_change.layers[-1].bias.grad),
            10.0,
        )

    def test_factorized_mode_encodes_fixed_spatial_context_once(self) -> None:
        import torch

        from hccb_p418_spatiotemporal_regional_operator import (
            HCCBP418SpatiotemporalRegionalOperator,
            P418ThermalStepRegionalGraph,
        )

        graph = P418ThermalStepRegionalGraph.from_tensors(
            centroid_m=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            volume_m3=torch.ones(2),
            node_type=torch.tensor([0, 1]),
            edge_source=torch.tensor([0, 1]),
            edge_target=torch.tensor([1, 0]),
            edge_kind=torch.tensor([2, 2]),
            edge_area_m2=torch.ones(2),
            edge_area_vector_m2=torch.tensor(
                [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
            ),
        )
        initial = torch.tensor(
            [[[1.0, 0.0, 0.0, 10.0, 300.0], [0.0, 0.0, 0.0, 0.0, 350.0]]]
        )
        condition = torch.zeros((1, 8))
        times = torch.linspace(0.0, 1.0, 5)

        calls = {}
        for mode in ("repeated_query_spatial", "factorized_static_spatial"):
            model = HCCBP418SpatiotemporalRegionalOperator(
                hidden_dim=8,
                local_pre_iterations=1,
                physics_attention_blocks=1,
                local_post_iterations=1,
                physics_attention_heads=2,
                physics_slices=4,
                temporal_layers=1,
                temporal_heads=1,
                spatial_time_chunk_size=1,
                temporal_node_chunk_size=2,
                spatial_temporal_mode=mode,
            ).eval()
            counter = {"value": 0}

            def count_call(_module, _inputs, _output):
                counter["value"] += 1

            hook = model.local_pre[0].register_forward_hook(count_call)
            with torch.no_grad():
                model(initial, condition, times, graph)
            hook.remove()
            calls[mode] = counter["value"]

        self.assertEqual(calls["repeated_query_spatial"], len(times))
        self.assertEqual(calls["factorized_static_spatial"], 1)


if __name__ == "__main__":
    unittest.main()
