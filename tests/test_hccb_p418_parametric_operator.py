#!/usr/bin/env python3

from __future__ import annotations

import unittest
import sys
from pathlib import Path

try:
    import torch
except ImportError:  # local document runtime does not ship the remote GPU stack
    torch = None


@unittest.skipIf(torch is None, "PyTorch is tested on the remote GPU machine")
class P418ParametricOperatorTest(unittest.TestCase):
    def test_chunked_forward_and_gradient(self) -> None:
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "code"))
        from hccb_p418_parametric_regional_operator import (
            HCCBP418ParametricRegionalOperator,
            P418RegionalLevel,
            P418RegionalMesh,
            allocate_processor_steps,
            collapse_mesh_to_level,
            output_mask,
        )

        self.assertEqual(allocate_processor_steps(12, 6), (2, 2, 2, 2, 2, 2))
        self.assertEqual(allocate_processor_steps(13, 6), (3, 2, 2, 2, 2, 2))

        level0 = P418RegionalLevel(
            centroid_m=torch.tensor([[0.5, 0., 0.], [0.5, 1., 1.]]),
            volume_m3=torch.tensor([2., 2.]),
            node_type=torch.tensor([0, 1]),
            boundary_fraction=torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
            parent_from_finer=torch.tensor([0, 0, 1, 1]),
            edge_source=torch.tensor([0, 1]),
            edge_target=torch.tensor([1, 0]),
            edge_kind=torch.tensor([2, 2]),
            edge_area_m2=torch.ones(2),
            edge_area_vector_m2=torch.tensor([[0., 1., 0.], [0., -1., 0.]]),
            edge_face_centroid_m=torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
        )
        level1 = P418RegionalLevel(
            centroid_m=level0.centroid_m.clone(),
            volume_m3=level0.volume_m3.clone(),
            node_type=level0.node_type.clone(),
            boundary_fraction=level0.boundary_fraction.clone(),
            parent_from_finer=torch.tensor([0, 1]),
            edge_source=level0.edge_source.clone(),
            edge_target=level0.edge_target.clone(),
            edge_kind=level0.edge_kind.clone(),
            edge_area_m2=level0.edge_area_m2.clone(),
            edge_area_vector_m2=level0.edge_area_vector_m2.clone(),
            edge_face_centroid_m=level0.edge_face_centroid_m.clone(),
        )
        mesh = P418RegionalMesh(
            fine_centroid_m=torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 1.], [1., 1., 1.]]),
            fine_volume_m3=torch.ones(4),
            fine_node_type=torch.tensor([0, 0, 1, 1]),
            fine_boundary_role=torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]]),
            coordinate_center_m=torch.tensor([0.5, 0.5, 0.5]),
            coordinate_scale_m=torch.ones(3),
            volume_scale_m3=torch.tensor(1.),
            levels=(level0, level1),
        )
        model = HCCBP418ParametricRegionalOperator(
            boundary_role_count=2,
            hidden_dim=16,
            processor_steps=1,
            active_levels=2,
        )
        self.assertEqual(sum(len(level) for level in model.processors), 1)
        condition = torch.zeros((2, 5), requires_grad=True)
        prediction = model(condition, mesh, chunk_size=2)
        self.assertEqual(prediction.shape, (2, 4, 5))
        regional_latent = model.encode_regions(condition, mesh)
        regional_prediction = model.decode_active_regions(
            condition, regional_latent, mesh
        )
        self.assertEqual(regional_prediction.shape, (2, 2, 5))
        self.assertEqual(output_mask(mesh.fine_node_type).sum().item(), 12)
        prediction.square().mean().backward()
        self.assertIsNotNone(condition.grad)
        self.assertTrue(torch.isfinite(condition.grad).all())

        attention_model = HCCBP418ParametricRegionalOperator(
            boundary_role_count=2,
            hidden_dim=16,
            processor_steps=2,
            active_levels=2,
            processor_kind="hybrid_attention",
            attention_heads=4,
            attention_start_level=1,
        )
        attention_prediction = attention_model(condition.detach(), mesh, chunk_size=3)
        self.assertEqual(attention_prediction.shape, (2, 4, 5))
        self.assertTrue(torch.isfinite(attention_prediction).all())

        physics_attention_model = HCCBP418ParametricRegionalOperator(
            boundary_role_count=2,
            hidden_dim=16,
            processor_steps=2,
            active_levels=2,
            processor_kind="hybrid_physics_attention",
            attention_heads=4,
            attention_start_level=1,
            physics_slices=4,
        )
        physics_prediction = physics_attention_model(
            condition.detach(), mesh, chunk_size=2
        )
        self.assertEqual(physics_prediction.shape, (2, 4, 5))
        self.assertTrue(torch.isfinite(physics_prediction).all())

        coarse_model = HCCBP418ParametricRegionalOperator(
            boundary_role_count=2,
            hidden_dim=16,
            processor_steps=2,
            active_levels=1,
            start_level=1,
        )
        coarse_prediction = coarse_model(condition.detach(), mesh, chunk_size=2)
        self.assertEqual(coarse_prediction.shape, (2, 4, 5))
        self.assertTrue(torch.isfinite(coarse_prediction).all())

        collapsed = collapse_mesh_to_level(mesh, 1)
        self.assertEqual(len(collapsed.levels), 1)
        self.assertTrue(
            torch.equal(
                collapsed.levels[0].parent_from_finer,
                torch.tensor([0, 0, 1, 1]),
            )
        )
        collapsed_model = HCCBP418ParametricRegionalOperator(
            boundary_role_count=2,
            hidden_dim=16,
            processor_steps=2,
            active_levels=1,
        )
        collapsed_prediction = collapsed_model(
            condition.detach(), collapsed, chunk_size=2
        )
        self.assertEqual(collapsed_prediction.shape, (2, 4, 5))


if __name__ == "__main__":
    unittest.main()
