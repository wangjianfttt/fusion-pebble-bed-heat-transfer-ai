#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


@unittest.skipIf(torch is None, "PyTorch is tested on the remote compute machine")
class P418CoordinatePINNTest(unittest.TestCase):
    def test_source_architecture_forward_and_gradient(self) -> None:
        from hccb_p418_coordinate_pinn import HCCBP418CoordinatePINNOperator
        from hccb_p418_parametric_regional_operator import (
            P418RegionalLevel,
            P418RegionalMesh,
        )

        level = P418RegionalLevel(
            centroid_m=torch.tensor([[0.25, 0.0, 0.0], [0.75, 0.0, 0.0]]),
            volume_m3=torch.ones(2),
            node_type=torch.tensor([0, 1]),
            boundary_fraction=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            parent_from_finer=torch.tensor([0, 1]),
            edge_source=torch.tensor([0, 1]),
            edge_target=torch.tensor([1, 0]),
            edge_kind=torch.tensor([2, 2]),
            edge_area_m2=torch.ones(2),
            edge_area_vector_m2=torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            edge_face_centroid_m=torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        )
        mesh = P418RegionalMesh(
            fine_centroid_m=level.centroid_m.clone(),
            fine_volume_m3=level.volume_m3.clone(),
            fine_node_type=level.node_type.clone(),
            fine_boundary_role=level.boundary_fraction.clone(),
            coordinate_center_m=torch.tensor([0.5, 0.0, 0.0]),
            coordinate_scale_m=torch.ones(3),
            volume_scale_m3=torch.tensor(1.0),
            levels=(level,),
        )
        model = HCCBP418CoordinatePINNOperator(boundary_role_count=2)
        self.assertEqual(len(model.hidden), 6)
        self.assertTrue(all(layer.out_features == 50 for layer in model.hidden))
        condition = torch.zeros((3, 5), requires_grad=True)
        regional = model.encode_regions(condition, mesh)
        prediction = model.decode_active_regions(condition, regional, mesh)
        fine_prediction = model(condition, mesh, chunk_size=1)
        self.assertEqual(tuple(regional.shape), (3, 2, 50))
        self.assertEqual(tuple(prediction.shape), (3, 2, 5))
        self.assertEqual(tuple(fine_prediction.shape), (3, 2, 5))
        fine_prediction.square().mean().backward()
        self.assertTrue(torch.isfinite(condition.grad).all())

    def test_build_model_uses_archived_settings(self) -> None:
        from train_hccb_p418_regional_operator import build_model

        model, settings = build_model("pinn", 2)
        self.assertEqual(model.hidden_dim, 50)
        self.assertEqual(model.hidden_layers, 6)
        self.assertEqual(settings["optimizer"], "Adam")
        self.assertEqual(settings["learning_rate"], 0.01)


if __name__ == "__main__":
    unittest.main()
