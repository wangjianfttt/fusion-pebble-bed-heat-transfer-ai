#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is tested on the remote machine")
class P418TemporalTemperatureDiffusionTest(unittest.TestCase):
    def test_model_gradient_and_exact_observation_conditioning(self) -> None:
        import torch

        from hccb_p418_temporal_temperature_diffusion import (
            P418TemporalTemperatureResidualRefiner,
            sample_temporal_temperature_residual,
        )

        batch, times, nodes = 2, 4, 7
        baseline = torch.randn(batch, times, nodes, 1)
        noised = torch.randn_like(baseline)
        condition = torch.randn(batch, 8)
        structure = torch.randn(nodes, 6)
        time = torch.linspace(0.0, 1.0, times)
        observed = baseline.clone()
        observed[:, 0, 1, 0] += 2.0
        observed[:, 2, 3, 0] += 0.5
        mask = torch.zeros_like(baseline, dtype=torch.bool)
        mask[:, 0, 1, 0] = True
        mask[:, 2, 3, 0] = True
        model = P418TemporalTemperatureResidualRefiner(
            structural_dim=6,
            hidden_dim=16,
            spatial_layers=2,
            spatial_attention_heads=4,
            physics_slices=4,
            temporal_layers=1,
            temporal_heads=1,
            spatial_time_chunk_size=1,
            temporal_node_chunk_size=3,
        )
        step = torch.tensor([0, 2], dtype=torch.long)
        velocity = model(
            baseline,
            noised,
            condition,
            structure,
            time,
            observed - baseline,
            mask,
            step,
        )
        self.assertEqual(velocity.shape, baseline.shape)
        velocity.square().mean().backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        self.assertTrue(all(value is not None for value in gradients))
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))

        model.eval()
        residual = sample_temporal_temperature_residual(
            model,
            baseline,
            condition,
            structure,
            time,
            observed - baseline,
            mask,
            initial_noise=torch.ones_like(baseline),
        )
        torch.testing.assert_close(residual[:, 0], torch.zeros_like(residual[:, 0]))
        dynamic_mask = mask.clone()
        dynamic_mask[:, 0] = False
        torch.testing.assert_close(
            residual[dynamic_mask], (observed - baseline)[dynamic_mask]
        )
        self.assertTrue(torch.isfinite(residual).all())


if __name__ == "__main__":
    unittest.main()
