#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is tested on the remote compute machine")
class P418RegionalDiffusionRefinerTest(unittest.TestCase):
    def test_official_schedule_training_pair_and_gradient(self) -> None:
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "code"))
        from hccb_p418_regional_diffusion_refiner import (
            P418RegionalResidualRefiner,
            make_velocity_training_pair,
            pde_refiner_schedule,
            sample_residual,
        )

        betas, alpha = pde_refiner_schedule()
        self.assertEqual(len(betas), 4)
        self.assertAlmostEqual(float(betas[0]), 4.0e-7, places=12)
        self.assertAlmostEqual(float(betas[-1]), 1.0, places=7)
        self.assertTrue(torch.all(alpha[1:] <= alpha[:-1]))

        residual = torch.randn(2, 7, 5)
        step = torch.tensor([0, 2], dtype=torch.long)
        noised, target = make_velocity_training_pair(
            residual, step, noise=torch.ones_like(residual)
        )
        self.assertEqual(noised.shape, residual.shape)
        self.assertTrue(torch.isfinite(noised).all())
        self.assertTrue(torch.isfinite(target).all())

        model = P418RegionalResidualRefiner(
            structural_dim=6,
            hidden_dim=16,
            layers=2,
            attention_heads=4,
            physics_slices=4,
        )
        baseline = torch.randn(2, 7, 5, requires_grad=True)
        condition = torch.randn(2, 5)
        structure = torch.randn(7, 6)
        prediction = model(baseline, noised, condition, structure, step)
        self.assertEqual(prediction.shape, residual.shape)
        prediction.square().mean().backward()
        self.assertIsNotNone(baseline.grad)
        self.assertTrue(torch.isfinite(baseline.grad).all())

        model.eval()
        sampled = sample_residual(
            model,
            baseline.detach(),
            condition,
            structure,
            initial_noise=torch.ones_like(residual),
        )
        self.assertEqual(sampled.shape, residual.shape)
        self.assertTrue(torch.isfinite(sampled).all())


if __name__ == "__main__":
    unittest.main()
