#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


@unittest.skipIf(torch is None, "PyTorch is tested on the remote compute machine")
class P418SteadyMicrobatchTest(unittest.TestCase):
    def test_accumulated_gradient_matches_full_effective_batch(self) -> None:
        from train_hccb_p418_conservative_mixed_operator import weighted_microbatches

        x = torch.tensor([[1.0], [2.0], [4.0], [7.0]])
        y = torch.tensor([[0.5], [1.5], [2.0], [3.5]])
        full_weight = torch.tensor(0.3, requires_grad=True)
        ((x * full_weight - y).square().mean()).backward()

        accumulated_weight = torch.tensor(0.3, requires_grad=True)
        indices = np.arange(len(x))
        parts = weighted_microbatches(indices, microbatch_size=1)
        self.assertAlmostEqual(sum(weight for _, weight in parts), 1.0)
        for part, weight in parts:
            loss = (x[part] * accumulated_weight - y[part]).square().mean()
            (loss * weight).backward()

        torch.testing.assert_close(accumulated_weight.grad, full_weight.grad)


if __name__ == "__main__":
    unittest.main()
