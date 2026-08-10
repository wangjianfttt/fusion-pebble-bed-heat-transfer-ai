#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from train_hccb_p418_regional_response_surface import (  # noqa: E402
    design_matrix,
    oriented_balance,
    state_metrics,
)


class P418RegionalResponseSurfaceTest(unittest.TestCase):
    def test_quadratic_basis_contains_no_physical_fit_parameter(self) -> None:
        condition = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0]])
        design = design_matrix(condition, 2)
        self.assertEqual(design.shape, (1, 10))
        np.testing.assert_allclose(
            design[0], [1.0, 1.0, 2.0, 3.0, 1.0, 4.0, 9.0, 2.0, 3.0, 6.0]
        )

    def test_volume_weighted_state_metric_is_zero_for_exact_prediction(self) -> None:
        target = np.arange(30, dtype=float).reshape(2, 3, 5)
        node_type = np.asarray([0, 0, 1])
        volume = np.asarray([1.0, 2.0, 4.0])
        metrics = state_metrics(target.copy(), target, node_type, volume)
        self.assertEqual(metrics["state_normalized_rmse"], 0.0)
        self.assertEqual(metrics["state_channel_rmse"], [0.0] * 6)

    def test_oriented_mass_and_energy_balance(self) -> None:
        internal = np.asarray([[2.0, 3.0]])
        boundary = np.asarray([[-2.0, 3.0]])
        balance = oriented_balance(
            internal,
            boundary,
            np.asarray([0, 1]),
            np.asarray([1, 2]),
            np.asarray([0, 2]),
            3,
            source=np.asarray([[0.0, 1.0, 0.0]]),
        )
        np.testing.assert_allclose(balance, np.zeros((1, 3)))


if __name__ == "__main__":
    unittest.main()
