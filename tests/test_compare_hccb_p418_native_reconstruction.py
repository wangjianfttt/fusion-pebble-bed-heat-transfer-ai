#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from compare_hccb_p418_native_reconstruction import (  # noqa: E402
    phase_reconstruction,
    reconstruction_operators,
    unique_neighbour_lists,
)


class NativeReconstructionTest(unittest.TestCase):
    def test_affine_field_is_reconstructed_exactly_and_preserves_region_mean(self) -> None:
        regional_centroid = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        owner = np.asarray([0, 0, 0, 1, 1, 2])
        neighbour = np.asarray([1, 2, 3, 2, 3, 3])
        neighbours = unique_neighbour_lists(4, owner, neighbour)
        operators, ranks = reconstruction_operators(regional_centroid, neighbours)
        self.assertEqual(int(ranks[0]), 3)

        parent = np.repeat(np.arange(4), 2)
        offset = np.asarray([[0.1, -0.1, 0.05], [-0.1, 0.1, -0.05]])
        native_centroid = np.vstack(
            [regional_centroid[node] + offset for node in range(4)]
        )
        volume = np.ones(len(parent))
        gradient = np.asarray([2.0, -3.0, 4.0])
        regional_temperature = 300.0 + regional_centroid @ gradient
        expected = 300.0 + native_centroid @ gradient
        constant, affine, limited, _, alpha = phase_reconstruction(
            regional_temperature,
            regional_centroid,
            native_centroid,
            volume,
            parent,
            neighbours,
            operators,
        )
        self.assertGreater(float(np.max(np.abs(constant - expected))), 0.0)
        np.testing.assert_allclose(affine, expected, atol=1.0e-12)
        self.assertTrue(np.all((alpha >= 0.0) & (alpha <= 1.0)))
        for node in range(4):
            np.testing.assert_allclose(
                np.mean(affine[parent == node]), regional_temperature[node], atol=1.0e-12
            )
            np.testing.assert_allclose(
                np.mean(limited[parent == node]), regional_temperature[node], atol=1.0e-12
            )
            local_bound = regional_temperature[neighbours[node]]
            lower = min(regional_temperature[node], float(np.min(local_bound)))
            upper = max(regional_temperature[node], float(np.max(local_bound)))
            self.assertGreaterEqual(float(np.min(limited[parent == node])), lower)
            self.assertLessEqual(float(np.max(limited[parent == node])), upper)


if __name__ == "__main__":
    unittest.main()
