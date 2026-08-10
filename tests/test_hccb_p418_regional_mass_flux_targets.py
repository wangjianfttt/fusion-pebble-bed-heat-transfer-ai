#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_p418_regional_mass_flux_targets import (  # noqa: E402
    regional_balance,
    regional_internal_pairs,
)


class P418RegionalMassFluxTargetTest(unittest.TestCase):
    def test_internal_face_orientation_and_balance(self) -> None:
        owner, neighbour, inverse, crossing, orientation = regional_internal_pairs(
            fine_owner=np.array([0, 1, 2]),
            fine_neighbour=np.array([1, 2, 3]),
            parent=np.array([2, 2, 4, 3]),
            global_to_local=np.array([-1, -1, 0, 1, 2]),
        )
        self.assertTrue(np.array_equal(crossing, [False, True, True]))
        self.assertTrue(np.array_equal(owner, [0, 1]))
        self.assertTrue(np.array_equal(neighbour, [2, 2]))
        self.assertTrue(np.array_equal(orientation, [1.0, -1.0]))
        flux = np.bincount(inverse, weights=np.array([5.0, 7.0]) * orientation)
        balance = regional_balance(
            internal_flux=flux,
            internal_owner=owner,
            internal_neighbour=neighbour,
            boundary_flux=np.array([-5.0, 7.0, -2.0]),
            boundary_owner=np.array([0, 1, 2]),
            cell_count=3,
        )
        self.assertTrue(np.allclose(balance, 0.0))


if __name__ == "__main__":
    unittest.main()
