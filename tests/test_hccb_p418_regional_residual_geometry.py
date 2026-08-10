#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_p418_regional_residual_geometry import (  # noqa: E402
    aggregate_boundary_faces,
)


class P418RegionalResidualGeometryTest(unittest.TestCase):
    def test_boundary_faces_are_grouped_by_parent_and_patch(self) -> None:
        grouped = aggregate_boundary_faces(
            owner=np.array([0, 1, 2]),
            patch=np.array([0, 0, 1]),
            centroid=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
            area_vector=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            area=np.array([1.0, 2.0, 1.0]),
            parent=np.array([3, 3, 4]),
            patch_count=3,
            excluded_patch=2,
        )
        self.assertTrue(np.array_equal(grouped["owner_global"], [3, 4]))
        self.assertTrue(np.array_equal(grouped["patch"], [0, 1]))
        self.assertTrue(np.allclose(grouped["area_m2"], [3.0, 1.0]))
        self.assertTrue(np.allclose(grouped["centroid_m"][0], [4.0 / 3.0, 0.0, 0.0]))
        self.assertTrue(np.array_equal(grouped["fine_face_count"], [2, 1]))


if __name__ == "__main__":
    unittest.main()
