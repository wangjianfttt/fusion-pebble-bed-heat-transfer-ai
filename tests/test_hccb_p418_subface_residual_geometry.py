#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_p418_subface_residual_geometry import (  # noqa: E402
    patch_area_by_id,
    preserved_external_boundary,
    select_unique_cross_region_subfaces,
)


class P418SubfaceResidualGeometryTest(unittest.TestCase):
    def test_cross_region_subfaces_are_unique_and_retain_vectors(self) -> None:
        selected = select_unique_cross_region_subfaces(
            source=np.array([0, 1, 1, 2, 2, 3]),
            target=np.array([1, 0, 2, 1, 3, 2]),
            kind=np.array([0, 0, 0, 0, 2, 2]),
            centroid=np.array([[0.5, 0.0, 0.0]] * 6),
            area_vector=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, -2.0, 0.0],
                    [0.0, 0.0, 3.0],
                    [0.0, 0.0, -3.0],
                ]
            ),
            area=np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]),
            local_face=np.arange(6, dtype=np.int64),
            parent=np.array([0, 0, 1, 2]),
        )
        self.assertTrue(np.array_equal(selected["source_global"], [1, 2]))
        self.assertTrue(np.array_equal(selected["target_global"], [2, 3]))
        self.assertTrue(np.array_equal(selected["kind"], [0, 2]))
        self.assertTrue(np.array_equal(selected["local_face"], [2, 4]))
        self.assertTrue(
            np.allclose(
                selected["area_vector_m2"],
                [[0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            )
        )

    def test_external_faces_are_preserved_individually(self) -> None:
        boundary = preserved_external_boundary(
            owner=np.array([0, 1, 2]),
            patch=np.array([0, 0, 2]),
            centroid=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            area_vector=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            area=np.array([1.0, 2.0, 3.0]),
            parent=np.array([3, 3, 4]),
            local_map=np.array([-1, -1, -1, 0, 1]),
            excluded_patch=2,
        )
        self.assertTrue(np.array_equal(boundary["owner"], [0, 0]))
        self.assertEqual(len(boundary["area_m2"]), 2)
        self.assertTrue(np.allclose(boundary["area_m2"], [1.0, 2.0]))
        self.assertTrue(
            np.allclose(patch_area_by_id(boundary["patch"], boundary["area_m2"], 3), [3.0, 0.0, 0.0])
        )


if __name__ == "__main__":
    unittest.main()
