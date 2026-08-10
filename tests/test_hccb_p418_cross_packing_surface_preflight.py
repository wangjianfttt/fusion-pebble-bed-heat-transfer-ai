#!/usr/bin/env python3
"""Regression checks for seed202/303 clipped particle surfaces."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/hccb_p418_cross_packing_surface_preflight"
PLAN = json.loads(
    (ROOT / "parameters/hccb_p418_cross_packing_plan.json").read_text(
        encoding="utf-8"
    )
)


class HCCBP418CrossPackingSurfacePreflightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expected_hash = {
            item["seed"]: item["packing_npz_sha256"]
            for item in PLAN["packing_realisations"]
        }

    def test_both_future_packings_have_closed_surfaces(self) -> None:
        for seed in (202, 303):
            with self.subTest(seed=seed):
                summary = json.loads(
                    (RESULTS / f"seed{seed}/solid_surface_summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertTrue(all(summary["checks"].values()))
                self.assertEqual(summary["closed_surface_edge_count"], 0)
                self.assertGreater(summary["output_triangles"], 100_000)

    def test_global_tolerance_fallback_is_recorded(self) -> None:
        expected = {202: 2.0e-6, 303: 1.0e-6}
        for seed, tolerance in expected.items():
            with self.subTest(seed=seed):
                summary = json.loads(
                    (RESULTS / f"seed{seed}/solid_surface_summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                manifest = json.loads(
                    (RESULTS / f"seed{seed}/case_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(summary["plane_snap_tolerance_m"], tolerance)
                self.assertEqual(
                    manifest["numerical_controls"][
                        "surface_plane_snap_tolerance_m"
                    ],
                    tolerance,
                )

    def test_surfaces_keep_the_registered_packings_and_crop(self) -> None:
        for seed in (202, 303):
            with self.subTest(seed=seed):
                manifest = json.loads(
                    (RESULTS / f"seed{seed}/case_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(
                    manifest["source_packing_sha256"], self.expected_hash[seed]
                )
                self.assertEqual(
                    manifest["crop_box_dp"],
                    [1.234, 5.157, 3.921, 8.163, 2.906, 6.396],
                )
                self.assertEqual(manifest["new_physical_parameters"], [])


if __name__ == "__main__":
    unittest.main()
