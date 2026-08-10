#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_p418_model_splits import make_splits, records, validate  # noqa: E402


class P418ModelSplitsTest(unittest.TestCase):
    def test_all_splits_are_disjoint_and_complete(self) -> None:
        items = records()
        splits = make_splits(items)
        validate(splits, {str(item["condition_id"]) for item in items})
        self.assertEqual(len(splits["interleaved_all_ranges"]["train"]), 36)
        self.assertEqual(len(splits["interleaved_all_ranges"]["validation"]), 12)
        self.assertEqual(len(splits["interleaved_all_ranges"]["test"]), 12)

    def test_extrapolation_tests_use_published_extremes(self) -> None:
        items = records()
        by_id = {str(item["condition_id"]): item for item in items}
        splits = make_splits(items)
        temperatures = {
            by_id[item]["inlet_temperature_K"]
            for item in splits["temperature_extrapolation"]["test"]
        }
        velocities = {
            by_id[item]["inlet_velocity_m_s"]
            for item in splits["velocity_extrapolation"]["test"]
        }
        self.assertEqual(temperatures, {900.0})
        self.assertEqual(velocities, {0.25})


if __name__ == "__main__":
    unittest.main()
