#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

from build_hccb_p418_learning_curve_splits import (  # noqa: E402
    AXIS_KEYS,
    TRAINING_SIZES,
    build_payload,
)


class P418LearningCurveSplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = json.loads(
            (ROOT / "parameters" / "hccb_p418_model_splits.json").read_text(
                encoding="utf-8"
            )
        )

    def test_subsets_are_nested_and_keep_validation_and_test_fixed(self) -> None:
        payload = build_payload(self.base)
        base_split = self.base["splits"]["interleaved_all_ranges"]
        previous: set[str] = set()
        for size in TRAINING_SIZES:
            split = payload["splits"][f"learning_curve_n{size:02d}"]
            train = set(split["train"])
            self.assertEqual(len(split["train"]), size)
            self.assertTrue(previous.issubset(train))
            self.assertEqual(split["validation"], base_split["validation"])
            self.assertEqual(split["test"], base_split["test"])
            self.assertEqual(len(split["unused"]), 36 - size)
            self.assertFalse(train.intersection(split["unused"]))
            previous = train

    def test_smallest_subset_covers_every_physical_level(self) -> None:
        payload = build_payload(self.base)
        by_id = {
            item["condition_id"]: item for item in payload["conditions"]
        }
        base_train = self.base["splits"]["interleaved_all_ranges"]["train"]
        selected = payload["splits"]["learning_curve_n09"]["train"]
        for key in AXIS_KEYS:
            expected = {by_id[item][key] for item in base_train}
            observed = {by_id[item][key] for item in selected}
            self.assertEqual(observed, expected)

    def test_selection_is_deterministic(self) -> None:
        first = build_payload(self.base)
        second = build_payload(self.base)
        self.assertEqual(first["splits"], second["splits"])


if __name__ == "__main__":
    unittest.main()
