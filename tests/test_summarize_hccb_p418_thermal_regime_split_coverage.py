#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from summarize_hccb_p418_thermal_regime_split_coverage import (  # noqa: E402
    summarize_split_file,
    write_chinese_summary,
)


class ThermalRegimeSplitCoverageTest(unittest.TestCase):
    def test_roles_are_reported_without_requiring_two_sided_extrapolation(self) -> None:
        physical = {
            "cold_train": {
                "cooling_wall_heat_direction": "wall_to_fluid",
                "solid_temperature_relation": "solid_maximum_at_or_below_wall",
            },
            "hot_train": {
                "cooling_wall_heat_direction": "fluid_to_wall",
                "solid_temperature_relation": "solid_maximum_above_wall",
            },
            "hot_test": {
                "cooling_wall_heat_direction": "fluid_to_wall",
                "solid_temperature_relation": "solid_maximum_above_wall",
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            split_file = Path(temporary) / "splits.json"
            split_file.write_text(
                json.dumps(
                    {
                        "splits": {
                            "temperature_extrapolation": {
                                "train": ["cold_train", "hot_train"],
                                "validation": [],
                                "test": ["hot_test"],
                                "unused": [],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            rows = summarize_split_file(split_file, physical)
        self.assertEqual(len(rows), 2)
        by_role = {row["role"]: row for row in rows}
        self.assertTrue(by_role["train"]["contains_both_nonzero_wall_heat_directions"])
        self.assertFalse(by_role["test"]["contains_both_nonzero_wall_heat_directions"])
        self.assertEqual(by_role["test"]["fluid_to_wall_count"], 1)

    def test_unknown_condition_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            split_file = Path(temporary) / "splits.json"
            split_file.write_text(
                json.dumps({"splits": {"x": {"train": ["missing"]}}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing"):
                summarize_split_file(split_file, {})

    def test_partial_mode_reports_known_and_unfinished_cases(self) -> None:
        physical = {
            "cold_train": {
                "cooling_wall_heat_direction": "wall_to_fluid",
                "solid_temperature_relation": "solid_maximum_at_or_below_wall",
            },
            "hot_train": {
                "cooling_wall_heat_direction": "fluid_to_wall",
                "solid_temperature_relation": "solid_maximum_above_wall",
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            split_file = Path(temporary) / "splits.json"
            split_file.write_text(
                json.dumps(
                    {
                        "splits": {
                            "interleaved": {
                                "train": ["cold_train", "unfinished", "hot_train"],
                                "test": ["unfinished_test"],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            rows = summarize_split_file(
                split_file,
                physical,
                allow_partial_physical_coverage=True,
            )
        by_role = {row["role"]: row for row in rows}
        train = by_role["train"]
        self.assertEqual(train["case_count"], 3)
        self.assertEqual(train["known_case_count"], 2)
        self.assertEqual(train["unknown_case_count"], 1)
        self.assertFalse(train["coverage_complete"])
        self.assertTrue(train["contains_both_nonzero_wall_heat_directions"])
        self.assertEqual(train["unknown_condition_ids"], "unfinished")
        test = by_role["test"]
        self.assertEqual(test["known_case_count"], 0)
        self.assertEqual(test["unknown_case_count"], 1)
        self.assertFalse(test["contains_both_nonzero_wall_heat_directions"])

    def test_chinese_summary_distinguishes_finished_and_total_counts(self) -> None:
        rows = [
            {
                "split": "interleaved",
                "role": "train",
                "known_case_count": 2,
                "case_count": 3,
                "wall_to_fluid_count": 1,
                "fluid_to_wall_count": 1,
                "contains_both_nonzero_wall_heat_directions": True,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "summary.md"
            write_chinese_summary(output, rows, coverage_complete=False)
            text = output.read_text(encoding="utf-8")
        self.assertIn("2/3", text)
        self.assertIn("仍在计算", text)
        self.assertIn("壁面向流体", text)


if __name__ == "__main__":
    unittest.main()
