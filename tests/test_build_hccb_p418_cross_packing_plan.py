#!/usr/bin/env python3
"""Tests for the exact-P418 cross-packing screening plan."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_p418_cross_packing_plan import build_plan  # noqa: E402


class HCCBP418CrossPackingPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = build_plan(ROOT)
        cls.conditions = cls.plan["screening_design"]["conditions"]
        cls.identifiers = [item["condition_id"] for item in cls.conditions]

    def test_screening_uses_eight_corners_and_one_published_interior(self) -> None:
        expected = {
            "u0p05_T300_q4p85",
            "u0p05_T300_q8p85",
            "u0p05_T900_q4p85",
            "u0p05_T900_q8p85",
            "u0p25_T300_q4p85",
            "u0p25_T300_q8p85",
            "u0p25_T900_q4p85",
            "u0p25_T900_q8p85",
            "u0p15_T700_q6p85",
        }
        self.assertEqual(set(self.identifiers), expected)
        self.assertEqual(len(self.identifiers), 9)

    def test_no_physical_value_is_added_outside_p418(self) -> None:
        self.assertEqual(self.plan["new_physical_parameter_values_added"], [])
        self.assertEqual(
            {item["inlet_velocity_m_s"] for item in self.conditions},
            {0.05, 0.15, 0.25},
        )
        self.assertEqual(
            {item["inlet_temperature_K"] for item in self.conditions},
            {300.0, 700.0, 900.0},
        )
        self.assertEqual(
            {item["solid_heat_source_MW_m3"] for item in self.conditions},
            {4.85, 6.85, 8.85},
        )

    def test_seed303_is_reserved_for_zero_shot_prediction(self) -> None:
        roles = {
            item["seed"]: item["role"] for item in self.plan["packing_realisations"]
        }
        self.assertEqual(roles[303], "nine_condition_final_zero_shot_packing")
        text = self.plan["model_use"]["seed303_zero_shot"]
        self.assertIn("before opening any seed303 field", text)

    def test_optional_few_shot_never_replaces_zero_shot(self) -> None:
        few_shot = self.plan["model_use"]["seed303_optional_few_shot_after_zero_shot"]
        self.assertEqual(len(few_shot["adaptation_conditions"]), 3)
        self.assertEqual(len(few_shot["test_conditions"]), 6)
        self.assertFalse(
            set(few_shot["adaptation_conditions"]).intersection(
                few_shot["test_conditions"]
            )
        )

    def test_all_packing_files_and_sources_are_traceable(self) -> None:
        self.assertEqual(
            [item["seed"] for item in self.plan["packing_realisations"]],
            [101, 202, 303],
        )
        self.assertIn("P418", self.plan["physical_parameter_ids"])
        self.assertIn("P050", self.plan["physical_parameter_ids"])
        self.assertIn("P390", self.plan["physical_parameter_ids"])
        for item in self.plan["packing_realisations"]:
            self.assertTrue((ROOT / item["packing_path"]).is_file())
            self.assertEqual(len(item["packing_npz_sha256"]), 64)

    def test_full_three_packing_study_is_optional_not_current_prerequisite(self) -> None:
        extension = self.plan["later_full_extension"]
        self.assertEqual(extension["case_count"], 180)
        self.assertEqual(
            extension["protocol"], "three_fold_leave_one_complete_packing_out"
        )
        self.assertIn("optional extension", extension["statement"])
        self.assertIn("not required for the current 60+9+9 paper", extension["statement"])


if __name__ == "__main__":
    unittest.main()
