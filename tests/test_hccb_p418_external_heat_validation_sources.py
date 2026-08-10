#!/usr/bin/env python3

from __future__ import annotations

import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "parameters/hccb_p418_external_heat_validation_sources.csv"


class P418ExternalHeatValidationSourcesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with SOURCES.open(newline="", encoding="utf-8-sig") as stream:
            cls.rows = list(csv.DictReader(stream))

    def test_each_source_has_a_concrete_use_and_limit(self) -> None:
        self.assertGreaterEqual(len(self.rows), 7)
        for row in self.rows:
            self.assertTrue(row["current_use"].strip(), row["source_id"])
            self.assertTrue(row["not_used_as"].strip(), row["source_id"])
            self.assertTrue(row["limitation"].strip(), row["source_id"])

    def test_literature_curves_are_not_declared_as_p418_training(self) -> None:
        forbidden = {"training", "calibration_training", "p418_training"}
        for row in self.rows:
            self.assertNotIn(row["training_status"], forbidden, row["source_id"])
            if row["numerical_data_available"].lower().startswith("no"):
                self.assertEqual(row["training_status"], "not_admitted", row["source_id"])

    def test_closest_mockups_keep_their_physical_difference(self) -> None:
        rows = {row["source_id"]: row for row in self.rows}
        self.assertIn("plate heaters", rows["EHV005"]["internal_heating"])
        self.assertIn("separate", rows["EHV005"]["through_flow"])
        self.assertIn("stagnant", rows["EHV007"]["fluid"])

    def test_condition_matrix_and_geometry_sources_are_separate(self) -> None:
        rows = {row["source_id"]: row for row in self.rows}
        self.assertIn("60-condition", rows["EHV001"]["current_use"])
        self.assertIn("geometry", rows["EHV008"]["current_use"])
        self.assertNotEqual(rows["EHV001"]["doi_or_url"], rows["EHV008"]["doi_or_url"])


if __name__ == "__main__":
    unittest.main()
