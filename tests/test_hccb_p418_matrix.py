#!/usr/bin/env python3
"""Regression tests for the literature-defined P418 operating matrix."""

from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_gmsh_cht_smoke_case import (  # noqa: E402
    matrix_condition_id,
    parse_p418_matrix,
    select_p418_condition,
)


class P418MatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (ROOT / "parameters/literature_parameter_manifest.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            cls.rows = {row["parameter_id"]: row for row in csv.DictReader(handle)}
        cls.value = cls.rows["P418"]["value"]

    def test_matrix_has_exactly_sixty_unique_cases(self) -> None:
        velocities, temperatures, sources = parse_p418_matrix(self.value)
        identifiers = {
            matrix_condition_id(velocity, temperature, source)
            for velocity in velocities
            for temperature in temperatures
            for source in sources
        }
        self.assertEqual(velocities, [0.05, 0.10, 0.15, 0.20, 0.25])
        self.assertEqual(temperatures, [300.0, 500.0, 700.0, 900.0])
        self.assertEqual(sources, [4.85, 6.85, 8.85])
        self.assertEqual(len(identifiers), 60)

    def test_central_case_is_resolved_without_rounding(self) -> None:
        self.assertEqual(
            select_p418_condition("u0p20_T700_q6p85", self.value),
            (0.20, 700.0, 6.85),
        )

    def test_unpublished_interpolation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_p418_condition("u0p18_T650_q7p00", self.value)


if __name__ == "__main__":
    unittest.main()
