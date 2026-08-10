#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from summarize_hccb_p418_smoke_repeatability import relative_difference  # noqa: E402


class SmokeRepeatabilitySummaryTest(unittest.TestCase):
    def test_relative_difference_is_symmetric_and_scaled(self) -> None:
        self.assertAlmostEqual(relative_difference(100.0, 99.0), 0.01)
        self.assertAlmostEqual(relative_difference(99.0, 100.0), 0.01)
        self.assertEqual(relative_difference(0.0, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
