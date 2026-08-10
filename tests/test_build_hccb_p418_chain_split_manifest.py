#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


class BuildP418ChainSplitManifestTest(unittest.TestCase):
    def test_repository_split_manifest_has_strict_temperature_tests(self) -> None:
        import json

        from build_hccb_p418_chain_split_manifest import build_rows, summarize

        steady = json.loads(
            (ROOT / "parameters/hccb_p418_model_splits.json").read_text()
        )
        plan = json.loads(
            (ROOT / "parameters/hccb_p418_transient_step_plan.json").read_text()
        )
        transient = json.loads(
            (ROOT / "parameters/hccb_p418_step_response_splits.json").read_text()
        )
        rows = build_rows(steady, plan, transient, "interleaved_all_ranges")
        self.assertEqual(len(rows), 36)
        payload = summarize(rows)
        self.assertEqual(payload["transient_test_row_count"], 10)
        strict = payload["strict_end_to_end_test_curves"]
        self.assertEqual(len(strict), 2)
        self.assertEqual(
            {row["sequence_id"] for row in strict},
            {"temperature_up_u0p15_q6p85", "temperature_down_u0p15_q6p85"},
        )


if __name__ == "__main__":
    unittest.main()
