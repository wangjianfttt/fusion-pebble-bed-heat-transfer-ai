#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/analyze_hccb_p418_step_split_coverage.py"
PLAN = ROOT / "parameters/hccb_p418_transient_step_plan.json"
SPLITS = ROOT / "parameters/hccb_p418_step_response_splits.json"
DIMENSIONLESS = (
    ROOT
    / "results/hccb_p418_inlet_dimensionless_envelope/inlet_dimensionless_conditions.csv"
)


class P418StepSplitCoverageTest(unittest.TestCase):
    def test_pair_disjoint_split_has_no_endpoint_pair_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--plan",
                    str(PLAN),
                    "--splits",
                    str(SPLITS),
                    "--dimensionless-conditions",
                    str(DIMENSIONLESS),
                    "--output-dir",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            stress = summary["split_summaries"]["pair_disjoint_stress_test"]
            self.assertEqual(stress["endpoint_pair_overlap_count_across_roles"], 0)
            self.assertEqual(stress["curve_counts"], {"train": 6, "validation": 2, "test": 4})
            for name in ("direction_down_test", "direction_up_test"):
                row = summary["split_summaries"][name]
                self.assertGreater(row["endpoint_pair_overlap_count_across_roles"], 0)
                self.assertTrue(all(role == "validation" for role in row["test_reverse_curve_roles"].values()))
            self.assertEqual(summary["new_physical_parameters"], [])
            self.assertEqual(
                summary["dimensionless_parameter_ids"],
                ["P048", "P068", "P070", "P071", "P073", "P388", "P418", "P426"],
            )
            self.assertEqual(
                stress["test_curves_outside_training_dimensionless_range"], 0
            )
            self.assertIn(
                "particle_reynolds_inlet",
                stress["endpoint_dimensionless_ranges_by_role"]["train"],
            )
            self.assertTrue((output / "curve_coverage.csv").is_file())
            self.assertTrue((output / "P418_热阶跃训练测试范围_CN.md").is_file())


if __name__ == "__main__":
    unittest.main()
