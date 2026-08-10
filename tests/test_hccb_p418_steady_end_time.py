#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

from summarize_hccb_p418_steady_end_time import value_at  # noqa: E402


class P418SteadyEndTimeTest(unittest.TestCase):
    def test_value_at_requires_exact_saved_iteration(self) -> None:
        history = np.asarray([[0.0, 1.0], [200.0, 2.0], [300.0, 3.0]])
        self.assertEqual(value_at(history, 200.0), 2.0)
        with self.assertRaises(ValueError):
            value_at(history, 250.0)

    def test_updater_changes_only_unfinished_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = root / "matrix"
            completed = matrix / "u0p05_T300_q4p85"
            unfinished = matrix / "u0p05_T300_q6p85"
            for case in (completed, unfinished):
                (case / "system").mkdir(parents=True)
                (case / "system/controlDict").write_text("endTime 300;\n", encoding="utf-8")
                (case / "cht_smoke_metadata.json").write_text(
                    json.dumps({"end_time": 300}), encoding="utf-8"
                )
            (completed / "formal_sample_complete.json").write_text("{}\n", encoding="utf-8")
            (unfinished / "transient_snapshot_plan.json").write_text(
                json.dumps({"steady_snapshot_iterations": [25, 100, 200, 225, 300]}),
                encoding="utf-8",
            )
            evidence = root / "summary.json"
            evidence.write_text(
                json.dumps(
                    {
                        "status": "steady_iteration_endpoint_comparison_complete",
                        "recommended_steady_end_iteration": 200,
                        "completed_reference_case_count": 5,
                        "decomposed_full_field_case_count": 3,
                    }
                ),
                encoding="utf-8",
            )
            completed_run = subprocess.run(
                [
                    sys.executable,
                    str(CODE / "apply_hccb_p418_steady_end_time.py"),
                    "--matrix-root",
                    str(matrix),
                    "--end-time",
                    "200",
                    "--evidence-summary",
                    str(evidence),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed_run.returncode, 0, completed_run.stderr)
            self.assertIn("endTime 300", (completed / "system/controlDict").read_text())
            self.assertIn("endTime 200", (unfinished / "system/controlDict").read_text())
            metadata = json.loads((unfinished / "cht_smoke_metadata.json").read_text())
            self.assertEqual(metadata["end_time"], 200)
            self.assertEqual(metadata["steady_iteration_end"], 200)
            self.assertEqual(metadata["solver_time_semantics"], "steady_iteration_index")
            self.assertIsNone(metadata["physical_time_s"])
            snapshots = json.loads((unfinished / "transient_snapshot_plan.json").read_text())
            self.assertEqual(snapshots["steady_snapshot_iterations"], [25, 100, 200])
            self.assertNotIn("snapshot_times_s", snapshots)


if __name__ == "__main__":
    unittest.main()
