from __future__ import annotations

import json
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_hccb_p418_transient_learning_curve",
    ROOT / "code/verify_hccb_p418_transient_learning_curve.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
check_learning_curve_plan = MODULE.check_learning_curve_plan


class TransientLearningCurvePlanTest(unittest.TestCase):
    def test_physical_curve_groups_are_complete_and_disjoint(self) -> None:
        result = check_learning_curve_plan(
            ROOT / "parameters/hccb_p418_transient_step_plan.json",
            ROOT / "parameters/hccb_p418_transient_learning_curve_splits.json",
            ROOT / "code/run_hccb_p418_transient_learning_curve.sh",
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["physical_curve_count"], 12)
        self.assertEqual(result["independent_training_curve_counts"], [3, 6])
        self.assertTrue(result["field_observations_are_not_independent_conditions"])

    def test_overlap_is_rejected(self) -> None:
        source = json.loads(
            (ROOT / "parameters/hccb_p418_transient_learning_curve_splits.json").read_text(
                encoding="utf-8"
            )
        )
        source["splits"]["transient_learning_n03_up"]["unused"][0] = source["splits"][
            "transient_learning_n03_up"
        ]["train"][0]
        with tempfile.TemporaryDirectory() as directory:
            broken = Path(directory) / "broken.json"
            broken.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "across roles"):
                check_learning_curve_plan(
                    ROOT / "parameters/hccb_p418_transient_step_plan.json",
                    broken,
                    ROOT / "code/run_hccb_p418_transient_learning_curve.sh",
                )

    def test_runner_defaults_to_print_only(self) -> None:
        completed = subprocess.run(
            ["bash", str(ROOT / "code/run_hccb_p418_transient_learning_curve.sh")],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("execute=0", completed.stdout)
        self.assertIn("No training started", completed.stdout)


if __name__ == "__main__":
    unittest.main()
