#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

from summarize_hccb_p418_learning_curve_efficiency import summarize  # noqa: E402


class P418LearningCurveEfficiencyTest(unittest.TestCase):
    def test_measured_restart_clock_time_is_joined_to_model_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = root / "matrix"
            for condition, segments in {
                "u0p05_T300_q4p85": [(5.0, 7.0), (3.0,)],
                "u0p10_T500_q6p85": [(4.0,)],
            }.items():
                case = matrix / condition
                case.mkdir(parents=True)
                text = ""
                for index, values in enumerate(segments):
                    if index:
                        text += "resumed from complete parallel time 25 at 2026-07-19T12:00:00\n"
                    for value in values:
                        text += f"ClockTime = {value} s\n"
                (case / "log.foamMultiRun.formal").write_text(text, encoding="utf-8")
            splits = root / "splits.json"
            splits.write_text(
                json.dumps(
                    {
                        "splits": {
                            "learning_curve_n02": {
                                "train": [
                                    "u0p05_T300_q4p85",
                                    "u0p10_T500_q6p85",
                                ],
                                "validation": ["v"],
                                "test": ["t"],
                                "unused": [],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            comparison = root / "comparison.csv"
            with comparison.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "architecture",
                        "split",
                        "train_case_count",
                        "validation_case_count",
                        "test_case_count",
                        "test_state_normalized_rmse",
                        "test_fluid_temperature_normalized_rmse",
                        "test_solid_temperature_normalized_rmse",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "architecture": "pinn",
                        "split": "learning_curve_n02",
                        "train_case_count": 2,
                        "validation_case_count": 1,
                        "test_case_count": 1,
                        "test_state_normalized_rmse": 0.1,
                        "test_fluid_temperature_normalized_rmse": 0.08,
                        "test_solid_temperature_normalized_rmse": 0.12,
                    }
                )
            output = root / "output"
            payload = summarize(
                comparison_csv=comparison,
                split_file=splits,
                matrix_root=matrix,
                output_dir=output,
                expected_training_counts=[2],
                expected_architectures=["pinn"],
                expected_split_names=["learning_curve_n02"],
                expected_validation_count=1,
                expected_test_count=1,
            )
            self.assertEqual(payload["training_condition_counts"], [2])
            with (output / "learning_curve_efficiency.csv").open(
                encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            # Case a contributes max(5,7)+3=10 s, and case b contributes 4 s.
            self.assertAlmostEqual(float(rows[0]["openfoam_training_solver_hours"]), 14.0 / 3600.0)
            self.assertAlmostEqual(float(rows[0]["openfoam_training_core_hours_32ranks"]), 14.0 * 32.0 / 3600.0)

            with self.assertRaisesRegex(
                ValueError, "learning-curve architectures differ"
            ):
                summarize(
                    comparison_csv=comparison,
                    split_file=splits,
                    matrix_root=matrix,
                    output_dir=output,
                    expected_training_counts=[2],
                    expected_architectures=["pinn", "graph"],
                    expected_split_names=["learning_curve_n02"],
                    expected_validation_count=1,
                    expected_test_count=1,
                )


if __name__ == "__main__":
    unittest.main()
