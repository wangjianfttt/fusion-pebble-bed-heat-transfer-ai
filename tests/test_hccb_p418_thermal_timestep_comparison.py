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
sys.path.insert(0, str(ROOT / "code"))

from compare_hccb_p418_thermal_timestep_sensitivity import (
    compare_pair,
    gci_triplet,
    label,
    refinement_trend,
    select_finest_declared_step,
)


class P418ThermalTimestepComparisonTest(unittest.TestCase):
    def test_successive_curve_metrics_are_exact(self) -> None:
        time = np.asarray([0.0, 1.0, 2.0])
        coarse = {"T": np.asarray([1.0, 2.0, 4.0])}
        fine = {"T": np.asarray([1.0, 2.0, 3.0])}
        row = compare_pair(time, coarse, time, fine)[0]
        self.assertAlmostEqual(row["maximum_absolute_difference"], 1.0)
        self.assertAlmostEqual(row["endpoint_absolute_difference"], 1.0)
        self.assertAlmostEqual(row["endpoint_relative_difference"], 1.0 / 3.0)
        self.assertAlmostEqual(row["maximum_difference_over_response_span"], 0.5)
        self.assertAlmostEqual(row["endpoint_difference_over_response_span"], 0.5)

    def test_delta_t_label_is_stable(self) -> None:
        self.assertEqual(label(1.0), "dt_1")
        self.assertEqual(label(0.25), "dt_0p25")
        self.assertEqual(label(1.0e-5), "dt_1em05")

    def test_formal_step_is_the_finest_predeclared_resolution(self) -> None:
        self.assertEqual(select_finest_declared_step([1.0, 0.5, 0.25]), 0.25)
        with self.assertRaises(ValueError):
            select_finest_declared_step([1.0])

    def test_refinement_trend_is_reported_without_selecting_a_threshold(self) -> None:
        rows = [
            {
                "coarse_delta_t_s": 1.0,
                "fine_delta_t_s": 0.5,
                "signal": "T",
                "maximum_difference_over_response_span": 0.005,
                "endpoint_difference_over_response_span": 0.004,
            },
            {
                "coarse_delta_t_s": 0.5,
                "fine_delta_t_s": 0.25,
                "signal": "T",
                "maximum_difference_over_response_span": 0.02,
                "endpoint_difference_over_response_span": 0.004,
            },
        ]
        trend = refinement_trend(rows, [1.0, 0.5, 0.25])["T"]
        self.assertFalse(trend["successive_maximum_difference_decreases"])
        self.assertEqual(trend["finest_pair_maximum_difference_over_response_span"], 0.02)

    def test_monotonic_three_step_gci_matches_equal_ratio_formula(self) -> None:
        result = gci_triplet(12.0, 11.0, 10.5, 2.0, 1.25)
        self.assertEqual(result["convergence_status"], "monotonic_gci_reported")
        self.assertAlmostEqual(result["observed_order"], 1.0)
        self.assertAlmostEqual(result["richardson_extrapolated_value"], 10.0)
        self.assertAlmostEqual(result["fine_gci_absolute"], 0.625)
        self.assertAlmostEqual(result["fine_gci_fraction"], 0.625 / 10.5)

    def test_oscillatory_triplet_is_labelled_without_forcing_gci(self) -> None:
        result = gci_triplet(10.0, 11.0, 10.5, 2.0, 1.25)
        self.assertEqual(result["convergence_status"], "oscillatory_no_gci_reported")
        self.assertIsNone(result["fine_gci_fraction"])

    def test_command_writes_finest_step_and_gci_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_root = root / "curves"
            time = np.asarray([[0.0, 1.0, 2.0]])
            for delta_t, curve in (
                (1.0, [0.0, 6.0, 12.0]),
                (0.5, [0.0, 5.5, 11.0]),
                (0.25, [0.0, 5.25, 10.5]),
            ):
                directory = result_root / label(delta_t)
                directory.mkdir(parents=True)
                np.savez(
                    directory / "hccb_p418_transient_observables.npz",
                    case_id=np.asarray(["case"]),
                    signal_names=np.asarray(["T"]),
                    time_mask=np.asarray([[True, True, True]]),
                    time_s=time,
                    values=np.asarray(curve, dtype=float)[None, :, None],
                )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "sequence_id": "test",
                        "comparison_quantities": ["T"],
                        "delta_t_s": [1.0, 0.5, 0.25],
                        "formal_time_step_schedule": [
                            {"start_s": 0.0, "end_s": 2.0, "delta_t_s": 0.25}
                        ],
                        "comparison_rule": "report differences",
                        "formal_selection_rule": "finest_completed_predeclared_step",
                        "discretization_uncertainty_method": {
                            "name": "GCI",
                            "refinement_ratio": 2.0,
                            "safety_factor": 1.25,
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "output"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "code/compare_hccb_p418_thermal_timestep_sensitivity.py"),
                    "--config",
                    str(config),
                    "--result-root",
                    str(result_root),
                    "--output-dir",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["selected_delta_t_s"], 0.25)
            self.assertTrue((output / "thermal_timestep_gci.csv").is_file())
            endpoint = next(
                row for row in summary["gci_results"] if row["quantity"] == "endpoint"
            )
            self.assertEqual(endpoint["convergence_status"], "monotonic_gci_reported")
            self.assertAlmostEqual(endpoint["fine_gci_absolute"], 0.625)


if __name__ == "__main__":
    unittest.main()
