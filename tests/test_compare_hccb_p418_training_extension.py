#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/compare_hccb_p418_training_extension.py"


def payload(epochs: int, value: float) -> dict[str, object]:
    return {
        "architecture": "graph",
        "split_name": "same",
        "split_case_ids": {"train": ["a", "b"], "validation": ["c"], "test": ["d"]},
        "epochs": epochs,
        "best_epoch": epochs - 1,
        "training_seconds": value * 10,
        "peak_gpu_memory_GB": 3.0,
        "run_provenance": {"common_comparison_fingerprint": "same"},
        "evaluations": {
            "validation": {"metrics": {"total_loss": value}},
            "test": {
                "metrics": {"state_normalized_rmse": value},
                "cases": [
                    {
                        "engineering_absolute_errors": {
                            "outlet_temperature_K": value,
                            "solid_maximum_temperature_K": value,
                            "cooling_wall_heat_into_fluid_W": value,
                        },
                        "global_energy_imbalance_over_generated_power": value,
                    }
                ],
            },
        },
    }


class P418TrainingExtensionTest(unittest.TestCase):
    def test_writes_traceable_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            short = root / "short.json"
            long = root / "long.json"
            short.write_text(json.dumps(payload(100, 2.0)), encoding="utf-8")
            long.write_text(json.dumps(payload(200, 1.5)), encoding="utf-8")
            output = root / "output"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--short-summary",
                    str(short),
                    "--long-summary",
                    str(long),
                    "--output-dir",
                    str(output),
                ],
                check=True,
            )
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["train_condition_count"], 2)
            self.assertAlmostEqual(
                summary["long_minus_short"]["test_outlet_temperature_error_K"],
                -0.5,
            )
            rows = list(csv.DictReader((output / "training_extension_comparison.csv").open()))
            self.assertEqual([row["label"] for row in rows], ["short", "long"])


if __name__ == "__main__":
    unittest.main()
