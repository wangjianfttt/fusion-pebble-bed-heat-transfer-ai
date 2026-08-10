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
SCRIPT = ROOT / "code/estimate_hccb_p418_training_wall_time.py"


class P418TrainingWallTimeTest(unittest.TestCase):
    def test_projection_uses_batches_and_complete_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            for architecture in ("pinn_data_only", "pinn", "graph", "transolver"):
                target = reference / architecture
                target.mkdir(parents=True)
                (target / "summary.json").write_text(
                    json.dumps(
                        {
                            "epochs": 1,
                            "effective_batch_size": 2,
                            "microbatch_size": 1,
                            "device": "cuda",
                            "peak_gpu_memory_GB": 3.5,
                            "optimization_seconds_per_update": 10.0,
                            "validation_seconds": 2.0,
                            "final_evaluation_seconds": 10.0,
                            "split_case_ids": {
                                "train": ["a", "b", "c"],
                                "validation": ["d"],
                                "test": ["e"],
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            split = root / "splits.json"
            split.write_text(
                json.dumps(
                    {
                        "splits": {
                            "formal": {
                                "train": list("abcdef"),
                                "validation": ["g", "h"],
                                "test": ["i", "j"],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            output = root / "output"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--reference-root",
                    str(reference),
                    "--split-file",
                    str(split),
                    "--split-names",
                    "formal",
                    "--epochs",
                    "4",
                    "--output-dir",
                    str(output),
                ],
                check=True,
            )
            with (output / "projected_training_wall_time.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual(int(rows[0]["parameter_updates"]), 12)
            self.assertEqual(rows[0]["device"], "cuda")
            self.assertEqual(rows[0]["microbatch_size"], "1")
            self.assertEqual(rows[0]["measured_peak_gpu_memory_GB"], "3.5")
            self.assertAlmostEqual(float(rows[0]["projected_total_h"]), 156.0 / 3600.0)


if __name__ == "__main__":
    unittest.main()
