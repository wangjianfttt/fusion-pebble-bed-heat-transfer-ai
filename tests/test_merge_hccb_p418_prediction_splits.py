#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/merge_hccb_p418_prediction_splits.py"
RUNNER = ROOT / "code/run_hccb_p418_learned_model_experimental_comparison.sh"


def prediction_file(path: Path, identifier: str, value: float) -> None:
    np.savez_compressed(
        path,
        condition_id=np.asarray([identifier]),
        condition_normalized=np.asarray([[value, 0.0]]),
        baseline_state_normalized=np.full((1, 2, 5), value),
        target_state_normalized=np.full((1, 2, 5), value + 1.0),
        node_type=np.asarray([0, 1], dtype=np.int8),
        node_volume_m3=np.asarray([1.0, 2.0]),
    )


class MergeP418PredictionSplitsTest(unittest.TestCase):
    def test_merge_preserves_all_conditions_and_sorts_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("train.npz", "validation.npz", "test.npz")]
            prediction_file(paths[0], "case_b", 2.0)
            prediction_file(paths[1], "case_a", 1.0)
            prediction_file(paths[2], "case_c", 3.0)
            output = root / "all.npz"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(paths[0]),
                    "--input",
                    str(paths[1]),
                    "--input",
                    str(paths[2]),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with np.load(output, allow_pickle=False) as loaded:
                self.assertEqual(loaded["condition_id"].tolist(), ["case_a", "case_b", "case_c"])
                self.assertEqual(loaded["baseline_state_normalized"][:, 0, 0].tolist(), [1.0, 2.0, 3.0])
            summary = json.loads(
                (root / "all_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["condition_count"], 3)
            self.assertEqual(summary["new_physical_parameters"], [])

    def test_empty_experimental_tables_stop_before_model_file_merge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison"
            env = os.environ.copy()
            env.update(
                {
                    "ROOT": str(ROOT),
                    "DATA_ROOT": str(ROOT / "experimental_data_templates"),
                    "MODEL_OUTPUT": str(Path(directory) / "not_created_model"),
                    "MODEL_NAME": "software fixture model",
                    "OUTPUT_DIR": str(output),
                }
            )
            subprocess.run(
                ["bash", str(RUNNER)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "no_experimental_measurements")
            self.assertFalse((output / "all_condition_regional_predictions.npz").exists())


if __name__ == "__main__":
    unittest.main()
