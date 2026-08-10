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
SCRIPT = ROOT / "code/build_hccb_p418_regional_model_table.py"
SPLIT = "pair_disjoint_stress_test"
MODELS = (
    ("regional_persistence", None),
    ("regional_dmdc", "selected_rank"),
    ("regional_graph_transformer_bounded_data_only", "selected_epoch"),
    ("regional_graph_transformer_bounded_physics", "selected_epoch"),
    ("regional_graph_transformer_bounded_factorized", "selected_epoch"),
)


class P418RegionalModelTableTest(unittest.TestCase):
    def write_summary(
        self, root: Path, prefix: str, selection_key: str | None, value: float
    ) -> None:
        directory = root / f"{prefix}_{SPLIT}"
        directory.mkdir()
        split_ids = {
            "train": ["train-a"],
            "validation": ["validation-a"],
            "test": ["test-a"],
        }
        payload = {
            "split_name": SPLIT,
            "selection_split": (
                "validation" if selection_key is not None else "not_applicable"
            ),
            "split_case_ids": split_ids,
            "temperature_metric_definition": (
                "regional-volume-weighted RMSE, reported separately for fluid and solid"
            ),
            "training_seconds": 10.0 if selection_key is not None else 0.0,
            "model_parameter_count": 1 if selection_key is not None else 0,
            "compute_device": (
                "cpu"
                if selection_key in (None, "selected_rank")
                else "cuda:0"
            ),
            "metrics": {
                "test": {
                    "fluid_temperature_RMSE_K": value + 0.2,
                    "solid_temperature_RMSE_K": value,
                    "solid_maximum_temperature_history_RMSE_K": value + 1.0,
                    "solid_regional_hotspot_location_p95_error_m": value / 1000.0,
                }
            },
        }
        if selection_key is not None:
            payload[selection_key] = (
                8 if selection_key == "selected_rank" else 12
            )
        (directory / "summary.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_builds_complete_regional_table(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, (prefix, selection_key) in enumerate(MODELS):
                self.write_summary(root, prefix, selection_key, 5.0 - index)
            csv_path = root / "comparison.csv"
            summary_path = root / "summary.json"
            tex_path = root / "comparison.tex"
            text_path = root / "comparison_text.tex"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--result-root",
                    str(root),
                    "--csv",
                    str(csv_path),
                    "--summary",
                    str(summary_path),
                    "--tex",
                    str(tex_path),
                    "--text",
                    str(text_path),
                ],
                check=True,
            )
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(
                summary["status"],
                "complete_p418_like_for_like_regional_model_comparison",
            )
            self.assertIn("46\\,089-node", tex_path.read_text(encoding="utf-8"))
            text = text_path.read_text(encoding="utf-8")
            self.assertIn("repeating the initial temperature", text)
            self.assertIn("lowest solid-field error", text)
            self.assertIn("hotspot p95", text)

    def test_incomplete_results_do_not_write_manuscript_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_summary(root, *MODELS[0], 5.0)
            csv_path = root / "comparison.csv"
            summary_path = root / "summary.json"
            tex_path = root / "comparison.tex"
            text_path = root / "comparison_text.tex"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--result-root",
                    str(root),
                    "--csv",
                    str(csv_path),
                    "--summary",
                    str(summary_path),
                    "--tex",
                    str(tex_path),
                    "--text",
                    str(text_path),
                    "--allow-incomplete",
                ],
                check=True,
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(
                summary["status"],
                "incomplete_p418_like_for_like_regional_model_comparison",
            )
            self.assertFalse(tex_path.exists())
            self.assertFalse(text_path.exists())

    def test_rejects_mismatched_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, (prefix, selection_key) in enumerate(MODELS):
                self.write_summary(root, prefix, selection_key, 5.0 - index)
            path = root / f"{MODELS[-1][0]}_{SPLIT}" / "summary.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["split_case_ids"]["test"] = ["different-test"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--result-root",
                    str(root),
                    "--csv",
                    str(root / "comparison.csv"),
                    "--summary",
                    str(root / "summary.json"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("different train/validation/test split", result.stderr)


if __name__ == "__main__":
    unittest.main()
