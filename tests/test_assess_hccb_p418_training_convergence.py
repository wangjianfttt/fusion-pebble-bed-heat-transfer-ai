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
SCRIPT = ROOT / "code/assess_hccb_p418_training_convergence.py"


def write_registry(path: Path) -> None:
    records = [
        ("PINO-paper coordinate PINN control", 3000),
        ("RIGNO-style regional graph operator", 2000),
        ("Transolver", 500),
    ]
    path.write_text(
        json.dumps(
            {
                "architectures": [
                    {"name": name, "source_settings": {"epochs": epochs}}
                    for name, epochs in records
                ]
            }
        ),
        encoding="utf-8",
    )


def write_history(path: Path, losses: list[float]) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        "".join(
            json.dumps(
                {"epoch": epoch, "validation": {"total_loss": loss}}
            )
            + "\n"
            for epoch, loss in enumerate(losses, start=1)
        ),
        encoding="utf-8",
    )


class P418TrainingConvergenceTest(unittest.TestCase):
    def run_assessment(
        self,
        root: Path,
        *,
        architecture: str,
        losses: list[float],
        epochs: int | None = None,
        result_prefix: str = "hccb_p418_60",
        split_name: str = "formal",
    ) -> subprocess.CompletedProcess[str]:
        requested = len(losses) if epochs is None else epochs
        history = (
            root
            / "results"
            / f"{result_prefix}_{architecture}_{split_name}_{requested}epoch"
            / "training_history.jsonl"
        )
        write_history(history, losses)
        registry = root / "registry.json"
        write_registry(registry)
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--results-root",
                str(root / "results"),
                "--epochs",
                str(requested),
                "--result-prefix",
                result_prefix,
                "--architectures",
                architecture,
                "--splits",
                split_name,
                "--architecture-registry",
                str(registry),
                "--output-dir",
                str(root / "output"),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_declining_loss_requires_longer_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed = self.run_assessment(
                root,
                architecture="pinn",
                losses=[1.0 / epoch for epoch in range(1, 21)],
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads((root / "output/training_convergence.json").read_text())
            model = payload["models"][0]
            self.assertTrue(model["best_epoch_is_final_epoch"])
            self.assertEqual(model["published_source_epochs"], 3000)
            self.assertEqual(payload["training_extension_required_count"], 1)
            self.assertEqual(payload["recommended_followup_epochs"]["formal"]["pinn"], 3000)

    def test_interior_best_checkpoint_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed = self.run_assessment(
                root,
                architecture="transolver",
                losses=[4.0, 3.0, 2.0, 1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with (root / "output/training_convergence.csv").open(encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(int(row["best_epoch"]), 4)
            self.assertEqual(row["best_epoch_is_final_epoch"], "False")
            self.assertEqual(int(row["published_source_epochs"]), 500)
            payload = json.loads((root / "output/training_convergence.json").read_text())
            self.assertEqual(
                payload["recommended_followup_epochs"]["formal"]["transolver"], 10
            )

    def test_incomplete_history_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed = self.run_assessment(
                root,
                architecture="graph",
                losses=[1.0, 0.8, 0.7],
                epochs=4,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("incomplete or non-consecutive", completed.stderr)

    def test_custom_result_prefix_and_split_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = "hccb_p418_mixed_endpoint_smoke_model"
            completed = self.run_assessment(
                root,
                architecture="pinn",
                losses=[1.0],
                result_prefix=prefix,
                split_name="completed_smoke",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads((root / "output/training_convergence.json").read_text())
            self.assertEqual(payload["result_prefix"], prefix)


if __name__ == "__main__":
    unittest.main()
