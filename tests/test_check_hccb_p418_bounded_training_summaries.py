#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/check_hccb_p418_bounded_training_summaries.py"
MODELS = (
    ("data.json", "formal_data_only", "data_only"),
    ("physics.json", "formal", "energy_and_flux"),
    ("factorized.json", "formal_factorized", "energy_and_flux"),
)


class P418BoundedTrainingSummariesTest(unittest.TestCase):
    def write_summaries(self, root: Path) -> None:
        for filename, run_role, physics_mode in MODELS:
            (root / filename).write_text(
                json.dumps(
                    {
                        "status": "completed_p418_spatiotemporal_regional_operator",
                        "run_role": run_role,
                        "physics_mode": physics_mode,
                        "split_name": "pair_disjoint_stress_test",
                        "architecture": {
                            "temperature_output_mode": "literature_bounded_residual"
                        },
                    }
                ),
                encoding="utf-8",
            )

    def command(self, root: Path) -> list[str]:
        return [
            sys.executable,
            str(SCRIPT),
            "--data-only",
            str(root / "data.json"),
            "--physics",
            str(root / "physics.json"),
            "--factorized",
            str(root / "factorized.json"),
            "--output",
            str(root / "checked.json"),
        ]

    def test_accepts_three_bounded_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_summaries(root)
            subprocess.run(self.command(root), check=True)
            result = json.loads((root / "checked.json").read_text())
            self.assertEqual(
                result["status"],
                "p418_three_bounded_graph_transformer_summaries_checked",
            )
            self.assertEqual(len(result["records"]), 3)

    def test_rejects_unbounded_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_summaries(root)
            path = root / "physics.json"
            payload = json.loads(path.read_text())
            payload["architecture"]["temperature_output_mode"] = "unbounded"
            path.write_text(json.dumps(payload))
            result = subprocess.run(
                self.command(root), capture_output=True, text=True
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("temperature_output_mode", result.stderr)


if __name__ == "__main__":
    unittest.main()
