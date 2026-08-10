#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_unbounded_output_diagnostic.py"


def record(
    *,
    fluid: float,
    solid: float,
    pressure: float,
    nonpositive: int = 0,
) -> dict:
    return {
        "role": "train",
        "sequence_id": "temperature_down",
        "fluid_temperature_K": {
            "value": fluid,
            "nonpositive_count": nonpositive,
        },
        "solid_temperature_K": {"value": solid, "nonpositive_count": 0},
        "fluid_absolute_pressure_Pa": {
            "value": pressure,
            "nonpositive_count": 0,
        },
    }


class P418UnboundedOutputDiagnosticTest(unittest.TestCase):
    def write_diagnostic(
        self,
        path: Path,
        *,
        epochs: int,
        fluid: float,
        solid: float,
        pressure: float,
        nonpositive: int = 0,
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "status": "p418_checkpoint_physical_domain_diagnosis",
                    "completed_epochs": epochs,
                    "records": [
                        record(
                            fluid=fluid,
                            solid=solid,
                            pressure=pressure,
                            nonpositive=nonpositive,
                        )
                    ],
                }
            ),
            encoding="utf-8",
        )

    def run_builder(self, root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--data-only",
                str(root / "data_only.json"),
                "--physics",
                str(root / "physics.json"),
                "--json-output",
                str(root / "summary.json"),
                "--tex-output",
                str(root / "table.tex"),
            ],
            capture_output=True,
            text=True,
        )

    def test_builds_excluded_negative_control(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_diagnostic(
                root / "data_only.json",
                epochs=500,
                fluid=94.5,
                solid=69.5,
                pressure=119999.0,
            )
            self.write_diagnostic(
                root / "physics.json",
                epochs=42,
                fluid=-18.0,
                solid=634.8,
                pressure=119999.0,
                nonpositive=4,
            )
            result = self.run_builder(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(
                summary["status"],
                "completed_p418_unbounded_output_diagnostic",
            )
            self.assertFalse(summary["formal_model_ranking_included"])
            self.assertEqual(
                summary["records"][1]["nonpositive_fluid_temperature_count"],
                4,
            )
            table = (root / "table.tex").read_text(encoding="ascii")
            self.assertIn("excluded from the final model comparison", table)
            self.assertIn("specified thermophysical intervals", table)
            self.assertIn("-18.000", table)

    def test_rejects_missing_nonpositive_temperature(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_diagnostic(
                root / "data_only.json",
                epochs=500,
                fluid=94.5,
                solid=69.5,
                pressure=119999.0,
            )
            self.write_diagnostic(
                root / "physics.json",
                epochs=42,
                fluid=0.6,
                solid=634.8,
                pressure=119999.0,
            )
            result = self.run_builder(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("non-positive fluid temperature", result.stderr)


if __name__ == "__main__":
    unittest.main()
