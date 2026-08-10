#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/plot_hccb_p418_steady_engineering_comparison.py"


class PlotP418SteadyEngineeringComparisonTest(unittest.TestCase):
    def test_plot_writes_vector_raster_and_metadata(self) -> None:
        columns = [
            "architecture",
            "split",
            "test_outlet_temperature_p95_K",
            "test_solid_maximum_temperature_p95_K",
            "test_cooling_wall_heat_over_generated_p95_percent",
            "test_interphase_net_heat_over_generated_p95_percent",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "comparison.csv"
            with table.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for method_index, architecture in enumerate(
                    ("response_surface", "pinn_data_only", "pinn", "graph", "transolver")
                ):
                    for split_index, split in enumerate(("interpolation", "extrapolation")):
                        scale = float(1 + method_index + 2 * split_index)
                        writer.writerow(
                            {
                                "architecture": architecture,
                                "split": split,
                                "test_outlet_temperature_p95_K": scale,
                                "test_solid_maximum_temperature_p95_K": 2.0 * scale,
                                "test_cooling_wall_heat_over_generated_p95_percent": 3.0 * scale,
                                "test_interphase_net_heat_over_generated_p95_percent": 4.0 * scale,
                            }
                        )
            completed = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--comparison-csv",
                    str(table),
                    "--output-dir",
                    str(root / "figures"),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertGreater((root / "figures/steady_engineering_error_comparison.pdf").stat().st_size, 1000)
            self.assertGreater((root / "figures/steady_engineering_error_comparison.png").stat().st_size, 1000)
            summary = json.loads(
                (root / "figures/steady_engineering_error_comparison.json").read_text()
            )
            self.assertEqual(summary["new_physical_parameters"], [])
            self.assertEqual(len(summary["panels"]), 4)


if __name__ == "__main__":
    unittest.main()
