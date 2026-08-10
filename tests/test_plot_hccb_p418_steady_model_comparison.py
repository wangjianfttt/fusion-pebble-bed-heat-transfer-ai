#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/plot_hccb_p418_steady_model_comparison.py"
METHODS = ("response_surface", "pinn_data_only", "pinn", "graph", "transolver")
SPLITS = (
    "interleaved_all_ranges",
    "temperature_extrapolation",
    "velocity_extrapolation",
    "heat_source_interpolation",
    "heat_source_extrapolation",
)
METRICS = (
    "test_fluid_temperature_normalized_rmse",
    "test_solid_temperature_normalized_rmse",
    "test_pressure_drop_p95_Pa",
    "test_solid_maximum_temperature_p95_K",
    "test_cooling_wall_heat_over_generated_p95_percent",
    "test_local_mass_l1_over_two_inlet_mean",
    "test_local_energy_l1_over_two_generated_power_mean",
)


def write_table(path: Path, omit_last: bool = False) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["architecture", "split", *METRICS])
        writer.writeheader()
        rows = [(method, split) for method in METHODS for split in SPLITS]
        if omit_last:
            rows.pop()
        for method_index, split_index in ((METHODS.index(m), SPLITS.index(s)) for m, s in rows):
            scale = float(1 + method_index + split_index)
            writer.writerow(
                {
                    "architecture": METHODS[method_index],
                    "split": SPLITS[split_index],
                    "test_fluid_temperature_normalized_rmse": 0.01 * scale,
                    "test_solid_temperature_normalized_rmse": 0.015 * scale,
                    "test_pressure_drop_p95_Pa": 2.0 * scale,
                    "test_solid_maximum_temperature_p95_K": 1.5 * scale,
                    "test_cooling_wall_heat_over_generated_p95_percent": 0.8 * scale,
                    "test_local_mass_l1_over_two_inlet_mean": 0.001 * scale,
                    "test_local_energy_l1_over_two_generated_power_mean": 0.002 * scale,
                }
            )


class PlotP418SteadyModelComparisonTest(unittest.TestCase):
    def test_complete_formal_matrix_writes_paper_figure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "comparison.csv"
            write_table(table)
            completed = subprocess.run(
                ["python3", str(SCRIPT), "--comparison-csv", str(table), "--output-dir", str(root / "figures")],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            for suffix in ("pdf", "svg", "png", "json"):
                self.assertGreater((root / f"figures/hccb_p418_steady_model_comparison.{suffix}").stat().st_size, 500)
            svg_text = (
                root / "figures/hccb_p418_steady_model_comparison.svg"
            ).read_text(encoding="utf-8")
            self.assertIn("<text", svg_text)
            self.assertNotIn("<image", svg_text)
            summary = json.loads((root / "figures/hccb_p418_steady_model_comparison.json").read_text())
            self.assertEqual(summary["matrix_shape"], [5, 5])
            self.assertEqual(summary["figure_size_inch"], [5.4, 6.7])
            self.assertGreaterEqual(summary["panel_width_to_height_ratio"], 1.15)
            self.assertLessEqual(summary["panel_width_to_height_ratio"], 1.35)
            self.assertEqual(summary["new_physical_parameter_values_added"], [])

    def test_partial_matrix_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "comparison.csv"
            write_table(table, omit_last=True)
            completed = subprocess.run(
                ["python3", str(SCRIPT), "--comparison-csv", str(table), "--output-dir", str(root / "figures")],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("matrix is incomplete", completed.stderr)

    def test_subplots_do_not_have_titles_above_axes(self) -> None:
        self.assertNotIn(".set_title(", SCRIPT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
