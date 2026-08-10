#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_steady_performance_table.py"
METHODS = ("response_surface", "pinn_data_only", "pinn", "graph", "transolver")
SPLITS = (
    "interleaved_all_ranges",
    "temperature_extrapolation",
    "velocity_extrapolation",
    "heat_source_interpolation",
    "heat_source_extrapolation",
)
FIELDS = (
    "architecture",
    "split",
    "test_fluid_temperature_normalized_rmse",
    "test_solid_temperature_normalized_rmse",
    "test_pressure_drop_p95_Pa",
    "test_solid_maximum_temperature_p95_K",
    "test_cooling_wall_heat_over_generated_p95_percent",
    "test_local_mass_l1_over_two_inlet_mean",
    "test_local_energy_l1_over_two_generated_power_mean",
    "model_parameter_count",
    "training_wall_time_s",
    "test_inference_s_per_case",
)


def write_fixture(path: Path, varying_parameters: bool = False) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for method_index, method in enumerate(METHODS):
            for split_index, split in enumerate(SPLITS):
                scale = float(1 + method_index + split_index)
                writer.writerow(
                    {
                        "architecture": method,
                        "split": split,
                        "test_fluid_temperature_normalized_rmse": 0.01 * scale,
                        "test_solid_temperature_normalized_rmse": 0.015 * scale,
                        "test_pressure_drop_p95_Pa": 2.0 * scale,
                        "test_solid_maximum_temperature_p95_K": 1.5 * scale,
                        "test_cooling_wall_heat_over_generated_p95_percent": 0.8 * scale,
                        "test_local_mass_l1_over_two_inlet_mean": 0.001 * scale,
                        "test_local_energy_l1_over_two_generated_power_mean": 0.002 * scale,
                        "model_parameter_count": 1000 * (method_index + 1) + (
                            split_index if varying_parameters and method_index == 0 else 0
                        ),
                        "training_wall_time_s": 60.0 * scale,
                        "test_inference_s_per_case": 0.001 * scale,
                    }
                )


class BuildSteadyPerformanceTableTest(unittest.TestCase):
    def test_complete_matrix_generates_tex_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "comparison.csv"
            write_fixture(source)
            completed = subprocess.run(
                [
                    "python3", str(SCRIPT), "--comparison-csv", str(source),
                    "--output", str(root / "generated.tex"), "--summary", str(root / "summary.json"),
                ],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            tex = (root / "generated.tex").read_text()
            self.assertIn("Steady prediction accuracy and computational cost", tex)
            self.assertIn("physics PINN", tex)
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["physical_split_count"], 5)
            self.assertEqual(len(summary["records"]), 5)
            self.assertEqual(summary["new_physical_parameter_values_added"], [])

    def test_parameter_count_must_be_constant_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "comparison.csv"
            write_fixture(source, varying_parameters=True)
            completed = subprocess.run(
                [
                    "python3", str(SCRIPT), "--comparison-csv", str(source),
                    "--output", str(root / "generated.tex"), "--summary", str(root / "summary.json"),
                ],
                text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("parameter count changes", completed.stderr)


if __name__ == "__main__":
    unittest.main()
