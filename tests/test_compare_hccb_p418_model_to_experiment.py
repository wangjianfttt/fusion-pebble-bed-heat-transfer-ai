#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/compare_hccb_p418_model_to_experiment.py"
sys.path.insert(0, str(ROOT / "code"))

from compare_hccb_p418_model_to_experiment import (  # noqa: E402
    load_temporal_states,
    transient_value,
)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class P418ModelExperimentComparisonTest(unittest.TestCase):
    def test_transient_temperature_is_interpolated_only_inside_model_time_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporal_path = Path(directory) / "temporal.npz"
            normalized = np.zeros((1, 2, 4, 1), dtype=float)
            normalized[0, :, 2, 0] = [100.0, 200.0]
            np.savez_compressed(
                temporal_path,
                sequence_id=np.asarray(["step_a"]),
                time_s=np.asarray([[0.0, 10.0]]),
                baseline_temperature_normalized=normalized,
                target_temperature_normalized=normalized,
                node_type=np.asarray([0, 0, 1, 1], dtype=np.int8),
                temperature_mean_K_by_node_type=np.asarray([0.0, 0.0]),
                temperature_std_K_by_node_type=np.asarray([1.0, 1.0]),
            )
            temporal = load_temporal_states(temporal_path, "prediction")
            value, method, coordinate, distance = transient_value(
                temporal,
                "step_a",
                "solid_temperature",
                5.0,
                np.asarray(
                    [
                        [0.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [3.0, 0.0, 0.0],
                    ]
                ),
                np.asarray([1.02, 0.0, 0.0]),
                None,
            )
            self.assertAlmostEqual(value, 150.0)
            self.assertIn("model time", method)
            np.testing.assert_allclose(coordinate, [1.0, 0.0, 0.0])
            self.assertAlmostEqual(distance, 0.02)

    def test_point_and_integral_measurements_use_declared_model_quantities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            output = root / "output"
            data.mkdir()
            write_rows(
                data / "experiment_conditions.csv",
                [
                    "experiment_id",
                    "model_condition_id",
                    "material",
                    "packing_id",
                    "run_type",
                    "source_reference",
                    "geometry_reference",
                    "notes",
                ],
                [
                    {
                        "experiment_id": "software_fixture",
                        "model_condition_id": "case_a",
                        "material": "test_only",
                        "packing_id": "test_only",
                        "run_type": "software_test_not_research_data",
                        "source_reference": "unit-test fixture",
                        "geometry_reference": "unit-test fixture",
                        "notes": "not experimental evidence",
                    }
                ],
            )
            write_rows(
                data / "sensor_layout.csv",
                [
                    "experiment_id",
                    "sensor_id",
                    "quantity",
                    "model_observable",
                    "sensor_response_model",
                    "x_m",
                    "y_m",
                    "z_m",
                    "measurement_method",
                    "calibration_id",
                    "notes",
                ],
                [
                    {
                        "experiment_id": "software_fixture",
                        "sensor_id": "solid_tc",
                        "quantity": "temperature",
                        "model_observable": "solid_temperature",
                        "sensor_response_model": "nearest_regional_phase_temperature",
                        "x_m": 1.02,
                        "y_m": 0.0,
                        "z_m": 0.0,
                        "measurement_method": "software fixture",
                        "calibration_id": "cal_test",
                        "notes": "not research data",
                    },
                    {
                        "experiment_id": "software_fixture",
                        "sensor_id": "bed_dp",
                        "quantity": "differential_pressure",
                        "model_observable": "pressure_drop",
                        "sensor_response_model": "direct_integral_or_boundary_quantity",
                        "x_m": "",
                        "y_m": "",
                        "z_m": "",
                        "measurement_method": "software fixture",
                        "calibration_id": "cal_test",
                        "notes": "not research data",
                    },
                    {
                        "experiment_id": "software_fixture",
                        "sensor_id": "explicit_tc",
                        "quantity": "temperature",
                        "model_observable": "solid_temperature",
                        "sensor_response_model": "explicit_sensor_body",
                        "x_m": 1.02,
                        "y_m": 0.0,
                        "z_m": 0.0,
                        "measurement_method": "software fixture",
                        "calibration_id": "cal_test",
                        "notes": "not research data",
                    },
                ],
            )
            write_rows(
                data / "steady_measurements.csv",
                [
                    "experiment_id",
                    "sensor_id",
                    "quantity",
                    "value",
                    "unit",
                    "standard_uncertainty",
                    "averaging_start_s",
                    "averaging_end_s",
                    "notes",
                ],
                [
                    {
                        "experiment_id": "software_fixture",
                        "sensor_id": "solid_tc",
                        "quantity": "temperature",
                        "value": 698.0,
                        "unit": "K",
                        "standard_uncertainty": 2.0,
                        "averaging_start_s": 10,
                        "averaging_end_s": 20,
                        "notes": "not research data",
                    },
                    {
                        "experiment_id": "software_fixture",
                        "sensor_id": "bed_dp",
                        "quantity": "differential_pressure",
                        "value": 9.0,
                        "unit": "Pa",
                        "standard_uncertainty": 1.0,
                        "averaging_start_s": 10,
                        "averaging_end_s": 20,
                        "notes": "not research data",
                    },
                    {
                        "experiment_id": "software_fixture",
                        "sensor_id": "explicit_tc",
                        "quantity": "temperature",
                        "value": 698.0,
                        "unit": "K",
                        "standard_uncertainty": 2.0,
                        "averaging_start_s": 10,
                        "averaging_end_s": 20,
                        "notes": "not research data",
                    },
                ],
            )
            write_rows(
                data / "transient_measurements.csv",
                [
                    "experiment_id",
                    "time_s",
                    "sensor_id",
                    "quantity",
                    "value",
                    "unit",
                    "standard_uncertainty",
                    "notes",
                ],
                [],
            )
            write_rows(
                data / "calibration_records.csv",
                [
                    "calibration_id",
                    "instrument_type",
                    "calibration_date",
                    "reference_standard",
                    "source_reference",
                    "notes",
                ],
                [
                    {
                        "calibration_id": "cal_test",
                        "instrument_type": "software fixture",
                        "calibration_date": "2000-01-01",
                        "reference_standard": "software fixture",
                        "source_reference": "unit-test fixture",
                        "notes": "not research data",
                    }
                ],
            )

            topology = root / "topology.npz"
            np.savez_compressed(
                topology,
                level_5_centroid_m=np.asarray(
                    [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
                ),
            )
            state_file = root / "state.npz"
            state = np.zeros((1, 4, 5), dtype=float)
            state[0, 0, 3:] = [110.0, 500.0]
            state[0, 1, 3:] = [100.0, 600.0]
            state[0, 2, 4] = 700.0
            state[0, 3, 4] = 800.0
            np.savez_compressed(
                state_file,
                condition_id=np.asarray(["case_a"]),
                condition_physical=np.asarray([[0.1, 500.0, 1.0e6, 100.0, 400.0]]),
                state_physical=state,
                node_type=np.asarray([0, 0, 1, 1], dtype=np.int8),
                node_volume_m3=np.ones(4),
                fluid_global_region=np.asarray([0, 1]),
                solid_global_region=np.asarray([2, 3]),
            )
            mass_file = root / "mass.npz"
            np.savez_compressed(
                mass_file,
                condition_id=np.asarray(["case_a"]),
                fluid_global_region=np.asarray([0, 1]),
                boundary_owner=np.asarray([0, 1]),
                boundary_patch=np.asarray([0, 1]),
                boundary_face_area_m2=np.asarray([1.0, 1.0]),
                boundary_mass_flow_kg_s=np.asarray([[-0.2, 0.2]]),
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--data-root",
                    str(data),
                    "--regional-topology",
                    str(topology),
                    "--state-file",
                    str(state_file),
                    "--mass-targets",
                    str(mass_file),
                    "--model-name",
                    "software fixture model",
                    "--output-dir",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["compared_count"], 2)
            self.assertEqual(summary["nearest_phase_approximation_count"], 1)
            self.assertEqual(
                summary["status_counts"]["sensor_response_model_not_implemented"], 1
            )
            with (output / "model_experiment_comparison.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                rows = {row["sensor_id"]: row for row in csv.DictReader(stream)}
            self.assertAlmostEqual(float(rows["solid_tc"]["predicted_value"]), 700.0)
            self.assertAlmostEqual(
                float(rows["solid_tc"]["sensor_to_model_distance_m"]), 0.02
            )
            self.assertAlmostEqual(float(rows["bed_dp"]["predicted_value"]), 10.0)
            self.assertAlmostEqual(
                float(rows["bed_dp"]["residual_over_standard_uncertainty"]), 1.0
            )
            self.assertEqual(
                rows["explicit_tc"]["status"],
                "sensor_response_model_not_implemented",
            )
            self.assertEqual(rows["explicit_tc"]["predicted_value"], "")

    def test_empty_templates_do_not_require_model_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--data-root",
                    str(ROOT / "experimental_data_templates"),
                    "--regional-topology",
                    str(Path(directory) / "not_used.npz"),
                    "--state-file",
                    str(Path(directory) / "not_used_state.npz"),
                    "--output-dir",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "no_experimental_measurements")
            self.assertEqual(summary["measurement_count"], 0)


if __name__ == "__main__":
    unittest.main()
