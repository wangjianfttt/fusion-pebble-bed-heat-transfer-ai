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
SCRIPT = ROOT / "code/validate_hccb_p418_experimental_data.py"


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class P418ExperimentalDataTest(unittest.TestCase):
    def test_empty_templates_contain_no_invented_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--schema",
                    str(ROOT / "parameters/hccb_p418_experimental_data_schema.json"),
                    "--data-root",
                    str(ROOT / "experimental_data_templates"),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "empty_experimental_templates_ready")
            self.assertEqual(payload["experiment_count"], 0)
            self.assertEqual(payload["steady_measurement_count"], 0)
            self.assertEqual(payload["transient_measurement_count"], 0)
            self.assertEqual(payload["new_physical_parameters"], [])

    def test_model_observable_requires_compatible_sensor_response_model(self) -> None:
        schema = json.loads(
            (ROOT / "parameters/hccb_p418_experimental_data_schema.json").read_text(
                encoding="utf-8"
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            data.mkdir()
            for filename, definition in schema["tables"].items():
                write_rows(data / filename, definition["columns"], [])
            write_rows(
                data / "experiment_conditions.csv",
                schema["tables"]["experiment_conditions.csv"]["columns"],
                [
                    {
                        "experiment_id": "fixture",
                        "model_condition_id": "case_a",
                        "material": "test",
                        "packing_id": "test",
                        "run_type": "software_test",
                        "source_reference": "unit-test fixture",
                        "geometry_reference": "unit-test fixture",
                        "notes": "not research data",
                    }
                ],
            )
            write_rows(
                data / "calibration_records.csv",
                schema["tables"]["calibration_records.csv"]["columns"],
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
            base_sensor = {
                "experiment_id": "fixture",
                "sensor_id": "tc",
                "quantity": "temperature",
                "model_observable": "solid_temperature",
                "sensor_response_model": "",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "measurement_method": "software fixture",
                "calibration_id": "cal_test",
                "notes": "not research data",
            }
            cases = [
                ("", "requires sensor_response_model"),
                (
                    "direct_integral_or_boundary_quantity",
                    "is not allowed for 'solid_temperature'",
                ),
            ]
            for response_model, expected in cases:
                with self.subTest(response_model=response_model):
                    sensor = dict(base_sensor)
                    sensor["sensor_response_model"] = response_model
                    write_rows(
                        data / "sensor_layout.csv",
                        schema["tables"]["sensor_layout.csv"]["columns"],
                        [sensor],
                    )
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(SCRIPT),
                            "--schema",
                            str(
                                ROOT
                                / "parameters/hccb_p418_experimental_data_schema.json"
                            ),
                            "--data-root",
                            str(data),
                            "--output",
                            str(Path(directory) / "summary.json"),
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(expected, result.stderr)


if __name__ == "__main__":
    unittest.main()
