#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/summarize_hccb_p418_native_cell_predictions.py"
METRICS = (
    "fluid_representation_rmse_K",
    "fluid_regional_model_rmse_K",
    "fluid_native_total_rmse_K",
    "fluid_limited_native_total_rmse_K",
    "solid_representation_rmse_K",
    "solid_regional_model_rmse_K",
    "solid_native_total_rmse_K",
    "solid_limited_native_total_rmse_K",
    "predicted_solid_max_temperature_error_K",
    "predicted_hotspot_nearest_cell_distance_dp",
    "limited_predicted_solid_max_temperature_error_K",
    "limited_predicted_hotspot_distance_dp",
)


class NativeCellPredictionSummaryTest(unittest.TestCase):
    def test_two_models_are_collected_without_changing_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = []
            for index, model in enumerate(("pinn", "transolver"), start=1):
                path = root / f"{model}.json"
                metrics = {
                    metric: {
                        "mean": float(index),
                        "maximum_absolute": float(index + 1),
                        "maximum_absolute_condition_id": "case",
                    }
                    for metric in METRICS
                }
                path.write_text(
                    json.dumps(
                        {
                            "status": "native_cell_prediction_metrics_ready",
                            "split_name": "interleaved_all_ranges",
                            "case_count": 3,
                            "metrics": metrics,
                        }
                    ),
                    encoding="utf-8",
                )
                inputs.extend(("--result", f"{model}={path}"))
            output = root / "output"
            subprocess.run(
                ["python3", str(SCRIPT), *inputs, "--output-dir", str(output)],
                check=True,
                capture_output=True,
                text=True,
            )
            with (output / "native_cell_model_comparison.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["model"] for row in rows], ["pinn", "transolver"])
            self.assertEqual(float(rows[0]["solid_native_total_rmse_K_mean"]), 1.0)
            self.assertEqual(float(rows[1]["solid_native_total_rmse_K_mean"]), 2.0)
            payload = json.loads(
                (output / "native_cell_model_comparison.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["new_physical_parameters"], [])


if __name__ == "__main__":
    unittest.main()
