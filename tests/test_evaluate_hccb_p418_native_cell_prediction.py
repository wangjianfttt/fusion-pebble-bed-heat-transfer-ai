#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/evaluate_hccb_p418_native_cell_prediction.py"


class NativeCellPredictionTest(unittest.TestCase):
    def test_exact_squared_error_decomposition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fields").mkdir()
            np.savez_compressed(
                root / "shared.npz",
                fluid_cell_volume_m3=np.asarray([1.0, 3.0]),
                solid_cell_volume_m3=np.asarray([1.0, 1.0]),
                fluid_cell_centroid_m=np.asarray([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]]),
                solid_cell_centroid_m=np.asarray([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]]),
            )
            np.savez_compressed(
                root / "fields/case.npz",
                fluid_temperature_K=np.asarray([300.0, 340.0]),
                solid_temperature_K=np.asarray([400.0, 500.0]),
            )
            (root / "dataset.json").write_text(
                json.dumps(
                    {
                        "case_count": 1,
                        "shared_topology_file": "shared.npz",
                        "conditions": [
                            {
                                "condition_id": "case",
                                "inlet_velocity_m_s": 0.1,
                                "inlet_temperature_K": 300.0,
                                "solid_heat_source_W_m3": 1.0,
                                "field_file": "fields/case.npz",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            np.savez_compressed(
                root / "geometry.npz",
                fine_to_regional_global=np.asarray([0, 0, 1, 1]),
                fluid_global_region=np.asarray([0]),
                solid_global_region=np.asarray([1]),
                fluid_cell_centroid_m=np.asarray([[0.00075, 0.0, 0.0]]),
                solid_cell_centroid_m=np.asarray([[0.0005, 0.0, 0.0]]),
                fluid_internal_subface_owner=np.asarray([], dtype=np.int64),
                fluid_internal_subface_neighbour=np.asarray([], dtype=np.int64),
                solid_internal_subface_owner=np.asarray([], dtype=np.int64),
                solid_internal_subface_neighbour=np.asarray([], dtype=np.int64),
            )
            target = np.zeros((1, 2, 5), dtype=np.float64)
            target[0, 0, 4] = 330.0
            target[0, 1, 4] = 450.0
            np.savez_compressed(
                root / "target.npz",
                condition_id=np.asarray(["case"]),
                state_physical=target,
            )
            prediction = np.zeros((1, 2, 5), dtype=np.float64)
            prediction[0, 0, 4] = 1.0
            prediction[0, 1, 4] = 2.0
            np.savez_compressed(
                root / "prediction.npz",
                condition_id=np.asarray(["case"]),
                baseline_state_normalized=prediction,
                node_type=np.asarray([0, 1], dtype=np.int8),
                node_volume_m3=np.asarray([4.0, 2.0]),
            )
            statistics = {
                "splits": {
                    "example": {
                        "targets": {
                            "fluid_temperature_K": {"mean": [330.0], "standard_deviation": [10.0]},
                            "solid_temperature_K": {"mean": [450.0], "standard_deviation": [10.0]},
                        }
                    }
                }
            }
            (root / "stats.json").write_text(json.dumps(statistics), encoding="utf-8")
            output = root / "output"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--dataset-index",
                    str(root / "dataset.json"),
                    "--subface-geometry",
                    str(root / "geometry.npz"),
                    "--regional-state-targets",
                    str(root / "target.npz"),
                    "--regional-predictions",
                    str(root / "prediction.npz"),
                    "--training-statistics",
                    str(root / "stats.json"),
                    "--split-name",
                    "example",
                    "--output-dir",
                    str(output),
                ],
                check=True,
            )
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            metrics = summary["metrics"]
            self.assertAlmostEqual(metrics["fluid_regional_model_rmse_K"]["mean"], 10.0)
            self.assertAlmostEqual(metrics["solid_regional_model_rmse_K"]["mean"], 20.0)
            self.assertAlmostEqual(
                metrics["fluid_native_total_rmse_K"]["mean"] ** 2,
                metrics["fluid_representation_rmse_K"]["mean"] ** 2 + 100.0,
            )
            self.assertAlmostEqual(
                metrics["solid_native_total_rmse_K"]["mean"] ** 2,
                metrics["solid_representation_rmse_K"]["mean"] ** 2 + 400.0,
            )
            self.assertLess(
                metrics["fluid_squared_error_identity_difference_K2"]["maximum_absolute"],
                1.0e-10,
            )
            self.assertLess(
                metrics["solid_squared_error_identity_difference_K2"]["maximum_absolute"],
                1.0e-10,
            )
            self.assertAlmostEqual(
                metrics["fluid_limited_native_total_rmse_K"]["mean"],
                metrics["fluid_native_total_rmse_K"]["mean"],
            )
            self.assertAlmostEqual(
                metrics["solid_limited_native_total_rmse_K"]["mean"],
                metrics["solid_native_total_rmse_K"]["mean"],
            )
            self.assertEqual(summary["new_physical_parameters"], [])


if __name__ == "__main__":
    unittest.main()
