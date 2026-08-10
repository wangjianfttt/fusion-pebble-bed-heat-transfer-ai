#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/train_hccb_p418_regional_dmdc.py"
import sys
sys.path.insert(0, str(ROOT / "code"))
from train_hccb_p418_regional_dmdc import field_metrics  # noqa: E402


class P418RegionalDMDcTest(unittest.TestCase):
    def test_temperature_metric_uses_regional_volume(self) -> None:
        truth = np.zeros((1, 2, 4), dtype=np.float64)
        prediction = np.asarray([[[1.0, 3.0, 2.0, 4.0], [1.0, 3.0, 2.0, 4.0]]])
        node_type = np.asarray([0, 0, 1, 1])
        volume = np.asarray([1.0, 3.0, 2.0, 6.0])
        centroid = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        )
        metrics = field_metrics(prediction, truth, node_type, volume, centroid)
        self.assertAlmostEqual(metrics["fluid_temperature_RMSE_K"], np.sqrt(7.0))
        self.assertAlmostEqual(metrics["solid_temperature_RMSE_K"], np.sqrt(13.0))

    def test_controlled_linear_temperature_curves_run_without_test_fitting(self) -> None:
        split_source = json.loads(
            (ROOT / "parameters/hccb_p418_step_response_splits.json").read_text(encoding="utf-8")
        )
        split = split_source["splits"]["direction_down_test"]
        sequence_ids = split["train"] + split["validation"] + split["test"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequence_dir = root / "sequences"
            sequence_dir.mkdir()
            np.savez_compressed(
                root / "geometry.npz",
                node_type=np.asarray([0, 0, 1, 1], dtype=np.int8),
                node_volume_m3=np.asarray([1.0, 2.0, 1.5, 2.5]),
                node_centroid_m=np.asarray(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
                ),
            )
            records = []
            time_s = np.asarray([0.0, 0.01, 0.5, 5.0, 25.0], dtype=np.float64)
            spatial_forcing = np.asarray([0.2, 0.4, 0.6, 0.8])
            for index, sequence_id in enumerate(sequence_ids):
                initial = 300.0 + index + np.arange(4)
                temperature = initial[None, :] + time_s[:, None] * spatial_forcing[None, :]
                state = np.zeros((5, 4, 5), dtype=np.float64)
                state[..., 4] = temperature
                path = sequence_dir / f"{sequence_id}.npz"
                np.savez_compressed(
                    path,
                    sequence_id=np.asarray(sequence_id),
                    time_s=time_s,
                    condition_physical=np.ones(8, dtype=np.float64),
                    state_physical=state,
                    fluid_internal_mass_flux_kg_s=np.asarray([1.0e-5]),
                    fluid_boundary_mass_flux_kg_s=np.asarray([-1.0e-5, 1.0e-5]),
                )
                records.append(
                    {"sequence_id": sequence_id, "sequence_file": f"sequences/{sequence_id}.npz", "complete": True}
                )
            (root / "dataset_index.json").write_text(
                json.dumps(
                    {
                        "sequence_count": 12,
                        "regional_geometry_file": "geometry.npz",
                        "sequences": records,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "result"
            subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--dataset-index",
                    str(root / "dataset_index.json"),
                    "--splits",
                    str(ROOT / "parameters/hccb_p418_step_response_splits.json"),
                    "--split-name",
                    "direction_down_test",
                    "--rank-candidates",
                    "1,2,4",
                    "--output-dir",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "completed_p418_regional_dmdc")
            self.assertEqual(summary["split_case_counts"], {"train": 6, "validation": 3, "test": 3})
            self.assertEqual(summary["new_physical_parameters"], [])
            self.assertEqual(
                summary["dynamics_form"], "continuous_time_midpoint_derivative"
            )
            self.assertGreater(summary["time_step_unique_count"], 1)
            self.assertEqual(summary["time_step_range_s"], [0.01, 20.0])
            self.assertEqual(
                summary["dataset_index"], str((root / "dataset_index.json").resolve())
            )
            self.assertLess(summary["metrics"]["test"]["solid_temperature_RMSE_K"], 1.0e-6)
            self.assertLess(
                summary["metrics"]["test"]["solid_maximum_temperature_history_RMSE_K"],
                1.0e-6,
            )
            self.assertEqual(
                summary["metrics"]["test"]["solid_regional_hotspot_location_mean_error_m"],
                0.0,
            )
            with np.load(output / summary["prediction_files"]["test"], allow_pickle=False) as data:
                np.testing.assert_allclose(
                    data["temperature_prediction_K"][:, 0],
                    data["temperature_target_K"][:, 0],
                )
                self.assertIn("fixed_hydrodynamics_physical", data.files)
                self.assertIn("fluid_internal_mass_flux_kg_s", data.files)
                self.assertIn("fluid_boundary_mass_flux_kg_s", data.files)
                self.assertIn("node_centroid_m", data.files)


if __name__ == "__main__":
    unittest.main()
