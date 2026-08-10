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
SCRIPT = ROOT / "code/evaluate_hccb_p418_persistence_baseline.py"
SPLIT = "pair_disjoint_stress_test"


class P418PersistenceBaselineTest(unittest.TestCase):
    def test_repeats_initial_field_without_fitting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence_ids = [f"curve-{index:02d}" for index in range(12)]
            records = []
            for index, sequence_id in enumerate(sequence_ids):
                time_s = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
                state = np.zeros((3, 2, 5), dtype=np.float32)
                state[..., :4] = float(index)
                state[:, 0, 4] = np.asarray([300.0, 301.0, 302.0])
                state[:, 1, 4] = np.asarray([500.0, 503.0, 506.0])
                path = root / f"{sequence_id}.npz"
                np.savez_compressed(
                    path,
                    time_s=time_s,
                    condition_physical=np.full(8, float(index), dtype=np.float32),
                    state_physical=state,
                    fluid_internal_mass_flux_kg_s=np.zeros((3, 1)),
                    fluid_boundary_mass_flux_kg_s=np.zeros((3, 1)),
                )
                records.append(
                    {
                        "sequence_id": sequence_id,
                        "sequence_file": path.name,
                        "complete": True,
                    }
                )
            geometry = root / "geometry.npz"
            np.savez_compressed(
                geometry,
                node_type=np.asarray([0, 1], dtype=np.int8),
                node_volume_m3=np.asarray([1.0, 2.0]),
                node_centroid_m=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            )
            index_path = root / "dataset_index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "sequence_count": 12,
                        "sequences": records,
                        "regional_geometry_file": geometry.name,
                    }
                ),
                encoding="utf-8",
            )
            split_path = root / "splits.json"
            split_path.write_text(
                json.dumps(
                    {
                        "splits": {
                            SPLIT: {
                                "train": sequence_ids[:6],
                                "validation": sequence_ids[6:8],
                                "test": sequence_ids[8:],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            output = root / "output"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--dataset-index",
                    str(index_path),
                    "--splits",
                    str(split_path),
                    "--split-name",
                    SPLIT,
                    "--output-dir",
                    str(output),
                ],
                check=True,
            )
            summary = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["model_parameter_count"], 0)
            self.assertEqual(summary["selection_split"], "not_applicable")
            self.assertEqual(summary["split_case_counts"]["test"], 4)
            self.assertAlmostEqual(
                summary["metrics"]["test"]["fluid_temperature_RMSE_K"],
                np.sqrt(5.0 / 3.0),
                places=6,
            )
            self.assertAlmostEqual(
                summary["metrics"]["test"]["solid_temperature_RMSE_K"],
                np.sqrt(15.0),
                places=6,
            )
            with np.load(output / "test_temperature_predictions.npz") as data:
                prediction = data["temperature_prediction_K"]
                target = data["temperature_target_K"]
                self.assertTrue(
                    np.array_equal(
                        prediction,
                        np.broadcast_to(target[:, :1], target.shape),
                    )
                )


if __name__ == "__main__":
    unittest.main()
