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
SCRIPT = ROOT / "code/quantify_hccb_p418_regional_representation_fidelity.py"


class RegionalRepresentationFidelityTest(unittest.TestCase):
    def test_known_volume_average_and_hotspot_loss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fields = root / "fields"
            fields.mkdir()
            np.savez_compressed(
                root / "shared.npz",
                fluid_cell_volume_m3=np.asarray([1.0, 3.0]),
                solid_cell_volume_m3=np.asarray([1.0, 1.0]),
                fluid_cell_centroid_m=np.asarray([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]]),
                solid_cell_centroid_m=np.asarray([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]]),
            )
            np.savez_compressed(
                fields / "case.npz",
                fluid_temperature_K=np.asarray([300.0, 340.0]),
                solid_temperature_K=np.asarray([400.0, 500.0]),
            )
            dataset = {
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
            (root / "dataset.json").write_text(json.dumps(dataset), encoding="utf-8")
            np.savez_compressed(
                root / "geometry.npz",
                fine_to_regional_global=np.asarray([0, 0, 1, 1]),
                fluid_global_region=np.asarray([0]),
                solid_global_region=np.asarray([1]),
                fluid_cell_centroid_m=np.asarray([[0.00075, 0.0, 0.0]]),
                fluid_cell_volume_m3=np.asarray([4.0]),
                solid_cell_centroid_m=np.asarray([[0.0005, 0.0, 0.0]]),
                solid_cell_volume_m3=np.asarray([2.0]),
            )
            state = np.zeros((1, 2, 5), dtype=np.float64)
            state[0, 0, 4] = 330.0
            state[0, 1, 4] = 450.0
            np.savez_compressed(
                root / "state.npz",
                condition_id=np.asarray(["case"]),
                state_physical=state,
            )
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
                    str(root / "state.npz"),
                    "--output-dir",
                    str(output),
                ],
                check=True,
            )
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["counts"]["cases"], 1)
            self.assertAlmostEqual(
                summary["metrics"]["fluid_volume_weighted_rmse_K"]["maximum"],
                np.sqrt((30.0**2 + 3.0 * 10.0**2) / 4.0),
            )
            self.assertAlmostEqual(
                summary["metrics"]["solid_hotspot_temperature_loss_K"]["maximum"],
                50.0,
            )
            self.assertAlmostEqual(
                summary["metrics"]["solid_hotspot_region_centroid_distance_dp"]["maximum"],
                0.5,
            )
            self.assertEqual(
                summary["metrics"]["solid_hotspot_nearest_cell_distance_dp"]["maximum"],
                0.0,
            )
            self.assertEqual(summary["new_physical_parameters"], [])
            self.assertEqual(
                summary["global_volume_mean_temperature_error_K"]["fluid"][
                    "maximum_absolute"
                ],
                0.0,
            )
            self.assertEqual(
                summary["global_volume_mean_temperature_error_K"]["solid"][
                    "maximum_absolute"
                ],
                0.0,
            )
            self.assertTrue((output / "区域图温度保真度_CN.md").is_file())

    def test_stale_regional_state_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fields = root / "fields"
            fields.mkdir()
            np.savez_compressed(
                root / "shared.npz",
                fluid_cell_volume_m3=np.asarray([1.0]),
                solid_cell_volume_m3=np.asarray([1.0]),
                fluid_cell_centroid_m=np.zeros((1, 3)),
                solid_cell_centroid_m=np.zeros((1, 3)),
            )
            np.savez_compressed(
                fields / "case.npz",
                fluid_temperature_K=np.asarray([300.0]),
                solid_temperature_K=np.asarray([400.0]),
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
                fine_to_regional_global=np.asarray([0, 1]),
                fluid_global_region=np.asarray([0]),
                solid_global_region=np.asarray([1]),
                fluid_cell_centroid_m=np.zeros((1, 3)),
                fluid_cell_volume_m3=np.ones(1),
                solid_cell_centroid_m=np.zeros((1, 3)),
                solid_cell_volume_m3=np.ones(1),
            )
            state = np.zeros((1, 2, 5), dtype=np.float64)
            state[0, 0, 4] = 301.0
            state[0, 1, 4] = 400.0
            np.savez_compressed(
                root / "state.npz",
                condition_id=np.asarray(["case"]),
                state_physical=state,
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--dataset-index",
                    str(root / "dataset.json"),
                    "--subface-geometry",
                    str(root / "geometry.npz"),
                    "--regional-state-targets",
                    str(root / "state.npz"),
                    "--output-dir",
                    str(root / "output"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("direct volume averaging", result.stderr)


if __name__ == "__main__":
    unittest.main()
