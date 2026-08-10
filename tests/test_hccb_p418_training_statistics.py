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
SCRIPT = ROOT / "code/build_hccb_p418_training_statistics.py"


class P418TrainingStatisticsTest(unittest.TestCase):
    def test_statistics_use_only_declared_training_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            fields = base / "fields"
            fields.mkdir()
            np.savez_compressed(
                base / "topology.npz",
                fluid_cell_volume_m3=np.array([1.0, 3.0]),
                solid_cell_volume_m3=np.array([2.0]),
                fluid_boundary_face_area_m2=np.array([1.0, 1.0]),
                solid_boundary_face_area_m2=np.array([1.0]),
                fluid_boundary_temperature_value_mask=np.array([True, True]),
                solid_boundary_temperature_value_mask=np.array([True]),
            )
            records = []
            for index, condition in enumerate(("a", "b", "c")):
                np.savez_compressed(
                    fields / f"{condition}.npz",
                    fluid_velocity_m_s=np.full((2, 3), index + 1.0),
                    fluid_pressure_Pa=np.array([120001.0, 120003.0]) + index,
                    fluid_temperature_K=np.array([300.0, 500.0]) + 100 * index,
                    solid_temperature_K=np.array([400.0 + 100 * index]),
                    fluid_internal_face_mass_flow_kg_s=np.array([1.0 + index]),
                    fluid_boundary_temperature_K=np.array(
                        [300.0 + 100 * index, 500.0 + 100 * index]
                    ),
                    solid_boundary_temperature_K=np.array([400.0 + 100 * index]),
                    fluid_boundary_face_mass_flow_kg_s=np.array(
                        [-(1.0 + index), 1.0 + index]
                    ),
                )
                records.append(
                    {
                        "condition_id": condition,
                        "inlet_velocity_m_s": 0.05 + 0.05 * index,
                        "inlet_temperature_K": 300.0 + 200.0 * index,
                        "solid_heat_source_W_m3": 4.85e6 + 2.0e6 * index,
                        "outlet_pressure_Pa": 120000.0,
                        "cooling_wall_temperature_K": 635.0,
                        "field_file": f"fields/{condition}.npz",
                    }
                )
            (base / "dataset.json").write_text(
                json.dumps(
                    {
                        "shared_topology_file": "topology.npz",
                        "conditions": records,
                    }
                )
            )
            (base / "splits.json").write_text(
                json.dumps(
                    {
                        "source_doi": "doi:test",
                        "conditions": [{"condition_id": item} for item in ("a", "b", "c")],
                        "splits": {
                            "example": {
                                "train": ["a", "b"],
                                "validation": [],
                                "test": ["c"],
                                "question": "test",
                            }
                        },
                    }
                )
            )
            output = base / "stats.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--dataset-index",
                    str(base / "dataset.json"),
                    "--split-file",
                    str(base / "splits.json"),
                    "--output",
                    str(output),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            result = json.loads(output.read_text())
            example = result["splits"]["example"]
            self.assertEqual(example["train_conditions"], ["a", "b"])
            self.assertNotIn("c", example["train_conditions"])
            self.assertAlmostEqual(
                example["targets"]["fluid_velocity_m_s"]["mean"][0], 1.5
            )
            self.assertEqual(
                example["condition_input"]["constant_training_inputs"],
                ["outlet_pressure_Pa", "cooling_wall_temperature_K"],
            )
            self.assertGreater(
                example["targets"]["fluid_boundary_face_mass_flow_kg_s"][
                    "root_mean_square"
                ],
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
