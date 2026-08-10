#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from validate_hccb_p418_steady_dataset_ready import validate  # noqa: E402


class P418SteadyDatasetReadyTest(unittest.TestCase):
    def files(self, root: Path, count: int) -> tuple[Path, Path, Path]:
        summary = root / "summary.json"
        dataset = root / "dataset_index.json"
        topology = root / "shared_mesh_topology.npz"
        summary.write_text(
            json.dumps(
                {
                    "status": "p418_60_training_data_ready",
                    "expected_case_count": count,
                }
            ),
            encoding="utf-8",
        )
        dataset.write_text(
            json.dumps(
                {
                    "case_count": count,
                    "sourceflow_mapping_required": True,
                    "steady_final_window_required": True,
                    "conditions": [
                        {
                            "pore_opening_boundary_velocity_m_s": 0.125,
                            "inlet_open_area_fraction": 0.4,
                        }
                        for _ in range(count)
                    ],
                }
            ),
            encoding="utf-8",
        )
        topology.write_bytes(b"topology")
        return summary, dataset, topology

    def test_sixty_case_data_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.files(Path(directory), 60)
            result = validate(*paths)
            self.assertEqual(result["case_count"], 60)
            self.assertEqual(result["new_physical_parameters"], [])

    def test_dataset_without_sourceflow_mapping_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary, dataset, topology = self.files(Path(directory), 60)
            payload = json.loads(dataset.read_text(encoding="utf-8"))
            payload["sourceflow_mapping_required"] = False
            dataset.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "corrected source-flow mapping"):
                validate(summary, dataset, topology)

    def test_five_case_pilot_data_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.files(Path(directory), 5)
            with self.assertRaisesRegex(ValueError, "required 60"):
                validate(*paths)

    def test_missing_topology_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary, dataset, topology = self.files(Path(directory), 60)
            topology.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "shared mesh topology"):
                validate(summary, dataset, topology)


if __name__ == "__main__":
    unittest.main()
