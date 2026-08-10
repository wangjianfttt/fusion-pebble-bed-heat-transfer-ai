#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "code/train_hccb_p418_transient_observable_transformer.py"


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is tested on the remote compute machine")
class P418PhysicalStepTransformerTest(unittest.TestCase):
    def test_formal_split_requires_all_complete_curves_and_no_role_overlap(self) -> None:
        import sys

        sys.path.insert(0, str(ROOT / "code"))
        from train_hccb_p418_transient_observable_transformer import split_indices

        with tempfile.TemporaryDirectory() as tmp:
            split_path = Path(tmp) / "splits.json"
            split_path.write_text(
                json.dumps(
                    {
                        "splits": {
                            "formal": {
                                "train": ["a"],
                                "validation": ["b"],
                                "test": ["c"],
                            },
                            "overlap": {
                                "train": ["a"],
                                "validation": ["a"],
                                "test": ["b"],
                            },
                        }
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "data and split differ"):
                split_indices(["a", "b"], split_path, "formal", require_complete=True)
            self.assertEqual(
                split_indices(["a", "b", "c"], split_path, "formal", require_complete=True),
                {"train": [0], "validation": [1], "test": [2]},
            )
            with self.assertRaisesRegex(ValueError, "overlap"):
                split_indices(["a", "b"], split_path, "overlap", require_complete=True)

    def test_complete_curve_step_pipeline_runs(self) -> None:
        plan = json.loads((ROOT / "parameters/hccb_p418_transient_step_plan.json").read_text())
        case_ids = np.asarray([row["sequence_id"] for row in plan["sequences"]], dtype=object)
        count = len(case_ids)
        time_points = 5
        condition_names = np.asarray(
            [
                "source_inlet_velocity_m_s",
                "source_inlet_temperature_K",
                "source_solid_heat_source_MW_m3",
                "target_inlet_velocity_m_s",
                "target_inlet_temperature_K",
                "target_solid_heat_source_MW_m3",
                "outlet_pressure_Pa",
                "cooling_wall_temperature_K",
            ],
            dtype=object,
        )
        signal_names = np.asarray(
            [
                "inlet_temperature_K",
                "outlet_temperature_K",
                "inlet_pressure_Pa",
                "outlet_pressure_Pa",
                "inlet_mass_flow_kg_s",
                "outlet_mass_flow_kg_s",
                "inlet_enthalpy_flow_W",
                "outlet_enthalpy_flow_W",
                "cooling_wall_power_W",
                "maximum_solid_temperature_K",
                "pressure_drop_Pa",
                "signed_mass_residual_kg_s",
                "net_outward_enthalpy_flow_W",
            ],
            dtype=object,
        )
        # These arrays only exercise software wiring. They are not saved under results
        # and are not interpreted as physical or scientific data.
        conditions = np.arange(count * len(condition_names), dtype=np.float64).reshape(count, -1)
        time_s = np.broadcast_to(np.arange(time_points, dtype=np.float64), (count, time_points)).copy()
        values = np.empty((count, time_points, len(signal_names)), dtype=np.float64)
        for i in range(count):
            for j in range(len(signal_names)):
                values[i, :, j] = (i + 1) * (j + 1) + np.arange(time_points)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "software_wiring_only.npz"
            np.savez_compressed(
                data_path,
                case_id=case_ids,
                complete=np.ones(count, dtype=bool),
                conditions=conditions,
                condition_names=condition_names,
                time_s=time_s,
                time_mask=np.ones((count, time_points), dtype=bool),
                values=values,
                signal_names=signal_names,
            )
            output = tmp_path / "out"
            subprocess.run(
                [
                    "python3",
                    str(TRAINER),
                    "--data",
                    str(data_path),
                    "--splits",
                    str(ROOT / "parameters/hccb_p418_step_response_splits.json"),
                    "--split-name",
                    "direction_down_test",
                    "--output-dir",
                    str(output),
                    "--run-role",
                    "smoke",
                    "--history-kind",
                    "physical_step_response",
                    "--epochs",
                    "2",
                    "--d-model",
                    "8",
                    "--heads",
                    "2",
                    "--layers",
                    "1",
                    "--batch-size",
                    "6",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["history_kind"], "physical_step_response")
            self.assertEqual(summary["available_complete_curves"], 12)
            self.assertEqual(summary["split_case_counts"], {"train": 6, "validation": 3, "test": 3})
            self.assertIn("source_inlet_temperature_K", summary["condition_names"])
            self.assertEqual(summary["selection_split"], "validation")
            self.assertIn(summary["selected_epoch"], (1, 2))
            self.assertTrue(np.isfinite(summary["best_validation_normalized_mse"]))
            self.assertTrue(summary["time_points_are_never_split_across_roles"])

    def test_formal_run_rejects_unregistered_model_size(self) -> None:
        completed = subprocess.run(
            [
                "python3",
                str(TRAINER),
                "--output-dir",
                "/tmp/not_written_formal_transformer",
                "--run-role",
                "formal",
                "--d-model",
                "8",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("must match the registered", completed.stderr + completed.stdout)


if __name__ == "__main__":
    unittest.main()
