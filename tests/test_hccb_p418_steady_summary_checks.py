#!/usr/bin/env python3

from __future__ import annotations

import json
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

from hccb_p418_comparison_contract import STEADY_METRIC_CONTRACT  # noqa: E402


def fake_summary(architecture: str, fingerprint: str) -> dict:
    payload = {
        "architecture": architecture,
        "split_name": "interleaved_all_ranges",
        "split_case_ids": {"train": ["a", "b"], "validation": ["c"], "test": ["d"]},
        "comparison_requested_epochs": 100 if architecture == "response_surface" else None,
        "epochs": 0 if architecture == "response_surface" else 100,
        "best_epoch": 0 if architecture == "response_surface" else 80,
        "training_seconds": 1.0,
        "model_parameter_count": 10,
        "effective_batch_size": 1,
        "microbatch_size": 1,
        "gradient_accumulation": False,
        "peak_gpu_memory_GB": 0.25,
        "training_seed": 0,
        "optimizer_name": "Adam",
        "initial_model_state_sha256": "1" * 64,
        "normalization": {"same": 1.0},
        "metric_contract": STEADY_METRIC_CONTRACT,
        "run_provenance": {"common_comparison_fingerprint": fingerprint},
        "evaluations": {
            "test": {
                "metrics": {
                    "state_normalized_rmse": 0.2,
                    "state_channel_rmse": [0.1, 0.1, 0.1, 0.1, 0.2, 0.3],
                    "continuity_normalized_rmse": 0.01,
                    "energy_balance_normalized_rmse": 0.02,
                },
                "cases": [
                    {
                        "condition_id": "d",
                        "generated_power_W": 100.0,
                        "engineering_absolute_errors": {
                            "pressure_drop_Pa": 1.0,
                            "outlet_temperature_K": 2.0,
                            "solid_maximum_temperature_K": 3.0,
                            "cooling_wall_heat_into_fluid_W": 4.0,
                            "solid_to_fluid_interphase_net_W": 5.0,
                            "fluid_solid_interphase_absolute_flow_W": 6.0,
                        },
                        "local_mass_l1_over_two_inlet": 0.1,
                        "global_mass_imbalance_over_inlet": 0.01,
                        "local_energy_l1_over_two_generated_power": 0.2,
                        "global_energy_imbalance_over_generated_power": 0.02,
                    }
                ],
                "inference_seconds_per_case": 0.01,
            }
        },
    }
    if architecture in {"pinn_data_only", "pinn"}:
        constrained = architecture == "pinn"
        payload.update(
            {
                "field_architecture": "pinn",
                "physics_constraints_in_training": constrained,
                "settings_from_archived_source": {"same": True},
                "loss_group_weights": {
                    "state_data": 5.0,
                    "face_flux": 1.0,
                    "physics_balance": 1.0,
                },
                "active_loss_groups": [
                    "state_data",
                    "face_flux",
                    *(["physics_balance"] if constrained else []),
                ],
                "loss_terms": [
                    "state",
                    "internal_mass",
                    "boundary_mass",
                    *(["continuity"] if constrained else []),
                    "internal_energy",
                    "boundary_energy",
                    *(["energy_balance"] if constrained else []),
                ],
                "effective_loss_term_weights": {
                    "state": 5.0,
                    "internal_mass": 0.25,
                    "boundary_mass": 0.25,
                    "internal_energy": 0.25,
                    "boundary_energy": 0.25,
                    **(
                        {"continuity": 0.5, "energy_balance": 0.5}
                        if constrained
                        else {}
                    ),
                },
            }
        )
    return payload


class P418SteadySummaryChecksTest(unittest.TestCase):
    def run_summary(
        self,
        root: Path,
        architectures: tuple[str, ...] = ("response_surface", "pinn"),
        *,
        result_prefix: str = "hccb_p418_60",
        split_name: str = "interleaved_all_ranges",
    ) -> subprocess.CompletedProcess[str]:
        split = root / "splits.json"
        split.write_text(
            json.dumps(
                {
                    "splits": {
                        split_name: {
                            "train": ["a", "b"],
                            "validation": ["c"],
                            "test": ["d"],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        return subprocess.run(
            [
                "python3",
                str(CODE / "summarize_hccb_p418_60_model_comparison.py"),
                "--results-root",
                str(root),
                "--epochs",
                "100",
                "--result-prefix",
                result_prefix,
                "--architectures",
                *architectures,
                "--splits",
                split_name,
                "--split-file",
                str(split),
                "--output-dir",
                str(root / "comparison"),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def make_result(
        self,
        root: Path,
        architecture: str,
        fingerprint: str,
        *,
        result_prefix: str = "hccb_p418_60",
        split_name: str = "interleaved_all_ranges",
    ) -> Path:
        directory = root / f"{result_prefix}_{architecture}_{split_name}_100epoch"
        directory.mkdir(parents=True)
        summary = directory / "summary.json"
        payload = fake_summary(architecture, fingerprint)
        payload["split_name"] = split_name
        summary.write_text(json.dumps(payload), encoding="utf-8")
        return summary

    def test_summary_accepts_custom_result_prefix_and_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = "hccb_p418_mixed_endpoint_smoke_model"
            split = "completed_smoke"
            self.make_result(
                root,
                "response_surface",
                "same",
                result_prefix=prefix,
                split_name=split,
            )
            self.make_result(
                root,
                "pinn",
                "same",
                result_prefix=prefix,
                split_name=split,
            )
            completed = self.run_summary(
                root,
                result_prefix=prefix,
                split_name=split,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads((root / "comparison/summary.json").read_text())
            self.assertEqual(payload["result_prefix"], prefix)
            rows = list(
                csv.DictReader(
                    (root / "comparison/model_comparison.csv").open(
                        encoding="utf-8"
                    )
                )
            )
            self.assertEqual(rows[0]["microbatch_size"], "1")
            self.assertEqual(rows[0]["gradient_accumulation"], "False")
            self.assertEqual(rows[0]["peak_gpu_memory_GB"], "0.25")

    def test_summary_accepts_identical_comparison_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_result(root, "response_surface", "same")
            self.make_result(root, "pinn", "same")
            completed = self.run_summary(root)
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_summary_rejects_changed_training_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_result(root, "response_surface", "first")
            self.make_result(root, "pinn", "changed")
            completed = self.run_summary(root)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("different fields", completed.stderr)

    def test_summary_rejects_changed_test_condition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_result(root, "response_surface", "same")
            changed = self.make_result(root, "pinn", "same")
            payload = json.loads(changed.read_text(encoding="utf-8"))
            payload["split_case_ids"]["test"] = ["c"]
            changed.write_text(json.dumps(payload), encoding="utf-8")
            completed = self.run_summary(root)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("test conditions differ", completed.stderr)

    def test_summary_checks_paired_pinn_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_result(root, "pinn_data_only", "same")
            constrained = self.make_result(root, "pinn", "same")
            completed = self.run_summary(root, ("pinn_data_only", "pinn"))
            self.assertEqual(completed.returncode, 0, completed.stderr)

            payload = json.loads(constrained.read_text(encoding="utf-8"))
            payload["model_parameter_count"] = 11
            constrained.write_text(json.dumps(payload), encoding="utf-8")
            changed = self.run_summary(root, ("pinn_data_only", "pinn"))
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn("different model_parameter_count", changed.stderr)

            payload = json.loads(constrained.read_text(encoding="utf-8"))
            payload["model_parameter_count"] = 10
            payload["initial_model_state_sha256"] = "2" * 64
            constrained.write_text(json.dumps(payload), encoding="utf-8")
            changed = self.run_summary(root, ("pinn_data_only", "pinn"))
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn("different initial_model_state_sha256", changed.stderr)

    def test_summary_rejects_changed_shared_supervised_weight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_result(root, "pinn_data_only", "same")
            constrained = self.make_result(root, "pinn", "same")
            payload = json.loads(constrained.read_text(encoding="utf-8"))
            payload["effective_loss_term_weights"]["state"] = 4.0
            constrained.write_text(json.dumps(payload), encoding="utf-8")
            completed = self.run_summary(root, ("pinn_data_only", "pinn"))
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("change shared supervised loss weights", completed.stderr)


if __name__ == "__main__":
    unittest.main()
