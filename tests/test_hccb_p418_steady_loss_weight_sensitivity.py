#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from summarize_hccb_p418_steady_loss_weight_sensitivity import (  # noqa: E402
    load_settings,
    summarize,
    write_outputs,
)


def fake_summary(weights: dict[str, float]) -> dict[str, object]:
    return {
        "architecture": "pinn",
        "split_name": "interleaved_all_ranges",
        "split_case_ids": {"train": ["a"], "validation": ["b"], "test": ["c"]},
        "epochs": 100,
        "model_parameter_count": 10,
        "optimizer_name": "Adam",
        "training_seed": 20260717,
        "initial_model_state_sha256": "1" * 64,
        "effective_batch_size": 1,
        "microbatch_size": 1,
        "updates_per_epoch": 1,
        "total_parameter_updates": 100,
        "field_architecture": {"hidden_dim": 50, "hidden_layers": 6},
        "settings_from_archived_source": {
            "hidden_dim": 50,
            "hidden_layers": 6,
            "optimizer": "Adam",
            "learning_rate": 0.01,
        },
        "normalization": {"source": "training conditions only"},
        "metric_contract": {"version": "test"},
        "best_epoch": 80,
        "loss_group_weights": weights,
        "run_provenance": {"common_comparison_fingerprint": "same"},
        "evaluations": {
            "test": {
                "metrics": {"state_normalized_rmse": 0.2},
                "cases": [
                    {
                        "engineering_absolute_errors": {
                            "outlet_temperature_K": 2.0,
                            "solid_maximum_temperature_K": 3.0,
                            "cooling_wall_heat_into_fluid_W": 4.0,
                        },
                        "generated_power_W": 100.0,
                        "global_mass_imbalance_over_inlet": 0.01,
                        "global_energy_imbalance_over_generated_power": 0.02,
                    }
                ],
            }
        },
    }


class P418SteadyLossWeightSensitivityTest(unittest.TestCase):
    def test_registered_settings_match_archived_sources(self) -> None:
        rows = load_settings(
            ROOT / "parameters/hccb_p418_steady_loss_weight_sensitivity.csv", ROOT
        )
        self.assertEqual(
            [
                (
                    row["state_data_weight"],
                    row["face_flux_weight"],
                    row["physics_balance_weight"],
                )
                for row in rows
            ],
            [(5.0, 1.0, 1.0), (10.0, 5.0, 1.0), (50.0, 10.0, 1.0)],
        )
        self.assertTrue(
            all(
                row["transfer_mapping"].endswith("ratio_only")
                for row in rows
            )
        )

    def test_summary_requires_same_model_data_and_initial_state(self) -> None:
        settings = load_settings(
            ROOT / "parameters/hccb_p418_steady_loss_weight_sensitivity.csv", ROOT
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            standard = root / "standard" / "summary.json"
            standard.parent.mkdir()
            standard.write_text(
                json.dumps(
                    fake_summary(
                        {
                            "state_data": 5.0,
                            "face_flux": 1.0,
                            "physics_balance": 1.0,
                        }
                    )
                ),
                encoding="utf-8",
            )
            selection = root / "selection.json"
            selection.write_text(
                json.dumps(
                    {
                        "status": "steady_PINN_chain_source_selected",
                        "selected_epochs": 100,
                        "split_name": "interleaved_all_ranges",
                        "selected_summary": str(standard),
                    }
                ),
                encoding="utf-8",
            )
            for row in settings[1:]:
                target = (
                    root
                    / f"hccb_p418_loss_weight_{row['setting_id']}_100epoch"
                    / "summary.json"
                )
                target.parent.mkdir()
                target.write_text(
                    json.dumps(
                        fake_summary(
                            {
                                "state_data": row["state_data_weight"],
                                "face_flux": row["face_flux_weight"],
                                "physics_balance": row["physics_balance_weight"],
                            }
                        )
                    ),
                    encoding="utf-8",
                )
            payload = summarize(
                project_root=ROOT,
                settings_path=ROOT
                / "parameters/hccb_p418_steady_loss_weight_sensitivity.csv",
                selection_path=selection,
                result_root=root,
            )
            self.assertEqual(payload["setting_count"], 3)
            self.assertTrue(payload["same_data_network_optimizer_seed_and_initial_state"])
            self.assertTrue(payload["all_settings_best_epoch_is_not_final"])
            output = root / "output"
            write_outputs(payload, output)
            self.assertTrue((output / "loss_weight_sensitivity.csv").is_file())
            self.assertTrue((output / "损失比例比较_CN.md").is_file())

            changed = json.loads(target.read_text(encoding="utf-8"))
            changed["initial_model_state_sha256"] = "2" * 64
            target.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "same data, model and initialization"):
                summarize(
                    project_root=ROOT,
                    settings_path=ROOT
                    / "parameters/hccb_p418_steady_loss_weight_sensitivity.csv",
                    selection_path=selection,
                    result_root=root,
                )

            changed["initial_model_state_sha256"] = "1" * 64
            changed["microbatch_size"] = 2
            target.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "same data, model and initialization"):
                summarize(
                    project_root=ROOT,
                    settings_path=ROOT
                    / "parameters/hccb_p418_steady_loss_weight_sensitivity.csv",
                    selection_path=selection,
                    result_root=root,
                )

    def test_formal_route_runs_sensitivity_after_epoch_selection(self) -> None:
        poststeady = (ROOT / "code/run_hccb_p418_poststeady_pipeline.sh").read_text(
            encoding="utf-8"
        )
        selection_position = poststeady.index("select_hccb_p418_steady_chain_source.py")
        sensitivity_position = poststeady.index(
            "run_hccb_p418_steady_loss_weight_sensitivity.sh"
        )
        chain_position = poststeady.index("run_hccb_p418_chained_initial_state_evaluation.sh")
        self.assertLess(selection_position, sensitivity_position)
        self.assertLess(sensitivity_position, chain_position)
        self.assertIn("steady_loss_weight_sensitivity_sha256", poststeady)
        runner = (ROOT / "code/run_hccb_p418_steady_loss_weight_sensitivity.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('--state-data-weight "${state_weight}"', runner)
        self.assertIn('--face-flux-weight "${face_weight}"', runner)
        self.assertIn('--physics-balance-weight "${physics_weight}"', runner)
        self.assertIn('--effective-batch-size "${effective_batch_size}"', runner)
        self.assertIn('--microbatch-size "${microbatch_size}"', runner)
        self.assertIn('print(summary["training_seed"])', runner)


if __name__ == "__main__":
    unittest.main()
