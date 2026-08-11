#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/summarize_hccb_p418_step_model_comparison.py"
sys.path.insert(0, str(ROOT / "code"))

from summarize_hccb_p418_step_model_comparison import (
    TEMPERATURE_METRIC_DEFINITION,
    openfoam_clock_times,
    require_strict_loss_selection,
    require_temperature_metric_definition,
)


class P418StepModelComparisonTest(unittest.TestCase):
    def test_strict_comparison_waits_for_validation_selected_loss_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selected_downstream_integration.json"
            with self.assertRaisesRegex(ValueError, "before independent test aggregation"):
                require_strict_loss_selection(["pair_disjoint_stress_test"], path)
            require_strict_loss_selection(["direction_up_test"], path)
            path.write_text("{}\n", encoding="utf-8")
            require_strict_loss_selection(["pair_disjoint_stress_test"], path)

    def test_requires_common_temperature_metric_definition(self) -> None:
        require_temperature_metric_definition(
            {"temperature_metric_definition": TEMPERATURE_METRIC_DEFINITION},
            "matching_model",
        )
        with self.assertRaisesRegex(ValueError, "differs from the common"):
            require_temperature_metric_definition({}, "missing_model")
        with self.assertRaisesRegex(ValueError, "differs from the common"):
            require_temperature_metric_definition(
                {"temperature_metric_definition": "unweighted RMSE over all nodes"},
                "mismatched_model",
            )

    def test_openfoam_time_sums_restart_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / "thermal_step"
            case.mkdir()
            (case / "log.foamMultiRun.step").write_text(
                "ExecutionTime = 10 s ClockTime = 40 s\n"
                "===== resumed from complete parallel time 25 at "
                "2026-07-21T12:00:00+08:00 =====\n"
                "ExecutionTime = 10 s ClockTime = 60 s\n",
                encoding="utf-8",
            )
            self.assertEqual(openfoam_clock_times(root)["thermal_step"], 100.0)

    def test_collects_same_split_model_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step_root = root / "steps"
            step_root.mkdir()
            split_file = ROOT / "parameters/hccb_p418_step_response_splits.json"
            split_source = json.loads(split_file.read_text(encoding="utf-8"))["splits"]
            all_ids = set()
            for split_values in split_source.values():
                for role in ("train", "validation", "test"):
                    all_ids.update(map(str, split_values[role]))
            for sequence_id in all_ids:
                case = step_root / sequence_id
                case.mkdir()
                (case / "log.foamMultiRun.step").write_text(
                    "ExecutionTime = 10 s  ClockTime = 100 s\n", encoding="utf-8"
                )
            split_names = (
                "direction_down_test",
                "direction_up_test",
                "pair_disjoint_stress_test",
            )
            for split in split_names:
                observable = root / f"transformer_{split}"
                persistence = root / f"regional_persistence_{split}"
                dmdc = root / f"regional_dmdc_{split}"
                data_only = (
                    root / f"regional_graph_transformer_bounded_data_only_{split}"
                )
                physics = (
                    root / f"regional_graph_transformer_bounded_physics_{split}"
                )
                factorized = (
                    root / f"regional_graph_transformer_bounded_factorized_{split}"
                )
                low_rank = root / f"low_rank_temperature_residual_{split}"
                diffusion = root / f"temporal_diffusion_{split}"
                for directory in (
                    observable,
                    persistence,
                    dmdc,
                    data_only,
                    physics,
                    factorized,
                    low_rank,
                    diffusion,
                ):
                    directory.mkdir(parents=True)
                (observable / "summary.json").write_text(
                    json.dumps(
                        {
                            "test_mean_rmse_by_target": {"outlet_temperature_K": 3.0},
                            "training_seconds": 1.0,
                            "model_parameter_count": 101,
                            "compute_device": "cpu",
                            "split_case_counts": {"train": 6, "validation": 3, "test": 3},
                            "split_name": split,
                            "selection_split": "validation",
                            "selection_metric": "normalized trajectory MSE",
                            "selected_epoch": 2,
                            "split_case_ids": {
                                role: list(map(str, split_source[split][role]))
                                for role in ("train", "validation", "test")
                            },
                            "inference_seconds_by_split": {"train": 0.3, "validation": 0.2, "test": 0.15},
                        }
                    ),
                    encoding="utf-8",
                )
                for directory, value in (
                    (persistence, 7.0),
                    (dmdc, 6.0),
                    (data_only, 5.0),
                    (physics, 3.0),
                    (factorized, 2.8),
                ):
                    (directory / "summary.json").write_text(
                        json.dumps(
                            {
                                "metrics": {
                                    role: {
                                        "fluid_temperature_RMSE_K": value,
                                        "solid_temperature_RMSE_K": value,
                                        "maximum_absolute_temperature_error_K": value * 2,
                                        "solid_maximum_temperature_history_RMSE_K": value * 1.1,
                                        "solid_maximum_temperature_history_maximum_absolute_error_K": value * 2.1,
                                        "solid_regional_hotspot_location_mean_error_m": 0.01 * value,
                                        "solid_regional_hotspot_location_p95_error_m": 0.02 * value,
                                        "solid_regional_hotspot_location_maximum_error_m": 0.03 * value,
                                        "solid_regional_hotspot_exact_match_fraction": 0.8,
                                        "solid_hotspot_target_temperature_deficit_mean_K": 0.10 * value,
                                        "solid_hotspot_target_temperature_deficit_p95_K": 0.20 * value,
                                        "solid_hotspot_target_temperature_deficit_maximum_K": 0.30 * value,
                                        "solid_hotspot_prediction_temperature_deficit_mean_K": 0.08 * value,
                                        "solid_hotspot_prediction_temperature_deficit_p95_K": 0.16 * value,
                                        "solid_hotspot_prediction_temperature_deficit_maximum_K": 0.24 * value,
                                        "solid_hotspot_dynamic_sample_count": 100,
                                        "inference_seconds": 0.1,
                                        "inference_seconds_per_curve": 0.03,
                                    }
                                    for role in ("train", "validation", "test")
                                },
                                "training_seconds": 2.0,
                                "status": (
                                    "completed_p418_spatiotemporal_regional_operator"
                                    if directory in (data_only, physics, factorized)
                                    else "completed_baseline"
                                ),
                                "temperature_metric_definition": (
                                    TEMPERATURE_METRIC_DEFINITION
                                ),
                                "model_parameter_count": (
                                    0 if directory == persistence else 202
                                ),
                                "model_storage_scalar_count": (
                                    0 if directory == persistence else 77
                                ),
                                "compute_device": "cpu",
                                "split_name": split,
                                "selection_split": (
                                    "not_applicable"
                                    if directory == persistence
                                    else "validation"
                                ),
                                "selection_metric": (
                                    "none; this baseline has no fitted parameters"
                                    if directory == persistence
                                    else "solid-temperature RMSE in K"
                                ),
                                **(
                                    {"selected_rank": 4}
                                    if directory == dmdc
                                    else {"selected_epoch": 2}
                                ),
                                "split_case_ids": {
                                    role: list(map(str, split_source[split][role]))
                                    for role in ("train", "validation", "test")
                                },
                                "architecture": {
                                    "hidden_dim": 64,
                                    "spatial_temporal_mode": (
                                        "factorized_static_spatial"
                                        if directory == factorized
                                        else "repeated_query_spatial"
                                    ),
                                },
                                "loss_weights": {
                                    "temperature_data": 5.0,
                                    "reference_edge_energy_flux": 1.0,
                                    "projection_aware_transient_energy": 1.0,
                                },
                                "evaluation_stage": "final",
                                "test_evaluated": True,
                                "loss_balancing": {
                                    "candidate_id": "fixed_registered_5_1_1"
                                },
                                "new_physical_parameters": [],
                            }
                        ),
                    encoding="utf-8",
                )
                for directory in (data_only, physics, factorized):
                    np.savez_compressed(
                        directory / "training_statistics.npz",
                        state_mean=np.asarray([1.0, 2.0]),
                        state_std=np.asarray([3.0, 4.0]),
                    )
                (low_rank / "summary.json").write_text(
                    json.dumps(
                        {
                            "status": "completed_p418_low_rank_temperature_residual",
                            "metrics": {
                                role: {
                                    "fluid_temperature_RMSE_K": 2.7,
                                    "solid_temperature_RMSE_K": 2.7,
                                    "maximum_absolute_temperature_error_K": 5.4,
                                    "solid_maximum_temperature_history_RMSE_K": 2.8,
                                    "solid_maximum_temperature_history_maximum_absolute_error_K": 5.5,
                                    "solid_regional_hotspot_location_mean_error_m": 0.02,
                                    "solid_regional_hotspot_location_p95_error_m": 0.04,
                                    "solid_regional_hotspot_location_maximum_error_m": 0.06,
                                    "solid_regional_hotspot_exact_match_fraction": 0.85,
                                    "solid_hotspot_target_temperature_deficit_mean_K": 0.30,
                                    "solid_hotspot_target_temperature_deficit_p95_K": 0.60,
                                    "solid_hotspot_target_temperature_deficit_maximum_K": 0.90,
                                    "solid_hotspot_prediction_temperature_deficit_mean_K": 0.20,
                                    "solid_hotspot_prediction_temperature_deficit_p95_K": 0.40,
                                    "solid_hotspot_prediction_temperature_deficit_maximum_K": 0.70,
                                    "solid_hotspot_dynamic_sample_count": 100,
                                    "inference_seconds": 0.04,
                                    "inference_seconds_per_curve": 0.01,
                                }
                                for role in ("train", "validation", "test")
                            },
                            "training_seconds": 0.5,
                            "temperature_metric_definition": (
                                TEMPERATURE_METRIC_DEFINITION
                            ),
                            "model_storage_scalar_count": 55,
                            "model_size_definition": "stored low-rank scalars",
                            "compute_device": "cpu",
                            "split_name": split,
                            "selection_split": "validation",
                            "selection_metric": "solid-temperature RMSE in K",
                            "selected_rank": 2,
                            "split_case_ids": {
                                role: list(map(str, split_source[split][role]))
                                for role in ("train", "validation", "test")
                            },
                            "deterministic_prediction_dir": str(physics.resolve()),
                            "new_physical_parameters": [],
                        }
                    ),
                    encoding="utf-8",
                )
                (diffusion / "summary.json").write_text(
                    json.dumps(
                        {
                            "status": "completed_p418_temporal_temperature_diffusion",
                            "metrics": {
                                role: {
                                    "deterministic_temperature_RMSE_K": 3.0,
                                    "deterministic_solid_temperature_RMSE_K": 3.0,
                                    "diffusion_refined_temperature_RMSE_K": 2.5,
                                    "diffusion_refined_solid_temperature_RMSE_K": 2.5,
                                    "diffusion_refined_solid_maximum_temperature_history_RMSE_K": 2.6,
                                    "diffusion_refined_solid_maximum_temperature_history_maximum_absolute_error_K": 5.2,
                                    "diffusion_refined_solid_regional_hotspot_location_mean_error_m": 0.015,
                                    "diffusion_refined_solid_regional_hotspot_location_p95_error_m": 0.03,
                                    "diffusion_refined_solid_regional_hotspot_location_maximum_error_m": 0.05,
                                    "diffusion_refined_solid_regional_hotspot_exact_match_fraction": 0.9,
                                    "diffusion_refined_solid_hotspot_target_temperature_deficit_mean_K": 0.20,
                                    "diffusion_refined_solid_hotspot_target_temperature_deficit_p95_K": 0.50,
                                    "diffusion_refined_solid_hotspot_target_temperature_deficit_maximum_K": 0.80,
                                    "diffusion_refined_solid_hotspot_prediction_temperature_deficit_mean_K": 0.15,
                                    "diffusion_refined_solid_hotspot_prediction_temperature_deficit_p95_K": 0.35,
                                    "diffusion_refined_solid_hotspot_prediction_temperature_deficit_maximum_K": 0.60,
                                    "diffusion_refined_solid_hotspot_dynamic_sample_count": 100,
                                    "deterministic_absolute_energy_equation_normalized_RMSE": 0.30,
                                    "diffusion_refined_absolute_energy_equation_normalized_RMSE": 0.20,
                                    "openfoam_reference_absolute_energy_equation_normalized_RMSE": 0.10,
                                    "diffusion_to_deterministic_energy_residual_ratio": 2.0 / 3.0,
                                    "diffusion_to_openfoam_reference_energy_residual_ratio": 2.0,
                                    "diffusion_member_projection_aware_energy_equation_normalized_RMSE_p95": 0.25,
                                    "diffusion_member_joint_temperature_energy_improvement_fraction": 0.75,
                                    "diffusion_member_sample_count": 32,
                                    "diffusion_90pct_interval_coverage_fraction": 0.82,
                                    "diffusion_90pct_interval_mean_width_K": 12.0,
                                    "observation_count": 0,
                                    "inference_seconds": 0.3,
                                    "inference_seconds_per_curve": 0.1,
                                    "ensemble_mean_RMSE_K_by_sample_count": {"2": 2.6},
                                }
                                for role in ("train", "validation", "test")
                            },
                            "training_seconds": 4.0,
                            "split_name": split,
                            "temperature_metric_definition": (
                                TEMPERATURE_METRIC_DEFINITION
                            ),
                            "selection_split": "validation",
                            "selection_metric": "fixed-noise diffusion velocity loss",
                            "selected_epoch": 2,
                            "model_parameter_count": 303,
                            "compute_device": "cpu",
                            "split_case_ids": {
                                role: list(map(str, split_source[split][role]))
                                for role in ("train", "validation", "test")
                            },
                            "deterministic_prediction_dir": str(physics.resolve()),
                            "new_physical_parameters": [],
                        }
                    ),
                    encoding="utf-8",
                )
                for directory, energy_value in (
                    (persistence, 0.70),
                    (dmdc, 0.60),
                    (data_only, 0.50),
                    (physics, 0.30),
                    (factorized, 0.28),
                    (low_rank, 0.25),
                    (diffusion, 0.20),
                ):
                    (directory / "energy_balance_summary.json").write_text(
                        json.dumps(
                            {
                                "status": "completed_p418_common_transient_energy_balance",
                                "split_name": split,
                                "role_metrics": {
                                    role: {
                                        "curve_count": len(split_source[split][role]),
                                        "prediction_fluid_energy_equation_normalized_RMSE": energy_value,
                                        "prediction_solid_energy_equation_normalized_RMSE": energy_value,
                                        "prediction_combined_energy_equation_normalized_RMSE": energy_value,
                                        "prediction_volume_weighted_energy_equation_normalized_RMSE": energy_value,
                                        "projection_aware_volume_weighted_energy_equation_normalized_RMSE": energy_value,
                                        "openfoam_reference_fluid_energy_equation_normalized_RMSE": 0.10,
                                        "openfoam_reference_solid_energy_equation_normalized_RMSE": 0.10,
                                        "openfoam_reference_combined_energy_equation_normalized_RMSE": 0.10,
                                        "openfoam_reference_volume_weighted_energy_equation_normalized_RMSE": 0.10,
                                        "prediction_to_openfoam_energy_residual_ratio": energy_value / 0.10,
                                    }
                                    for role in ("train", "validation", "test")
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                if split == "pair_disjoint_stress_test":
                    integration_root = (
                        root / "fixed_flow_loss_balancing_pair_disjoint_stress_test"
                    )
                    integration_root.mkdir()
                    selection_path = (
                        integration_root / "selected_loss_balancing_method.json"
                    )
                    selection_path.write_text(
                        json.dumps(
                            {
                                "status": "p418_loss_balancing_selected_on_validation_only",
                                "selected_candidate_id": "fixed_registered_5_1_1",
                                "independent_test_read": False,
                            }
                        ),
                        encoding="utf-8",
                    )
                    model_dirs = {
                        "graph_transformer_energy_flux": physics,
                        "graph_transformer_factorized_energy_flux": factorized,
                        "low_rank_residual_correction": low_rank,
                        "diffusion_residual_correction": diffusion,
                    }
                    model_paths = {}
                    for model_name, directory in model_dirs.items():
                        summary_path = directory / "summary.json"
                        model_paths[model_name] = {
                            "directory_relative_to_result_root": str(
                                directory.relative_to(root)
                            ),
                            "summary_sha256": hashlib.sha256(
                                summary_path.read_bytes()
                            ).hexdigest(),
                        }
                    (integration_root / "selected_downstream_integration.json").write_text(
                        json.dumps(
                            {
                                "status": "completed_p418_selected_loss_balancing_downstream",
                                "split_name": split,
                                "selected_candidate_id": "fixed_registered_5_1_1",
                                "selection_record_sha256": hashlib.sha256(
                                    selection_path.read_bytes()
                                ).hexdigest(),
                                "independent_test_read_after_validation_selection": True,
                                "model_paths": model_paths,
                                "new_physical_parameters": [],
                            }
                        ),
                        encoding="utf-8",
                    )
            seed_summary_path = root / "strict_seed_summary.json"
            seed_summary_path.write_text(
                json.dumps(
                    {
                        "status": "completed_p418_strict_split_seed_robustness",
                        "split_name": "pair_disjoint_stress_test",
                        "seeds": [20260717, 20260718, 20260719],
                        "metrics": [
                            {
                                "model": "graph_transformer_energy_flux",
                                "metric": "solid_temperature_RMSE_K",
                                "seed_count": 3,
                                "mean_K": 3.1,
                                "sample_std_K": 0.2,
                                "minimum_K": 2.9,
                                "maximum_K": 3.3,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "comparison"
            subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--result-dir",
                    str(root),
                    "--step-root",
                    str(step_root),
                    "--splits",
                    str(split_file),
                    "--seed-robustness-summary",
                    str(seed_summary_path),
                    "--output-dir",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(
                summary["lowest_test_temperature_RMSE_model_by_split"]["direction_down_test"]
                ["lowest_test_solid_temperature_RMSE_model"],
                "diffusion_residual_correction",
            )
            self.assertTrue((output / "physical_step_model_metrics.csv").is_file())
            self.assertTrue((output / "physical_step_model_speedup.csv").is_file())
            self.assertTrue((output / "model_selection_evidence.csv").is_file())
            self.assertTrue(summary["fair_comparison_checks"]["same_complete_curve_splits"])
            with (output / "physical_step_model_metrics.csv").open(encoding="utf-8") as handle:
                metric_rows = list(csv.DictReader(handle))
            units = {row["metric"]: row["unit"] for row in metric_rows}
            self.assertEqual(units["solid_maximum_temperature_history_RMSE_K"], "K")
            self.assertEqual(units["solid_regional_hotspot_location_p95_error_m"], "m")
            self.assertEqual(units["solid_regional_hotspot_exact_match_fraction"], "fraction")
            self.assertEqual(units["solid_hotspot_target_temperature_deficit_p95_K"], "K")
            self.assertTrue(
                summary["fair_comparison_checks"]
                ["all_models_selected_using_validation_curves"]
            )
            self.assertTrue(
                summary["fair_comparison_checks"]
                ["diffusion_reports_energy_balance_before_and_after_correction"]
            )
            self.assertTrue(
                summary["fair_comparison_checks"]
                ["all_regional_field_models_use_common_energy_balance"]
            )
            diffusion_effect = summary[
                "diffusion_temperature_and_energy_effect_by_split"
            ]["direction_down_test"]
            self.assertTrue(diffusion_effect["temperature_improved"])
            self.assertTrue(diffusion_effect["projection_aware_energy_improved"])
            self.assertTrue(
                diffusion_effect[
                    "diffusion_is_joint_temperature_energy_improvement"
                ]
            )
            self.assertEqual(diffusion_effect["held_out_outcome"], "joint_improvement")
            self.assertFalse(diffusion_effect["model_selection_uses_this_outcome"])
            self.assertEqual(
                diffusion_effect["diffusion_90pct_interval_coverage_fraction"],
                0.82,
            )
            self.assertEqual(
                diffusion_effect["diffusion_90pct_interval_mean_width_K"],
                12.0,
            )
            self.assertTrue(
                diffusion_effect["diffusion_90pct_interval_is_underdispersed"]
            )
            self.assertEqual(
                summary["temperature_energy_pareto_models_by_split"]
                ["direction_down_test"][0]["model"],
                "diffusion_residual_correction",
            )
            self.assertIn("pair_disjoint_stress_test", summary["splits"])
            self.assertTrue(
                summary["fair_comparison_checks"]["pair_disjoint_split_in_main_table"]
            )
            self.assertEqual(
                summary["strict_split_seed_robustness"]["seeds"],
                [20260717, 20260718, 20260719],
            )
            self.assertEqual(
                summary["strict_split_seed_robustness"]
                ["lowest_three_seed_mean_field_model"],
                "graph_transformer_energy_flux",
            )
            self.assertAlmostEqual(
                summary["strict_split_seed_robustness"]
                ["corresponding_sample_std_K"],
                0.2,
            )
            speed_rows = list(
                csv.DictReader((output / "physical_step_model_speedup.csv").open(encoding="utf-8"))
            )
            dmdc_row = next(row for row in speed_rows if row["model"] == "dmdc")
            self.assertEqual(dmdc_row["model_size_scalar_count"], "77")
            self.assertEqual(dmdc_row["model_size_definition"], "stored DMDc scalars")
            persistence_row = next(
                row
                for row in speed_rows
                if row["model"] == "initial_temperature_persistence"
            )
            self.assertEqual(persistence_row["model_size_scalar_count"], "0")
            self.assertEqual(
                persistence_row["model_size_definition"],
                "no fitted or stored model scalars",
            )
            low_rank_row = next(
                row for row in speed_rows if row["model"] == "low_rank_residual_correction"
            )
            self.assertEqual(low_rank_row["upstream_model"], "graph_transformer_energy_flux")
            self.assertEqual(low_rank_row["component_model_size_scalar_count"], "55")
            self.assertEqual(low_rank_row["model_size_scalar_count"], "257")
            self.assertAlmostEqual(
                float(low_rank_row["model_inference_seconds_per_curve"]), 0.04
            )
            self.assertAlmostEqual(float(low_rank_row["training_wall_time_s"]), 2.5)
            self.assertEqual(low_rank_row["training_only_break_even_curve_count"], "1")
            self.assertEqual(low_rank_row["full_workflow_break_even_curve_count"], "10")
            diffusion_row = next(
                row for row in speed_rows if row["model"] == "diffusion_residual_correction"
            )
            self.assertEqual(diffusion_row["component_model_size_scalar_count"], "303")
            self.assertEqual(diffusion_row["model_size_scalar_count"], "505")
            self.assertAlmostEqual(
                float(diffusion_row["model_inference_seconds_per_curve"]), 0.13
            )
            self.assertAlmostEqual(float(diffusion_row["training_wall_time_s"]), 6.0)
            self.assertTrue(
                summary["fair_comparison_checks"]
                ["correction_speed_includes_upstream_prediction"]
            )
            self.assertTrue(
                summary["fair_comparison_checks"]["low_rank_uses_physics_constrained_prediction"]
            )

            bad_selection_path = root / "transformer_direction_down_test/summary.json"
            bad_selection = json.loads(
                bad_selection_path.read_text(encoding="utf-8")
            )
            bad_selection["selection_split"] = "test"
            bad_selection_path.write_text(
                json.dumps(bad_selection), encoding="utf-8"
            )
            rejected_selection = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--result-dir",
                    str(root),
                    "--step-root",
                    str(step_root),
                    "--splits",
                    str(split_file),
                    "--seed-robustness-summary",
                    str(seed_summary_path),
                    "--output-dir",
                    str(root / "comparison_bad_selection"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected_selection.returncode, 0)
            self.assertIn("validation curves only", rejected_selection.stderr)
            bad_selection["selection_split"] = "validation"
            bad_selection_path.write_text(
                json.dumps(bad_selection), encoding="utf-8"
            )

            bad_diffusion_path = root / "temporal_diffusion_direction_down_test/summary.json"
            bad_diffusion = json.loads(bad_diffusion_path.read_text(encoding="utf-8"))
            bad_diffusion["split_case_ids"]["test"][0] = "wrong_curve"
            bad_diffusion_path.write_text(json.dumps(bad_diffusion), encoding="utf-8")
            rejected_split = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--result-dir",
                    str(root),
                    "--step-root",
                    str(step_root),
                    "--splits",
                    str(split_file),
                    "--seed-robustness-summary",
                    str(seed_summary_path),
                    "--output-dir",
                    str(root / "comparison_bad_split"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected_split.returncode, 0)
            self.assertIn("curves differ", rejected_split.stderr)

    def test_diffusion_temperature_improvement_is_not_joint_when_energy_worsens(self) -> None:
        import sys

        sys.path.insert(0, str(ROOT / "code"))
        from summarize_hccb_p418_step_model_comparison import (
            diffusion_temperature_energy_decision,
        )

        decision = diffusion_temperature_energy_decision(3.0, 2.0, 0.2, 0.3)
        self.assertTrue(decision["temperature_improved"])
        self.assertFalse(decision["projection_aware_energy_not_worse"])
        self.assertFalse(
            decision["diffusion_is_joint_temperature_energy_improvement"]
        )
        self.assertEqual(decision["held_out_outcome"], "not_joint_improvement")
        self.assertFalse(decision["model_selection_uses_this_outcome"])
        self.assertEqual(
            decision["outcome_reason"],
            "energy_residual_increased_despite_lower_temperature_error",
        )

    def test_registered_range_rejection_keeps_partial_energy_record(self) -> None:
        from summarize_hccb_p418_step_model_comparison import (
            load_common_energy_summary,
        )

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "energy_balance_summary.json").write_text(
                json.dumps(
                    {
                        "status": (
                            "completed_p418_common_transient_energy_balance_"
                            "with_rejected_roles"
                        ),
                        "split_name": "pair_disjoint_stress_test",
                        "requested_roles": ["train", "validation", "test"],
                        "evaluated_roles": ["train", "validation"],
                        "rejected_roles": ["test"],
                        "role_metrics": {"train": {}, "validation": {}},
                        "role_failures": {
                            "test": {
                                "status": (
                                    "prediction_outside_registered_"
                                    "thermophysical_range"
                                ),
                                "prediction_solid_out_of_range_value_count": 4,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            _, summary = load_common_energy_summary(
                directory,
                split_name="pair_disjoint_stress_test",
                model_name="low_rank_residual_correction",
                allow_registered_range_rejections=True,
            )
            self.assertEqual(summary["rejected_roles"], ["test"])
            with self.assertRaisesRegex(ValueError, "lacks common"):
                load_common_energy_summary(
                    directory,
                    split_name="pair_disjoint_stress_test",
                    model_name="strict_model",
                )


if __name__ == "__main__":
    unittest.main()
