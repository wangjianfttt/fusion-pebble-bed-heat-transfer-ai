#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import hashlib
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is tested remotely")
class EvaluateP418ChainedInitialStateTest(unittest.TestCase):
    def test_source_temperature_and_target_hydrodynamics_are_composed(self) -> None:
        from evaluate_hccb_p418_chained_initial_state import (
            chained_prediction_artifact,
            compose_chained_initial_state,
            deterministic_chain_model_cost,
            load_steady_prediction_map,
            registered_steady_endpoint_inference_time,
            verify_registered_file,
            volume_weighted_temperature_rmse,
        )
        from hccb_p418_chain_roles import (
            DETERMINISTIC_CHAIN_STATUS,
            FUSED_CHAIN_STATUS,
            endpoint_novelty_class,
            steady_condition_roles,
            summarize_endpoint_groups,
        )

        self.assertEqual(
            DETERMINISTIC_CHAIN_STATUS,
            "completed_p418_steady_PINN_to_graph_transformer_chain",
        )
        self.assertEqual(
            FUSED_CHAIN_STATUS,
            "completed_p418_steady_PINN_graph_transformer_diffusion_chain",
        )

        node_type = np.asarray([0, 1], dtype=np.int64)
        source = np.asarray(
            [[1.0, 2.0, 3.0, 4.0, 300.0], [0.0, 0.0, 0.0, 0.0, 500.0]]
        )
        target = np.asarray(
            [[5.0, 6.0, 7.0, 8.0, 900.0], [0.0, 0.0, 0.0, 0.0, 700.0]]
        )
        chained = compose_chained_initial_state(source, target, node_type)
        np.testing.assert_allclose(chained[0, :4], target[0, :4])
        np.testing.assert_allclose(chained[:, 4], source[:, 4])
        np.testing.assert_allclose(chained[1, :4], 0.0)
        rmse = volume_weighted_temperature_rmse(
            chained[None, :, 4], source[None, :, 4], node_type, np.ones(2), 1
        )
        self.assertEqual(rmse, 0.0)

        upstream = {
            "sequence_id": np.asarray(["step-a"]),
            "baseline_temperature_normalized": np.zeros((1, 2, 2, 1)),
            "target_temperature_normalized": np.ones((1, 2, 2, 1)),
            "fluid_internal_mass_flux_kg_s": np.asarray([[1.0, 2.0]]),
            "fluid_boundary_mass_flux_kg_s": np.asarray([[3.0, 4.0, 5.0]]),
        }
        chained_artifact = chained_prediction_artifact(
            upstream,
            ["step-a"],
            [np.full((2, 2, 1), 0.25, dtype=np.float32)],
            [np.asarray([10.0, 20.0])],
            [np.asarray([30.0, 40.0, 50.0])],
        )
        np.testing.assert_allclose(
            chained_artifact["baseline_temperature_normalized"], 0.25
        )
        np.testing.assert_allclose(
            chained_artifact["exact_initial_baseline_temperature_normalized"], 0.0
        )
        np.testing.assert_allclose(upstream["baseline_temperature_normalized"], 0.0)
        np.testing.assert_allclose(
            chained_artifact["fluid_internal_mass_flux_kg_s"], [[10.0, 20.0]]
        )
        np.testing.assert_allclose(
            chained_artifact["fluid_boundary_mass_flux_kg_s"],
            [[30.0, 40.0, 50.0]],
        )
        np.testing.assert_allclose(
            chained_artifact["exact_target_fluid_internal_mass_flux_kg_s"],
            [[1.0, 2.0]],
        )
        np.testing.assert_allclose(
            chained_artifact["exact_target_fluid_boundary_mass_flux_kg_s"],
            [[3.0, 4.0, 5.0]],
        )
        with self.assertRaisesRegex(ValueError, "differs from upstream shape"):
            chained_prediction_artifact(
                upstream,
                ["step-a"],
                [np.full((2, 2, 1), 0.25, dtype=np.float32)],
                [np.asarray([10.0])],
                [np.asarray([30.0, 40.0, 50.0])],
            )

        roles = steady_condition_roles(
            {
                "split_case_ids": {
                    "train": ["a"],
                    "validation": ["b"],
                    "test": ["c"],
                }
            }
        )
        self.assertEqual(roles, {"a": "train", "b": "validation", "c": "test"})
        self.assertEqual(
            endpoint_novelty_class("test", "validation"),
            "both_steady_endpoints_unseen",
        )
        self.assertEqual(
            endpoint_novelty_class("train", "test"),
            "one_steady_endpoint_unseen",
        )
        self.assertEqual(
            endpoint_novelty_class("train", "train"),
            "steady_endpoints_seen_transient_held_out",
        )
        grouped = summarize_endpoint_groups(
            [
                {
                    "endpoint_novelty_class": "both_steady_endpoints_unseen",
                    "source_solid_initial_temperature_RMSE_K": 2.0,
                    "exact_initial_solid_temperature_RMSE_K": 4.0,
                    "steady_PINN_initial_solid_temperature_RMSE_K": 5.0,
                },
                {
                    "endpoint_novelty_class": "both_steady_endpoints_unseen",
                    "source_solid_initial_temperature_RMSE_K": 4.0,
                    "exact_initial_solid_temperature_RMSE_K": 6.0,
                    "steady_PINN_initial_solid_temperature_RMSE_K": 10.0,
                },
            ]
        )["both_steady_endpoints_unseen"]
        self.assertEqual(grouped["curve_count"], 2)
        self.assertEqual(grouped["mean_source_initial_temperature_RMSE_K"], 3.0)
        self.assertEqual(grouped["exact_initial_mean_solid_temperature_RMSE_K"], 5.0)
        self.assertEqual(grouped["steady_PINN_initial_mean_solid_temperature_RMSE_K"], 7.5)
        self.assertEqual(grouped["mean_error_amplification"], 1.5)

        endpoint_timing = registered_steady_endpoint_inference_time(
            steady_summary={
                "evaluations": {
                    "train": {"inference_seconds_per_case": 0.1},
                    "validation": {"inference_seconds_per_case": 0.2},
                    "test": {"inference_seconds_per_case": 0.3},
                }
            },
            role_by_condition={"a": "train", "b": "validation", "c": "test"},
            condition_ids=["a", "b", "a", "c"],
        )
        self.assertEqual(endpoint_timing["unique_endpoint_count"], 3)
        self.assertAlmostEqual(
            endpoint_timing["measured_unique_endpoint_inference_seconds"], 0.6
        )
        self.assertEqual(
            endpoint_timing["endpoint_count_by_steady_role"],
            {"train": 1, "validation": 1, "test": 1},
        )

        model_cost = deterministic_chain_model_cost(
            steady_summary={"model_parameter_count": 100, "training_seconds": 20.0},
            transient_summary={"model_parameter_count": 200, "training_seconds": 30.0},
        )
        self.assertEqual(model_cost["deterministic_chain_model_parameter_count"], 300)
        self.assertEqual(model_cost["deterministic_chain_training_seconds"], 50.0)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prediction.npz"
            path.write_bytes(b"formal prediction")
            record = {
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            verify_registered_file(path, record, "prediction")
            path.write_bytes(b"changed prediction")
            with self.assertRaisesRegex(ValueError, "differs"):
                verify_registered_file(path, record, "prediction")

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            node_type = np.asarray([0, 1], dtype=np.int64)
            node_volume = np.asarray([1.0, 2.0], dtype=np.float64)
            conditions = np.zeros((3, 4), dtype=np.float64)
            identifiers = np.asarray(["train-a", "validation-b", "test-c"])
            state_targets = directory / "state_targets.npz"
            np.savez_compressed(
                state_targets,
                condition_id=identifiers,
                condition_physical=conditions,
                node_type=node_type,
                node_volume_m3=node_volume,
            )
            statistics = directory / "statistics.json"
            statistics.write_text(
                json.dumps(
                    {
                        "splits": {
                            "interleaved_all_ranges": {
                                "condition_input": {
                                    "mean": [0.0, 0.0, 0.0, 0.0],
                                    "standard_deviation": [1.0, 1.0, 1.0, 1.0],
                                },
                                "targets": {
                                    "fluid_velocity_m_s": {
                                        "mean": [0.0, 0.0, 0.0],
                                        "standard_deviation": [1.0, 1.0, 1.0],
                                    },
                                    "fluid_gauge_pressure_Pa": {
                                        "mean": [0.0],
                                        "standard_deviation": [1.0],
                                    },
                                    "fluid_temperature_K": {
                                        "mean": [0.0],
                                        "standard_deviation": [1.0],
                                    },
                                    "solid_temperature_K": {
                                        "mean": [0.0],
                                        "standard_deviation": [1.0],
                                    },
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            files = {}
            records = {}
            expected_flux = {}
            for index, (role, identifier) in enumerate(
                zip(("train", "validation", "test"), identifiers)
            ):
                prediction_path = directory / f"{role}_regional_predictions.npz"
                internal = np.asarray([[index + 1.0, index + 2.0]])
                boundary = np.asarray([[index + 3.0]])
                np.savez_compressed(
                    prediction_path,
                    condition_id=np.asarray([identifier]),
                    baseline_state_normalized=np.full((1, 2, 5), index + 1.0),
                    internal_mass_flow_kg_s=internal,
                    boundary_mass_flow_kg_s=boundary,
                    node_type=node_type,
                    node_volume_m3=node_volume,
                )
                files[role] = prediction_path.name
                records[role] = {
                    "size_bytes": prediction_path.stat().st_size,
                    "sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
                }
                expected_flux[str(identifier)] = (internal[0], boundary[0])
            summary_path = directory / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "architecture": "pinn",
                        "split_name": "interleaved_all_ranges",
                        "split_case_ids": {
                            "train": ["train-a"],
                            "validation": ["validation-b"],
                            "test": ["test-c"],
                        },
                        "regional_prediction_files": files,
                        "regional_prediction_file_records": records,
                    }
                ),
                encoding="utf-8",
            )
            states, fluxes, loaded_type, loaded_volume, loaded_roles = (
                load_steady_prediction_map(
                    summary_path=summary_path,
                    state_targets_path=state_targets,
                    training_statistics_path=statistics,
                    split_name="interleaved_all_ranges",
                )
            )
            self.assertEqual(set(states), set(identifiers))
            self.assertEqual(loaded_roles["test-c"], "test")
            np.testing.assert_array_equal(loaded_type, node_type)
            np.testing.assert_allclose(loaded_volume, node_volume)
            for identifier in identifiers:
                np.testing.assert_allclose(
                    fluxes[str(identifier)][0], expected_flux[str(identifier)][0]
                )
                np.testing.assert_allclose(
                    fluxes[str(identifier)][1], expected_flux[str(identifier)][1]
                )


if __name__ == "__main__":
    unittest.main()
