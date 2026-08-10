#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "code/train_hccb_p418_temporal_temperature_diffusion.py"


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is tested on the remote machine")
class TrainP418TemporalTemperatureDiffusionTest(unittest.TestCase):
    def test_inference_progress_is_atomically_replaced(self) -> None:
        sys.path.insert(0, str(ROOT / "code"))
        from train_hccb_p418_temporal_temperature_diffusion import (
            write_inference_progress,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inference_progress.json"
            write_inference_progress(path, {"completed_case_count": 1})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"completed_case_count": 1},
            )
            write_inference_progress(path, {"completed_case_count": 2})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"completed_case_count": 2},
            )
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_observation_source_must_match_computed_sparse_values(self) -> None:
        sys.path.insert(0, str(ROOT / "code"))
        from train_hccb_p418_temporal_temperature_diffusion import (
            observation_masks,
            validate_observation_input,
        )

        validate_observation_input(
            "computed_residual_benchmark",
            None,
            "none",
        )
        with self.assertRaisesRegex(ValueError, "explicit observation source"):
            validate_observation_input(
                "sparse_reconstruction",
                Path("mask.npz"),
                "none",
            )
        with self.assertRaisesRegex(ValueError, "does not yet ingest measured"):
            validate_observation_input(
                "sparse_reconstruction",
                Path("mask.npz"),
                "external_experiment",
            )
        validate_observation_input(
            "sparse_reconstruction",
            Path("mask.npz"),
            "computed_openfoam_target",
        )
        split_values = {
            "train": {
                "baseline_temperature_normalized": np.zeros(
                    (2, 3, 4, 1), dtype=np.float32
                )
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mask.npz"
            mask = np.zeros((2, 3, 4, 1), dtype=bool)
            mask[0, 1, 2, 0] = True
            np.savez_compressed(
                path,
                train_mask=mask,
                observation_source_kind=np.asarray(
                    "computed_openfoam_target"
                ),
                observed_values_kind=np.asarray(
                    "target_temperature_normalized_from_openfoam_reference"
                ),
            )
            loaded = observation_masks(
                path,
                split_values,
                "computed_openfoam_target",
            )
            np.testing.assert_array_equal(loaded["train"], mask)

    def test_training_checkpoint_restores_model_ema_optimizer_and_generators(self) -> None:
        import torch

        sys.path.insert(0, str(ROOT / "code"))
        from train_hccb_p418_temporal_temperature_diffusion import (
            load_diffusion_training_checkpoint,
            save_diffusion_training_checkpoint,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_checkpoint.pt"
            model = torch.nn.Linear(3, 2)
            ema = copy.deepcopy(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=1.0e-3, total_steps=2
            )
            order_generator = torch.Generator(device="cpu").manual_seed(18)
            diffusion_generator = torch.Generator(device="cpu").manual_seed(19)
            loss = model(torch.ones(1, 3)).square().mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema.load_state_dict(model.state_dict())
            saved_parameters = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            contract = {"split": ["train_a", "validation_a", "test_a"], "epochs": 2}
            save_diffusion_training_checkpoint(
                path,
                contract=contract,
                next_epoch=1,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                order_generator=order_generator,
                diffusion_generator=diffusion_generator,
                best_validation=0.4,
                best_epoch=1,
                best_ema_state=saved_parameters,
                history=[{"epoch": 1, "training_velocity_loss": 0.5}],
                validation_history=[{"epoch": 1, "validation_velocity_loss": 0.4}],
                training_seconds=12.0,
            )
            expected_order = torch.randperm(8, generator=order_generator)
            expected_noise = torch.randn(4, generator=diffusion_generator)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                for parameter in ema.parameters():
                    parameter.fill_(2.0)
            restored = load_diffusion_training_checkpoint(
                path,
                contract=contract,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                order_generator=order_generator,
                diffusion_generator=diffusion_generator,
                device=torch.device("cpu"),
            )
            self.assertEqual(restored["next_epoch"], 1)
            self.assertEqual(restored["best_epoch"], 1)
            self.assertEqual(restored["training_seconds"], 12.0)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, saved_parameters[name])
            for name, value in ema.state_dict().items():
                torch.testing.assert_close(value, saved_parameters[name])
            torch.testing.assert_close(
                torch.randperm(8, generator=order_generator), expected_order
            )
            torch.testing.assert_close(
                torch.randn(4, generator=diffusion_generator), expected_noise
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_diffusion_training_checkpoint(
                    path,
                    contract={"split": ["different"], "epochs": 2},
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    order_generator=order_generator,
                    diffusion_generator=diffusion_generator,
                    device=torch.device("cpu"),
                )

    def test_formal_prediction_contract_keeps_complete_curves_disjoint(self) -> None:
        import sys

        sys.path.insert(0, str(ROOT / "code"))
        from train_hccb_p418_temporal_temperature_diffusion import (
            validate_deterministic_prediction_contract,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = {
                "train": np.asarray(["train_a", "train_b"]),
                "validation": np.asarray(["validation_a", "validation_b"]),
                "test": np.asarray(["test_a", "test_b"]),
            }
            for role in ids:
                (root / f"{role}_temporal_temperature_predictions.npz").write_bytes(
                    b"fixture"
                )
            summary = {
                "status": "completed_p418_spatiotemporal_regional_operator",
                "run_role": "formal",
                "physics_mode": "energy_and_flux",
                "selection_split": "validation",
                "new_physical_parameters": [],
                "split_case_ids": {
                    role: values.tolist() for role, values in ids.items()
                },
                "temporal_temperature_prediction_files": {
                    role: f"{role}_temporal_temperature_predictions.npz"
                    for role in ids
                },
            }
            ids["train"] = np.asarray(
                [f"train_{index}" for index in range(8)]
            )
            summary["split_case_ids"]["train"] = ids["train"].tolist()
            splits = {role: {"sequence_id": values} for role, values in ids.items()}
            validate_deterministic_prediction_contract(
                summary=summary,
                splits=splits,
                prediction_dir=root,
                run_role="computed_residual_benchmark",
            )

            summary["split_case_ids"]["test"] = ["train_0"]
            splits["test"]["sequence_id"] = np.asarray(["train_0"])
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_deterministic_prediction_contract(
                    summary=summary,
                    splits=splits,
                    prediction_dir=root,
                    run_role="computed_residual_benchmark",
                )

            summary["run_role"] = "smoke"
            splits["test"]["sequence_id"] = np.asarray(["test_a", "test_b"])
            summary["split_case_ids"]["test"] = ["test_a", "test_b"]
            with self.assertRaisesRegex(ValueError, "formal deterministic run"):
                validate_deterministic_prediction_contract(
                    summary=summary,
                    splits=splits,
                    prediction_dir=root,
                    run_role="computed_residual_benchmark",
                )

    def test_curve_microbatch_gradients_match_full_effective_batch(self) -> None:
        import sys

        import torch

        sys.path.insert(0, str(ROOT / "code"))
        from hccb_p418_temporal_temperature_diffusion import (
            P418TemporalTemperatureResidualRefiner,
        )
        from train_hccb_p418_temporal_temperature_diffusion import (
            energy_residual_summary,
            observation_masks,
            physical_temperature_state,
            projection_aware_energy_difference_mse,
            training_time_scale,
            unobserved_dynamic_selection,
            weighted_velocity_loss,
        )

        torch.manual_seed(17)
        batch, times, nodes = 2, 3, 5
        model = P418TemporalTemperatureResidualRefiner(
            structural_dim=6,
            hidden_dim=16,
            spatial_layers=1,
            spatial_attention_heads=4,
            physics_slices=4,
            temporal_layers=1,
            temporal_heads=1,
            temporal_node_chunk_size=3,
        )
        micro_model = copy.deepcopy(model)
        baseline = torch.randn(batch, times, nodes, 1)
        noised = torch.randn_like(baseline)
        target = torch.randn_like(baseline)
        condition = torch.randn(batch, 8)
        structure = torch.randn(nodes, 6)
        time_value = torch.linspace(0, 1, times).expand(batch, -1)
        observed = torch.zeros_like(baseline)
        mask = torch.zeros_like(baseline, dtype=torch.bool)
        step = torch.ones(batch, dtype=torch.long)
        volume = torch.linspace(1.0, 2.0, nodes)

        full_prediction = model(
            baseline,
            noised,
            condition,
            structure,
            time_value,
            observed,
            mask,
            step,
        )
        weighted_velocity_loss(full_prediction, target, volume).backward()

        for index in range(batch):
            micro_prediction = micro_model(
                baseline[index : index + 1],
                noised[index : index + 1],
                condition[index : index + 1],
                structure,
                time_value[index : index + 1],
                observed[index : index + 1],
                mask[index : index + 1],
                step[index : index + 1],
            )
            (weighted_velocity_loss(
                micro_prediction, target[index : index + 1], volume
            ) / batch).backward()

        for full_parameter, micro_parameter in zip(
            model.parameters(), micro_model.parameters()
        ):
            self.assertIsNotNone(full_parameter.grad)
            self.assertIsNotNone(micro_parameter.grad)
            torch.testing.assert_close(
                full_parameter.grad, micro_parameter.grad, rtol=2.0e-5, atol=2.0e-6
            )

        fixed_hydrodynamics = np.arange(20, dtype=np.float32).reshape(5, 4)
        state = physical_temperature_state(
            np.zeros((3, 5, 1), dtype=np.float32),
            fixed_hydrodynamics,
            np.asarray([0, 0, 1, 1, 1]),
            np.asarray([300.0, 600.0]),
            np.asarray([10.0, 20.0]),
        )
        self.assertEqual(state.shape, (3, 5, 5))
        np.testing.assert_array_equal(
            state[..., :4], np.broadcast_to(fixed_hydrodynamics, (3, 5, 4))
        )
        np.testing.assert_allclose(state[:, :2, 4], 300.0)
        np.testing.assert_allclose(state[:, 2:, 4], 600.0)
        energy = energy_residual_summary(4.0, 1.0, 0.25)
        self.assertEqual(
            energy["diffusion_to_deterministic_energy_residual_ratio"], 0.5
        )
        self.assertEqual(
            energy["diffusion_to_openfoam_reference_energy_residual_ratio"], 2.0
        )
        from types import SimpleNamespace

        predicted_residual = SimpleNamespace(
            fluid_energy_w_m3=torch.tensor([[[3.0, 5.0]]]),
            solid_energy_w_m3=torch.tensor([[[8.0]]]),
        )
        reference_residual = SimpleNamespace(
            fluid_energy_w_m3=torch.tensor([[[1.0, 1.0]]]),
            solid_energy_w_m3=torch.tensor([[[2.0]]]),
        )
        condition = torch.zeros((1, 8))
        condition[:, 5] = 2.0
        projection_mse = projection_aware_energy_difference_mse(
            predicted_residual,
            reference_residual,
            condition,
            torch.ones(2),
            torch.ones(1),
        )
        torch.testing.assert_close(projection_mse, torch.tensor(5.75e-12))
        split_values = {
            "train": {
                "baseline_temperature_normalized": np.zeros(
                    (2, 3, 5, 1), dtype=np.float32
                )
            }
        }
        empty_masks = observation_masks(None, split_values)
        self.assertFalse(empty_masks["train"].any())
        source_mask = np.zeros((3, 5, 1), dtype=bool)
        source_mask[1, 2, 0] = True
        node_type = np.asarray([0, 0, 0, 1, 1], dtype=np.int8)
        all_unobserved = unobserved_dynamic_selection(source_mask, node_type)
        self.assertFalse(all_unobserved[0].any())
        self.assertFalse(all_unobserved[1, 2])
        self.assertTrue(all_unobserved[2, 2])
        fluid_unobserved = unobserved_dynamic_selection(
            source_mask, node_type, material=0
        )
        self.assertFalse(fluid_unobserved[:, node_type == 1].any())
        time_splits = {
            "train": {"time_s": np.asarray([[0.0, 1.0, 2.0]])},
            "validation": {"time_s": np.asarray([[0.0, 5.0]])},
            "test": {"time_s": np.asarray([[0.0, 9.0]])},
        }
        self.assertEqual(training_time_scale(time_splits), 2.0)

    def test_small_complete_curve_residual_pipeline_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nodes, times = 7, 4
            node_type = np.asarray([0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
            volume = np.ones(nodes, dtype=np.float32)
            structure = np.column_stack(
                (np.linspace(0, 1, nodes), np.zeros((nodes, 5)))
            ).astype(np.float32)
            counts = {"train": 6, "validation": 3, "test": 3}
            for role, count in counts.items():
                baseline = np.zeros((count, times, nodes, 1), dtype=np.float32)
                target = baseline.copy()
                for index in range(count):
                    target[index, :, :, 0] = (
                        0.01 * (index + 1) * np.linspace(0, 1, times)[:, None]
                    )
                np.savez_compressed(
                    root / f"{role}_temporal_temperature_predictions.npz",
                    sequence_id=np.asarray([f"{role}_{i}" for i in range(count)]),
                    time_s=np.broadcast_to(np.arange(times), (count, times)),
                    condition_physical=np.zeros((count, 8), dtype=np.float32),
                    condition_normalized=np.zeros((count, 8), dtype=np.float32),
                    fixed_hydrodynamics_physical=np.zeros(
                        (count, nodes, 4), dtype=np.float32
                    ),
                    fluid_internal_mass_flux_kg_s=np.zeros(
                        (count, 1), dtype=np.float32
                    ),
                    fluid_boundary_mass_flux_kg_s=np.zeros(
                        (count, 2), dtype=np.float32
                    ),
                    baseline_temperature_normalized=baseline,
                    target_temperature_normalized=target,
                    node_type=node_type,
                    node_volume_m3=volume,
                    node_centroid_m=np.column_stack(
                        (np.arange(nodes, dtype=np.float32), np.zeros((nodes, 2), dtype=np.float32))
                    ),
                    structural_features=structure,
                    temperature_mean_K_by_node_type=np.asarray([300.0, 600.0]),
                    temperature_std_K_by_node_type=np.asarray([10.0, 20.0]),
                )
            output = root / "result"
            command = [
                    sys.executable,
                    str(TRAINER),
                    "--prediction-dir",
                    str(root),
                    "--output-dir",
                    str(output),
                    "--run-role",
                    "software_smoke",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--microbatch-size",
                    "1",
                    "--activation-precision",
                    "float32",
                    "--hidden-dim",
                    "16",
                    "--spatial-layers",
                    "1",
                    "--spatial-attention-heads",
                    "4",
                    "--physics-slices",
                    "4",
                    "--temporal-layers",
                    "1",
                    "--temporal-heads",
                    "1",
                    "--temporal-node-chunk-size",
                    "3",
                    "--device",
                    "cpu",
                    "--threads",
                    "2",
                    "--ensemble-samples",
                    "2",
                ]
            result = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=result.stdout + "\n" + result.stderr,
            )
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["split_case_counts"], counts)
            self.assertEqual(summary["run_role"], "software_smoke")
            self.assertEqual(summary["metrics"]["test"]["observation_count"], 0)
            self.assertEqual(summary["ensemble_samples"], 2)
            self.assertEqual(summary["effective_batch_size"], 2)
            self.assertEqual(summary["microbatch_size"], 1)
            self.assertEqual(summary["activation_precision"], "float32")
            self.assertEqual(summary["selection_split"], "validation")
            self.assertEqual(summary["time_normalization_source"], "training_curves_only")
            self.assertEqual(summary["time_normalization_maximum_s"], times - 1)
            self.assertEqual(summary["selected_epoch"], 1)
            self.assertEqual(summary["training_resumed_from_epoch"], 0)
            self.assertEqual(summary["training_checkpoint"], "training_checkpoint.pt")
            self.assertTrue((output / "training_checkpoint.pt").is_file())
            self.assertEqual(
                summary["observation_input"]["role"],
                "none_full_field_computed_residual_benchmark",
            )
            self.assertIsNone(summary["observation_input"]["mask_file"])
            self.assertEqual(summary["observation_input"]["source_kind"], "none")
            self.assertFalse(
                summary["observation_input"]["hard_conditioning_is_exact"]
            )
            self.assertFalse(
                summary["observation_input"]
                ["external_measurements_supported_by_this_trainer"]
            )
            self.assertFalse(
                summary["observation_input"]
                ["computed_openfoam_targets_are_measurements"]
            )
            self.assertEqual(summary["corrected_state_channels"], ["temperature"])
            self.assertEqual(
                summary["fixed_state_channels"],
                ["velocity_x", "velocity_y", "velocity_z", "pressure"],
            )
            self.assertTrue(np.isfinite(summary["best_validation_velocity_loss"]))
            self.assertIn("diffusion_member_RMSE_K_p95", summary["metrics"]["test"])
            self.assertIn(
                "diffusion_90pct_interval_coverage_fraction", summary["metrics"]["test"]
            )
            self.assertIn(
                "diffusion_refined_solid_temperature_RMSE_K", summary["metrics"]["test"]
            )
            self.assertIn(
                "diffusion_refined_solid_maximum_temperature_history_RMSE_K",
                summary["metrics"]["test"],
            )
            self.assertIn(
                "diffusion_refined_solid_regional_hotspot_location_p95_error_m",
                summary["metrics"]["test"],
            )
            self.assertIn(
                "diffusion_unobserved_dynamic_90pct_interval_coverage_fraction",
                summary["metrics"]["test"],
            )
            self.assertIn(
                "diffusion_unobserved_dynamic_fluid_90pct_interval_mean_width_K",
                summary["metrics"]["test"],
            )
            self.assertIn(
                "diffusion_unobserved_dynamic_solid_90pct_interval_mean_width_K",
                summary["metrics"]["test"],
            )
            self.assertIn(
                "diffusion_unobserved_dynamic_CRPS_K",
                summary["metrics"]["test"],
            )
            self.assertIn(
                "diffusion_unobserved_dynamic_fluid_CRPS_K",
                summary["metrics"]["test"],
            )
            self.assertIn(
                "diffusion_unobserved_dynamic_solid_CRPS_K",
                summary["metrics"]["test"],
            )
            with np.load(output / summary["prediction_files"]["test"], allow_pickle=False) as data:
                refined = data["refined_temperature_normalized"]
                np.testing.assert_allclose(refined[:, 0], 0.0, atol=1.0e-7)
                self.assertIn("refined_temperature_std_normalized", data.files)
                for required in (
                    "time_s",
                    "condition_physical",
                    "fixed_hydrodynamics_physical",
                    "fluid_internal_mass_flux_kg_s",
                    "fluid_boundary_mass_flux_kg_s",
                    "node_type",
                    "node_volume_m3",
                    "node_centroid_m",
                    "temperature_mean_K_by_node_type",
                    "temperature_std_K_by_node_type",
                ):
                    self.assertIn(required, data.files)
            resumed_result = subprocess.run(
                command + ["--resume"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                resumed_result.returncode,
                0,
                msg=resumed_result.stdout + "\n" + resumed_result.stderr,
            )
            resumed_summary = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(resumed_summary["training_resumed_from_epoch"], 1)
            self.assertEqual(len(resumed_summary["training_history"]), 1)


if __name__ == "__main__":
    unittest.main()
