#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
TRAINER = ROOT / "code/train_hccb_p418_spatiotemporal_regional_operator.py"


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is tested on the remote machine")
class TrainP418SpatiotemporalRegionalOperatorTest(unittest.TestCase):
    def test_physics_time_chunks_preserve_derivatives_losses_and_gradients(self) -> None:
        import torch

        from hccb_p418_transient_regional_physics import time_derivative
        from train_hccb_p418_spatiotemporal_regional_operator import (
            physics_time_chunks,
            physics_training_losses,
            residual_loss_view,
        )

        time_s = torch.tensor(
            [[0.0, 0.02, 0.07, 0.15, 0.31, 0.56, 0.92, 1.45, 2.1]],
            dtype=torch.float64,
        )
        values = torch.arange(27, dtype=torch.float64).reshape(1, 9, 3)
        values = (0.2 * values.square() + torch.sin(values)).requires_grad_(True)
        complete_derivative = time_derivative(values, time_s)
        assembled_derivative = torch.empty_like(complete_derivative)
        weighted_derivative_mse = values.new_zeros(())
        chunks = physics_time_chunks(time_s.shape[1], 3)
        for extended, local_core, global_core, weight in chunks:
            derivative = time_derivative(values[:, extended], time_s[:, extended])
            assembled_derivative[:, global_core] = derivative[:, local_core]
            weighted_derivative_mse = (
                weighted_derivative_mse
                + weight * derivative[:, local_core].square().mean()
            )
        torch.testing.assert_close(assembled_derivative, complete_derivative)
        torch.testing.assert_close(
            weighted_derivative_mse, complete_derivative.square().mean()
        )

        reference = {
            "fluid_internal_energy_flux_w": torch.zeros((1, 9, 2), dtype=torch.float64),
            "solid_internal_heat_flux_w": torch.zeros((1, 9, 2), dtype=torch.float64),
            "fluid_energy_w_m3": torch.zeros((1, 9, 2), dtype=torch.float64),
            "solid_energy_w_m3": torch.zeros((1, 9, 2), dtype=torch.float64),
        }
        predicted = SimpleNamespace(
            fluid_internal_energy_flux_w=complete_derivative[..., :2],
            solid_internal_heat_flux_w=0.5 * complete_derivative[..., 1:],
            fluid_energy_w_m3=2.0 * complete_derivative[..., :2],
            solid_energy_w_m3=3.0 * complete_derivative[..., 1:],
        )
        condition = torch.tensor(
            [[0.05, 300.0, 4.85, 0.25, 900.0, 1.0e-6, 120000.0, 635.0]],
            dtype=torch.float64,
        )
        volume = torch.tensor([1.0, 2.0], dtype=torch.float64)
        area = torch.tensor([1.0, 3.0], dtype=torch.float64)
        complete_losses = physics_training_losses(
            predicted, reference, condition, volume, volume, area, area
        )
        chunked_losses = [values.new_zeros(()) for _ in complete_losses]
        for _, _, global_core, weight in chunks:
            chunk_losses = physics_training_losses(
                residual_loss_view(predicted, global_core),
                {
                    name: tensor[:, global_core]
                    for name, tensor in reference.items()
                },
                condition,
                volume,
                volume,
                area,
                area,
            )
            for index, loss in enumerate(chunk_losses):
                chunked_losses[index] = chunked_losses[index] + weight * loss
        for complete, chunked in zip(complete_losses, chunked_losses, strict=True):
            torch.testing.assert_close(chunked, complete)

        complete_gradient = torch.autograd.grad(
            sum(complete_losses), values, retain_graph=True
        )[0]
        chunked_gradient = torch.autograd.grad(sum(chunked_losses), values)[0]
        torch.testing.assert_close(chunked_gradient, complete_gradient)

    def test_physics_comparison_is_zero_for_identical_heat_flux_and_energy_terms(self) -> None:
        import torch

        from train_hccb_p418_spatiotemporal_regional_operator import (
            area_weighted_flux_density_mean_square,
            physics_difference_losses,
            physics_training_losses,
        )

        reference = {
            "fluid_internal_energy_flux_w": torch.zeros((1, 3, 4)),
            "solid_internal_heat_flux_w": torch.zeros((1, 3, 5)),
            "fluid_energy_w_m3": torch.zeros((1, 3, 2)),
            "solid_energy_w_m3": torch.zeros((1, 3, 2)),
        }
        predicted = SimpleNamespace(**{name: value.clone() for name, value in reference.items()})
        condition = torch.tensor([[0.05, 300.0, 4.85, 0.25, 900.0, 8.85, 120000.0, 635.0]])
        fluid_volume = torch.tensor([3.0e-9, 1.0e-9])
        solid_volume = torch.tensor([1.0e-9, 2.0e-9])
        fluid_area = torch.ones(4)
        solid_area = torch.ones(5)
        edge, energy = physics_difference_losses(
            predicted, reference, condition, fluid_volume, solid_volume,
            fluid_area, solid_area
        )
        torch.testing.assert_close(edge, torch.zeros_like(edge))
        torch.testing.assert_close(energy, torch.zeros_like(energy))
        predicted.fluid_internal_energy_flux_w[..., 0] = 1.0e-3
        predicted.solid_energy_w_m3[..., 0] = 1.0e3
        edge, energy = physics_difference_losses(
            predicted, reference, condition, fluid_volume, solid_volume,
            fluid_area, solid_area
        )
        self.assertGreater(float(edge), 0.0)
        self.assertGreater(float(energy), 0.0)

        # A projected OpenFOAM field generally has a non-zero coarse residual.
        # Matching it must give zero trainable projection-aware loss while the
        # absolute residual remains visible as a separate diagnostic.
        predicted = SimpleNamespace(**{name: value.clone() for name, value in reference.items()})
        predicted.fluid_energy_w_m3[..., 0] = 2.0e3
        reference["fluid_energy_w_m3"][..., 0] = 2.0e3
        edge, projection_energy, absolute_diagnostic = physics_training_losses(
            predicted, reference, condition, fluid_volume, solid_volume,
            fluid_area, solid_area
        )
        torch.testing.assert_close(edge, torch.zeros_like(edge))
        torch.testing.assert_close(
            projection_energy, torch.zeros_like(projection_energy)
        )
        self.assertGreater(float(absolute_diagnostic), 0.0)

        whole = area_weighted_flux_density_mean_square(
            torch.tensor([[[4.0]]]),
            torch.tensor([2.0]),
            torch.tensor([8.0]),
        )
        split = area_weighted_flux_density_mean_square(
            torch.tensor([[[2.0, 2.0]]]),
            torch.tensor([1.0, 1.0]),
            torch.tensor([8.0]),
        )
        torch.testing.assert_close(whole, split)

    def test_energy_difference_uses_cell_volume_weights(self) -> None:
        import torch

        from train_hccb_p418_spatiotemporal_regional_operator import (
            physics_difference_losses,
        )

        reference = {
            "fluid_internal_energy_flux_w": torch.zeros((1, 1, 1)),
            "solid_internal_heat_flux_w": torch.zeros((1, 1, 1)),
            "fluid_energy_w_m3": torch.zeros((1, 1, 2)),
            "solid_energy_w_m3": torch.zeros((1, 1, 2)),
        }
        predicted = SimpleNamespace(**{name: value.clone() for name, value in reference.items()})
        predicted.fluid_energy_w_m3[..., 0] = 2.0
        predicted.solid_energy_w_m3[..., 1] = 4.0
        condition = torch.tensor(
            [[0.05, 300.0, 4.85, 0.25, 900.0, 1.0e-6, 120000.0, 635.0]]
        )
        fluid_volume = torch.tensor([1.0, 3.0])
        solid_volume = torch.tensor([3.0, 1.0])
        _, energy = physics_difference_losses(
            predicted, reference, condition, fluid_volume, solid_volume,
            torch.ones(1), torch.ones(1)
        )
        # q''' is 1 W/m3 here. Fluid: 2^2*(1/4); solid: 4^2*(1/4).
        self.assertAlmostEqual(float(energy), 2.5)

    def test_physics_loss_is_device_independent_when_cuda_is_available(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the CPU/CUDA equivalence check")
        from train_hccb_p418_spatiotemporal_regional_operator import (
            physics_difference_losses,
        )

        reference_cpu = {
            "fluid_internal_energy_flux_w": torch.tensor([[[1.0, 2.0]]]),
            "solid_internal_heat_flux_w": torch.tensor([[[3.0, 4.0]]]),
            "fluid_energy_w_m3": torch.tensor([[[5.0, 6.0]]]),
            "solid_energy_w_m3": torch.tensor([[[7.0, 8.0]]]),
        }
        predicted_cpu = SimpleNamespace(
            **{
                name: value.clone().requires_grad_(True)
                for name, value in reference_cpu.items()
            }
        )
        predicted_cpu.fluid_internal_energy_flux_w = (
            predicted_cpu.fluid_internal_energy_flux_w + 0.25
        )
        predicted_cpu.solid_internal_heat_flux_w = (
            predicted_cpu.solid_internal_heat_flux_w - 0.50
        )
        predicted_cpu.fluid_energy_w_m3 = predicted_cpu.fluid_energy_w_m3 + 1.0
        predicted_cpu.solid_energy_w_m3 = predicted_cpu.solid_energy_w_m3 - 2.0
        condition_cpu = torch.tensor(
            [[0.05, 300.0, 4.85, 0.25, 900.0, 8.85, 120000.0, 635.0]]
        )
        fluid_volume_cpu = torch.tensor([1.0e-9, 2.0e-9])
        solid_volume_cpu = torch.tensor([2.0e-9, 1.0e-9])
        fluid_area_cpu = torch.tensor([1.0e-6, 2.0e-6])
        solid_area_cpu = torch.tensor([1.5e-6, 0.5e-6])
        cpu_losses = physics_difference_losses(
            predicted_cpu,
            reference_cpu,
            condition_cpu,
            fluid_volume_cpu,
            solid_volume_cpu,
            fluid_area_cpu,
            solid_area_cpu,
        )

        reference_cuda = {
            name: value.cuda() for name, value in reference_cpu.items()
        }
        predicted_cuda = SimpleNamespace(
            **{
                name: getattr(predicted_cpu, name).detach().cuda().requires_grad_(True)
                for name in reference_cpu
            }
        )
        cuda_losses = physics_difference_losses(
            predicted_cuda,
            reference_cuda,
            condition_cpu.cuda(),
            fluid_volume_cpu.cuda(),
            solid_volume_cpu.cuda(),
            fluid_area_cpu.cuda(),
            solid_area_cpu.cuda(),
        )
        for cpu_loss, cuda_loss in zip(cpu_losses, cuda_losses, strict=True):
            torch.testing.assert_close(cpu_loss, cuda_loss.cpu())

    def test_formal_data_only_is_a_same_architecture_ablation(self) -> None:
        text = TRAINER.read_text(encoding="utf-8")
        self.assertIn('"formal_data_only"', text)
        self.assertIn('"formal_factorized"', text)
        self.assertIn('expected_mode = "data_only"', text)
        self.assertIn('"factorized_static_spatial"', text)
        self.assertIn(
            'include_physics=args.physics_mode == "energy_and_flux"',
            text,
        )

    def test_checkpoint_selection_uses_the_declared_physics_objective(self) -> None:
        from train_hccb_p418_spatiotemporal_regional_operator import (
            energy_residual_rmse_ratio,
            loss_balancing_validation_score,
            validation_selection_score,
        )

        metrics = {
            "normalized_temperature_data_MSE": 0.25,
            "weighted_physics_objective": 0.75,
        }
        self.assertEqual(validation_selection_score(metrics, "data_only"), 0.25)
        self.assertEqual(validation_selection_score(metrics, "energy_and_flux"), 0.75)
        predicted, reference, ratio = energy_residual_rmse_ratio(4.0, 1.0)
        self.assertEqual(predicted, 2.0)
        self.assertEqual(reference, 1.0)
        self.assertEqual(ratio, 2.0)
        balancing_metrics = {
            "normalized_temperature_data_MSE": 0.25,
            "reference_edge_energy_flux_normalized_RMSE": 0.5,
            "projection_aware_energy_equation_normalized_RMSE": 1.0,
        }
        self.assertEqual(
            loss_balancing_validation_score(balancing_metrics),
            (0.25 + 0.25 + 1.0) / 3.0,
        )

    def test_checkpoint_round_trip_rejects_changed_contract(self) -> None:
        import torch

        from train_hccb_p418_spatiotemporal_regional_operator import (
            load_training_checkpoint,
            save_training_checkpoint,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_checkpoint.pt"
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
            contract = {"split_name": "pair_disjoint_stress_test", "epochs": 3}
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            save_training_checkpoint(
                path,
                contract=contract,
                next_epoch=1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_validation=2.0,
                best_state=best_state,
                history=[{"epoch": 1.0}],
                training_seconds=4.0,
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertNotIn("loss_balancer_state", payload)
            expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10.0)
            restored = load_training_checkpoint(
                path,
                contract=contract,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=torch.device("cpu"),
            )
            self.assertEqual(restored["next_epoch"], 1)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[name])
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_training_checkpoint(
                    path,
                    contract={"split_name": "changed", "epochs": 3},
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=torch.device("cpu"),
                )

    def test_checkpoint_round_trip_restores_relobralo_state(self) -> None:
        import torch

        from hccb_p418_fixed_flow_loss_balancing import (
            balanced_fixed_flow_loss,
            build_fixed_flow_loss_balancer,
        )
        from train_hccb_p418_spatiotemporal_regional_operator import (
            load_training_checkpoint,
            save_training_checkpoint,
        )

        source = ROOT / "parameters/hccb_p418_fixed_flow_loss_balancing_candidates.json"
        candidate_id = "relobralo_burgers_table_viii"
        groups = {
            "temperature_data": torch.tensor(2.0),
            "reference_edge_energy_flux": torch.tensor(0.5),
            "projection_aware_transient_energy": torch.tensor(1.25),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_checkpoint.pt"
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=3
            )
            balancer = build_fixed_flow_loss_balancer(
                source_path=source,
                candidate_id=candidate_id,
                seed=20260717,
            )
            balanced_fixed_flow_loss(**groups, balancer=balancer)
            contract = {
                "split_name": "pair_disjoint_stress_test",
                "loss_balancing": candidate_id,
            }
            save_training_checkpoint(
                path,
                contract=contract,
                next_epoch=1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_validation=1.0,
                best_state={
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                },
                history=[{"epoch": 1.0}],
                training_seconds=2.0,
                loss_balancer=balancer,
            )
            restored_balancer = build_fixed_flow_loss_balancer(
                source_path=source,
                candidate_id=candidate_id,
                seed=20260717,
            )
            load_training_checkpoint(
                path,
                contract=contract,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=torch.device("cpu"),
                loss_balancer=restored_balancer,
            )
            expected_loss, expected_weights = balanced_fixed_flow_loss(
                **groups, balancer=balancer
            )
            restored_loss, restored_weights = balanced_fixed_flow_loss(
                **groups, balancer=restored_balancer
            )
            torch.testing.assert_close(restored_loss, expected_loss)
            for name in expected_weights:
                torch.testing.assert_close(
                    torch.as_tensor(restored_weights[name]),
                    torch.as_tensor(expected_weights[name]),
                )

    def test_complete_curve_data_pipeline_runs(self) -> None:
        split_path = ROOT / "parameters/hccb_p418_step_response_splits.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))["splits"]["direction_down_test"]
        sequence_ids = split["train"] + split["validation"] + split["test"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequence_dir = root / "sequences"
            sequence_dir.mkdir()
            geometry = root / "regional_sequence_geometry.npz"
            np.savez_compressed(
                geometry,
                node_centroid_m=np.asarray(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
                    dtype=np.float32,
                ),
                node_volume_m3=np.ones(4, dtype=np.float32),
                node_type=np.asarray([0, 0, 1, 1], dtype=np.int8),
                edge_source=np.asarray([0, 1, 2, 3, 0, 2]),
                edge_target=np.asarray([1, 0, 3, 2, 2, 0]),
                edge_kind=np.asarray([0, 0, 1, 1, 2, 2]),
                edge_area_m2=np.ones(6, dtype=np.float32),
                edge_area_vector_m2=np.asarray(
                    [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
                    dtype=np.float32,
                ),
            )
            records = []
            time_s = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
            for index, sequence_id in enumerate(sequence_ids):
                state = np.zeros((3, 4, 5), dtype=np.float32)
                state[:, 0, 0] = 0.05
                state[:, 1, 0] = 0.04
                state[:, :2, 3] = 120000.0
                state[:, :, 4] = 300.0 + index + time_s[:, None]
                condition = np.asarray(
                    [0.05, 300.0, 4.85, 0.25, 900.0, 8.85, 120000.0, 635.0],
                    dtype=np.float32,
                )
                path = sequence_dir / f"{sequence_id}.npz"
                np.savez_compressed(
                    path,
                    sequence_id=np.asarray(sequence_id),
                    time_s=time_s,
                    condition_physical=condition,
                    state_physical=state,
                    fluid_internal_mass_flux_kg_s=np.asarray([1.0e-5], dtype=np.float32),
                    fluid_boundary_mass_flux_kg_s=np.asarray(
                        [-1.0e-5, 1.0e-5], dtype=np.float32
                    ),
                )
                records.append(
                    {
                        "sequence_id": sequence_id,
                        "sequence_file": f"sequences/{sequence_id}.npz",
                        "complete": True,
                    }
                )
            dataset = {
                "sequence_count": 12,
                "condition_names": [
                    "source_inlet_velocity_m_s",
                    "source_inlet_temperature_K",
                    "source_solid_heat_source_MW_m3",
                    "target_inlet_velocity_m_s",
                    "target_inlet_temperature_K",
                    "target_solid_heat_source_MW_m3",
                    "target_outlet_pressure_Pa",
                    "target_cooling_wall_temperature_K",
                ],
                "regional_geometry_file": geometry.name,
                "boundary_patch_names": {"fluid": [], "solid": []},
                "sequences": records,
            }
            index_path = root / "dataset_index.json"
            index_path.write_text(json.dumps(dataset), encoding="utf-8")
            output = root / "result"
            subprocess.run(
                [
                    sys.executable,
                    str(TRAINER),
                    "--dataset-index",
                    str(index_path),
                    "--splits",
                    str(split_path),
                    "--output-dir",
                    str(output),
                    "--run-role",
                    "smoke",
                    "--physics-mode",
                    "data_only",
                    "--epochs",
                    "1",
                    "--hidden-dim",
                    "8",
                    "--local-pre-iterations",
                    "1",
                    "--physics-attention-blocks",
                    "1",
                    "--local-post-iterations",
                    "1",
                    "--physics-attention-heads",
                    "2",
                    "--physics-slices",
                    "4",
                    "--temporal-layers",
                    "1",
                    "--temporal-heads",
                    "1",
                    "--temporal-node-chunk-size",
                    "2",
                    "--spatial-temporal-mode",
                    "factorized_static_spatial",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["split_case_counts"], {"train": 6, "validation": 3, "test": 3})
            self.assertEqual(summary["regional_node_count"], 4)
            self.assertEqual(summary["physics_mode"], "data_only")
            self.assertEqual(summary["physics_computation_device"], "cpu")
            self.assertEqual(
                summary["registered_solid_temperature_range_K"],
                [298.0, 1300.0],
            )
            self.assertEqual(
                summary["architecture"]["spatial_temporal_mode"],
                "factorized_static_spatial",
            )
            self.assertIn(
                "solid_maximum_temperature_history_RMSE_K",
                summary["metrics"]["test"],
            )
            self.assertIn(
                "solid_regional_hotspot_location_p95_error_m",
                summary["metrics"]["test"],
            )
            self.assertIn(
                "predicted_solid_temperature_outside_registered_range_fraction",
                summary["metrics"]["test"],
            )
            for role, count in (("train", 6), ("validation", 3), ("test", 3)):
                prediction_path = output / summary["temporal_temperature_prediction_files"][role]
                with np.load(prediction_path, allow_pickle=False) as prediction:
                    self.assertEqual(prediction["baseline_temperature_normalized"].shape, (count, 3, 4, 1))
                    self.assertEqual(prediction["target_temperature_normalized"].shape, (count, 3, 4, 1))
                    self.assertIn("fluid_internal_mass_flux_kg_s", prediction.files)
                    self.assertIn("fluid_boundary_mass_flux_kg_s", prediction.files)
                    self.assertIn("node_centroid_m", prediction.files)


if __name__ == "__main__":
    unittest.main()
