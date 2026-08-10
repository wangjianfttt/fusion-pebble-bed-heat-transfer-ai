#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from hccb_p418_conservative_mixed_operator import (  # noqa: E402
    ConservativeRegionalOutput,
    RegionalEnergyFluxGeometry,
    RegionalMassFluxGeometry,
    regional_energy_balance,
    regional_mass_balance,
)
from hccb_p418_comparison_contract import numerical_state_sha256  # noqa: E402
from train_hccb_p418_conservative_mixed_operator import (  # noqa: E402
    STEADY_CONSERVATIVE_LOSS_WEIGHTS,
    aggregate_batch_metrics,
    grouped_conservative_loss,
    load_steady_training_checkpoint,
    normalized_conditions,
    save_steady_training_checkpoint,
    selected_loss_term_names,
)
from train_hccb_p418_regional_operator import FieldScales  # noqa: E402


class P418ConservativeMixedOperatorTest(unittest.TestCase):
    def test_constant_training_inputs_remain_zero_for_held_out_cases(self) -> None:
        scales = FieldScales(
            condition_mean=np.array([0.15, 600.0, 4.85e6, 120000.0, 635.0]),
            condition_std=np.array([0.1, 100.0, 0.0, 0.0, 0.0]),
            velocity_mean=np.zeros(3),
            velocity_std=np.ones(3),
            pressure_mean=0.0,
            pressure_std=1.0,
            fluid_temperature_mean=300.0,
            fluid_temperature_std=1.0,
            solid_temperature_mean=500.0,
            solid_temperature_std=1.0,
        )
        physical = np.array(
            [
                [0.05, 500.0, 4.85e6, 120000.0, 635.0],
                [0.25, 900.0, 8.85e6, 120000.0, 635.0],
            ]
        )
        normalized = normalized_conditions(physical, scales)
        self.assertTrue(
            np.allclose(
                normalized,
                [[-1.0, -1.0, 0.0, 0.0, 0.0], [1.0, 3.0, 0.0, 0.0, 0.0]],
            )
        )

    def test_training_checkpoint_restores_optimizer_scheduler_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "training_checkpoint.pt"
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=1.0e-3, total_steps=2
            )
            optimizer.zero_grad(set_to_none=True)
            model(torch.ones((1, 2))).sum().backward()
            optimizer.step()
            scheduler.step()
            saved_parameters = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            order_rng = np.random.default_rng(0)
            order_rng.permutation(5)
            contract = {"split": {"train": ["a"], "validation": ["b"]}, "epochs": 2}
            save_steady_training_checkpoint(
                checkpoint,
                contract=contract,
                next_epoch=1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                training_rng=order_rng,
                best_validation=0.4,
                best_epoch=1,
                history=[{"epoch": 1}],
                training_seconds=2.0,
                optimization_seconds=1.5,
                validation_seconds=0.5,
                update_index=1,
            )
            expected_order = order_rng.permutation(8)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10.0)
            order_rng.permutation(11)
            restored = load_steady_training_checkpoint(
                checkpoint,
                contract=contract,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                training_rng=order_rng,
                device=torch.device("cpu"),
            )
            self.assertEqual(restored["next_epoch"], 1)
            self.assertEqual(restored["best_epoch"], 1)
            self.assertEqual(restored["update_index"], 1)
            self.assertEqual(scheduler.last_epoch, 1)
            self.assertTrue(np.array_equal(order_rng.permutation(8), expected_order))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, saved_parameters[name]))
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_steady_training_checkpoint(
                    checkpoint,
                    contract={"split": {"train": ["different"]}, "epochs": 2},
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    training_rng=order_rng,
                    device=torch.device("cpu"),
                )

    def test_numerical_state_hash_changes_with_a_weight(self) -> None:
        first = {"weight": torch.tensor([[1.0, 2.0]]), "bias": torch.tensor([3.0])}
        same = {"weight": torch.tensor([[1.0, 2.0]]), "bias": torch.tensor([3.0])}
        changed = {"weight": torch.tensor([[1.0, 2.1]]), "bias": torch.tensor([3.0])}
        self.assertEqual(numerical_state_sha256(first), numerical_state_sha256(same))
        self.assertNotEqual(
            numerical_state_sha256(first), numerical_state_sha256(changed)
        )

    def test_batch_metrics_reconstruct_complete_epoch_rmse(self) -> None:
        combined = aggregate_batch_metrics(
            [
                (2, {"state_normalized_rmse": 1.0, "state_channel_rmse": [1.0] * 6, "total_loss": 1.0}),
                (1, {"state_normalized_rmse": 2.0, "state_channel_rmse": [2.0] * 6, "total_loss": 4.0}),
            ]
        )
        expected_rmse = (2.0) ** 0.5
        self.assertAlmostEqual(combined["state_normalized_rmse"], expected_rmse)
        self.assertTrue(
            all(abs(value - expected_rmse) < 1e-12 for value in combined["state_channel_rmse"])
        )
        self.assertAlmostEqual(combined["total_loss"], 2.0)

    def test_paired_pinn_differs_only_by_balance_terms(self) -> None:
        data_only = set(
            selected_loss_term_names(
                energy_enabled=True, physics_constrained=False
            )
        )
        constrained = set(
            selected_loss_term_names(
                energy_enabled=True, physics_constrained=True
            )
        )
        self.assertEqual(
            data_only,
            {"state", "internal_mass", "boundary_mass", "internal_energy", "boundary_energy"},
        )
        self.assertEqual(constrained - data_only, {"continuity", "energy_balance"})

    def test_grouped_loss_keeps_shared_supervised_coefficients(self) -> None:
        values = {
            "state": torch.tensor(2.0),
            "internal_mass": torch.tensor(3.0),
            "boundary_mass": torch.tensor(5.0),
            "continuity": torch.tensor(7.0),
            "internal_energy": torch.tensor(11.0),
            "boundary_energy": torch.tensor(13.0),
            "energy_balance": torch.tensor(17.0),
        }
        data_total, data_groups, data_weights = grouped_conservative_loss(
            values,
            energy_enabled=True,
            physics_constrained=False,
            weights=STEADY_CONSERVATIVE_LOSS_WEIGHTS,
        )
        physics_total, physics_groups, physics_weights = grouped_conservative_loss(
            values,
            energy_enabled=True,
            physics_constrained=True,
            weights=STEADY_CONSERVATIVE_LOSS_WEIGHTS,
        )
        self.assertAlmostEqual(float(data_groups["face_flux"]), 8.0)
        self.assertAlmostEqual(float(data_total), 18.0)
        self.assertAlmostEqual(float(physics_groups["physics_balance"]), 12.0)
        self.assertAlmostEqual(float(physics_total), 30.0)
        self.assertEqual(data_weights, {name: physics_weights[name] for name in data_weights})
        self.assertEqual(
            set(physics_weights) - set(data_weights),
            {"continuity", "energy_balance"},
        )

    def test_grouped_loss_does_not_dilute_data_terms_when_physics_is_added(self) -> None:
        state = torch.tensor(1.0, requires_grad=True)
        face = torch.tensor(1.0, requires_grad=True)
        continuity = torch.tensor(1.0, requires_grad=True)
        values = {
            "state": state,
            "internal_mass": face,
            "boundary_mass": face,
            "continuity": continuity,
        }
        data_total, _, _ = grouped_conservative_loss(
            values,
            energy_enabled=False,
            physics_constrained=False,
            weights=STEADY_CONSERVATIVE_LOSS_WEIGHTS,
        )
        data_gradient = torch.autograd.grad(data_total, (state, face), retain_graph=True)
        physics_total, _, _ = grouped_conservative_loss(
            values,
            energy_enabled=False,
            physics_constrained=True,
            weights=STEADY_CONSERVATIVE_LOSS_WEIGHTS,
        )
        physics_gradient = torch.autograd.grad(physics_total, (state, face))
        self.assertEqual(data_gradient, physics_gradient)

    def test_regional_mass_balance_is_differentiable(self) -> None:
        internal = torch.tensor([[5.0, -7.0]], requires_grad=True)
        boundary = torch.tensor([[-5.0, 7.0, -2.0]], requires_grad=True)
        output = ConservativeRegionalOutput(
            regional_state=torch.zeros((1, 3, 5)),
            internal_mass_flow_kg_s=internal,
            boundary_mass_flow_kg_s=boundary,
        )
        geometry = RegionalMassFluxGeometry(
            internal_owner=torch.tensor([0, 1]),
            internal_neighbour=torch.tensor([2, 2]),
            internal_face_centroid_m=torch.zeros((2, 3)),
            internal_face_area_vector_m2=torch.zeros((2, 3)),
            internal_face_area_m2=torch.ones(2),
            boundary_owner=torch.tensor([0, 1, 2]),
            boundary_patch=torch.tensor([0, 1, 1]),
            boundary_face_centroid_m=torch.zeros((3, 3)),
            boundary_face_area_vector_m2=torch.zeros((3, 3)),
            boundary_face_area_m2=torch.ones(3),
            patch_count=2,
        )
        balance = regional_mass_balance(output, geometry, 3)
        self.assertTrue(torch.allclose(balance, torch.zeros_like(balance)))
        loss = balance.square().sum() + internal.square().sum() + boundary.square().sum()
        gradients = torch.autograd.grad(loss, (internal, boundary))
        self.assertTrue(all(torch.all(torch.isfinite(value)) for value in gradients))

    def test_regional_energy_balance_uses_same_oriented_edge_once(self) -> None:
        internal_energy = torch.tensor([[2.0, 3.0]], requires_grad=True)
        boundary_energy = torch.tensor([[-2.0, 3.0]], requires_grad=True)
        output = ConservativeRegionalOutput(
            regional_state=torch.zeros((1, 3, 5)),
            internal_mass_flow_kg_s=torch.zeros((1, 1)),
            boundary_mass_flow_kg_s=torch.zeros((1, 1)),
            internal_energy_flow_W=internal_energy,
            boundary_energy_flow_W=boundary_energy,
        )
        geometry = RegionalEnergyFluxGeometry(
            internal_owner=torch.tensor([0, 1]),
            internal_neighbour=torch.tensor([1, 2]),
            internal_kind=torch.tensor([0, 2]),
            internal_face_centroid_m=torch.zeros((2, 3)),
            internal_face_area_vector_m2=torch.zeros((2, 3)),
            internal_face_area_m2=torch.ones(2),
            boundary_owner=torch.tensor([0, 2]),
            boundary_kind=torch.tensor([0, 1]),
            boundary_face_centroid_m=torch.zeros((2, 3)),
            boundary_face_area_vector_m2=torch.zeros((2, 3)),
            boundary_face_area_m2=torch.ones(2),
            internal_kind_count=3,
            boundary_kind_count=2,
        )
        source = torch.tensor([[0.0, 1.0, 0.0]])
        balance = regional_energy_balance(output, geometry, source)
        self.assertTrue(torch.allclose(balance, torch.zeros_like(balance)))
        gradients = torch.autograd.grad(balance.square().sum() + internal_energy.square().sum(), internal_energy)
        self.assertTrue(torch.all(torch.isfinite(gradients[0])))


if __name__ == "__main__":
    unittest.main()
