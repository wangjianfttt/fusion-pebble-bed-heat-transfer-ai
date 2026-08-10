#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
SOURCE = ROOT / "parameters/hccb_p418_fixed_flow_loss_balancing_candidates.json"


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required")
class FixedFlowLossBalancingTest(unittest.TestCase):
    def test_registered_fixed_candidate_reproduces_legacy_5_1_1_loss(self) -> None:
        import torch

        from hccb_p418_fixed_flow_loss_balancing import (
            balanced_fixed_flow_loss,
            build_fixed_flow_loss_balancer,
        )

        balancer = build_fixed_flow_loss_balancer(
            source_path=SOURCE,
            candidate_id="fixed_registered_5_1_1",
            seed=20260717,
        )
        temperature = torch.tensor(2.0, requires_grad=True)
        edge = torch.tensor(3.0, requires_grad=True)
        energy = torch.tensor(5.0, requires_grad=True)
        total, weights = balanced_fixed_flow_loss(
            temperature_data=temperature,
            reference_edge_energy_flux=edge,
            projection_aware_transient_energy=energy,
            balancer=balancer,
        )
        self.assertEqual(float(total.detach()), 18.0)
        self.assertEqual(float(weights["temperature_data"].detach()), 5.0)
        self.assertEqual(
            float(weights["reference_edge_energy_flux"].detach()), 1.0
        )
        self.assertEqual(
            float(weights["projection_aware_transient_energy"].detach()), 1.0
        )
        total.backward()
        self.assertEqual(float(temperature.grad), 5.0)
        self.assertEqual(float(edge.grad), 1.0)
        self.assertEqual(float(energy.grad), 1.0)

    def test_relobralo_candidate_is_deterministic_and_resumable(self) -> None:
        import torch

        from hccb_p418_fixed_flow_loss_balancing import (
            balanced_fixed_flow_loss,
            build_fixed_flow_loss_balancer,
        )

        original = build_fixed_flow_loss_balancer(
            source_path=SOURCE,
            candidate_id="relobralo_burgers_table_viii",
            seed=20260717,
        )
        groups = [
            (torch.tensor(1.0), torch.tensor(10.0), torch.tensor(100.0)),
            (torch.tensor(0.8), torch.tensor(12.0), torch.tensor(90.0)),
            (torch.tensor(0.6), torch.tensor(9.0), torch.tensor(110.0)),
        ]
        for temperature, edge, energy in groups[:2]:
            balanced_fixed_flow_loss(
                temperature_data=temperature,
                reference_edge_energy_flux=edge,
                projection_aware_transient_energy=energy,
                balancer=original,
            )
        state = original.state_dict()
        resumed = build_fixed_flow_loss_balancer(
            source_path=SOURCE,
            candidate_id="relobralo_burgers_table_viii",
            seed=20260717,
        )
        resumed.load_state_dict(state)
        original_total, original_weights = balanced_fixed_flow_loss(
            temperature_data=groups[2][0],
            reference_edge_energy_flux=groups[2][1],
            projection_aware_transient_energy=groups[2][2],
            balancer=original,
        )
        resumed_total, resumed_weights = balanced_fixed_flow_loss(
            temperature_data=groups[2][0],
            reference_edge_energy_flux=groups[2][1],
            projection_aware_transient_energy=groups[2][2],
            balancer=resumed,
        )
        torch.testing.assert_close(original_total, resumed_total)
        for name in original_weights:
            torch.testing.assert_close(
                original_weights[name], resumed_weights[name]
            )

    def test_candidates_are_literature_recorded_and_test_is_read_once(self) -> None:
        source = json.loads(SOURCE.read_text(encoding="utf-8"))
        self.assertEqual(source["physical_parameter_status"]["new_physical_parameters"], [])
        self.assertEqual(
            source["primary_source"]["doi"],
            "10.1016/j.cma.2025.117914",
        )
        candidates = source["formal_candidates"]
        self.assertEqual(len(candidates), 4)
        self.assertEqual(
            {row["candidate_id"] for row in candidates},
            {
                "fixed_registered_5_1_1",
                "relobralo_burgers_table_viii",
                "relobralo_kirchhoff_table_viii",
                "relobralo_helmholtz_table_viii",
            },
        )
        self.assertIn(
            "once",
            source["selection_protocol"]["independent_test"].lower(),
        )

    def test_unknown_candidate_is_rejected(self) -> None:
        from hccb_p418_fixed_flow_loss_balancing import (
            load_fixed_flow_candidate,
        )

        with self.assertRaisesRegex(ValueError, "expected one"):
            load_fixed_flow_candidate(SOURCE, "invented")


if __name__ == "__main__":
    unittest.main()
