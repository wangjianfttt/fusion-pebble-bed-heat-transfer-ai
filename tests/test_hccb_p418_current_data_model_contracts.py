#!/usr/bin/env python3
"""Regression tests for the current P418 data and model descriptions."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class HCCBP418CurrentDataModelContractsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run(
            [sys.executable, "code/build_hccb_p418_cross_packing_plan.py"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            [sys.executable, "code/build_hccb_pore_resolved_cht_case_matrix.py"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            [sys.executable, "code/audit_hccb_pore_resolved_cht_case_matrix.py"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            [sys.executable, "code/audit_hccb_pore_resolved_cht_dataset_contract.py"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            [sys.executable, "code/audit_hccb_pore_resolved_ml_tensor_contract.py"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )

        cls.dataset = yaml.safe_load(
            (ROOT / "parameters/apd006_hccb_pore_resolved_cht_dataset_contract.yaml").read_text(
                encoding="utf-8"
            )
        )
        cls.model = yaml.safe_load(
            (ROOT / "parameters/apd006_hccb_pore_resolved_ml_tensor_contract.yaml").read_text(
                encoding="utf-8"
            )
        )
        with (ROOT / "data/apd006_hccb_pore_resolved_cht_case_matrix/case_matrix.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            cls.rows = list(csv.DictReader(handle))

    def test_current_paper_plan_is_exactly_60_plus_9_plus_9(self) -> None:
        counts = Counter(int(row["packing_seed"]) for row in self.rows)
        self.assertEqual(len(self.rows), 78)
        self.assertEqual(counts, Counter({101: 60, 202: 9, 303: 9}))
        packing = self.dataset["packing_realizations"]
        self.assertEqual(packing["current_paper_total_cases"], 78)
        self.assertEqual(self.model["upstream_contract"]["expected_cases_in_current_paper"], 78)

    def test_seed303_remains_unseen_during_architecture_selection(self) -> None:
        final_rows = [row for row in self.rows if int(row["packing_seed"]) == 303]
        self.assertEqual(len(final_rows), 9)
        self.assertTrue(
            all(row["fields_used_for_architecture_selection"] == "false" for row in final_rows)
        )
        final_split = self.model["pre_registered_splits"]["geometry_generalization"][-1]
        self.assertEqual(final_split["seed"], 303)
        self.assertIs(final_split["fields_read_during_selection"], False)

    def test_current_plan_does_not_restore_obsolete_parameter_ids_or_mesh_name(self) -> None:
        obsolete = {f"P{value:03d}" for value in range(51, 56)}
        for row in self.rows:
            self.assertFalse(obsolete.intersection(row["parameter_ids"].split(";")))
            self.assertEqual(row["parameter_ids"], "P418;P425;P426;P427")
            self.assertEqual(row["mesh_target"], "current_local_fine_mesh_not_published_G2")

    def test_prediction_targets_match_the_implemented_steady_models(self) -> None:
        self.assertEqual(
            self.model["prediction_targets"]["primary"],
            ["fluid_velocity_U", "fluid_pressure_p", "fluid_temperature_T", "solid_temperature_T"],
        )
        self.assertEqual(
            self.model["model_comparison"]["same_mesh_neural_models"],
            ["pinn_data_only", "pinn", "graph", "transolver"],
        )
        self.assertEqual(
            self.model["model_comparison"]["diffusion_role"]["present_use"],
            "transient_temperature_residual_refinement_after_deterministic_prediction",
        )

    def test_all_current_description_checks_pass(self) -> None:
        expected = {
            "case_matrix_audit.json": "p418_60_plus_9_plus_9_case_plan_checked",
            "dataset_contract_audit.json": "hccb_p418_dataset_description_current",
            "ml_tensor_contract_audit.json": "hccb_p418_model_input_target_description_current",
        }
        root = ROOT / "results/apd006_hccb_pore_resolved_cht"
        for filename, status in expected.items():
            payload = json.loads((root / filename).read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], status)
            self.assertTrue(all(payload["checks"].values()))


if __name__ == "__main__":
    unittest.main()
