#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import sys
import unittest
from decimal import Decimal
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from hccb_p418_spatiotemporal_regional_operator import FORMAL_ARCHITECTURE  # noqa: E402
from hccb_p418_temporal_temperature_diffusion import (  # noqa: E402
    FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE,
)
from hccb_source_backed_thermophysical import OPENFOAM_TSTD_K  # noqa: E402
from train_hccb_p418_spatiotemporal_regional_operator import (  # noqa: E402
    P418_TRANSIENT_PARAMETER_IDS,
)


class P418SourceContractTest(unittest.TestCase):
    def test_openfoam_enthalpy_reference_is_registered_as_software_constant(self) -> None:
        with (ROOT / "parameters/hccb_p418_model_numerical_settings.csv").open(
            newline="", encoding="utf-8-sig"
        ) as stream:
            rows = list(csv.DictReader(stream))
        matches = [
            row
            for row in rows
            if row["setting"] == "enthalpy_reference_temperature_K"
        ]
        self.assertEqual(len(matches), 1)
        row = matches[0]
        self.assertEqual(float(row["value"]), OPENFOAM_TSTD_K)
        self.assertEqual(row["is_physical_parameter"], "no")
        source = ROOT / row["source_path"]
        implementation = ROOT / row["implementation_path"]
        self.assertTrue(source.is_file())
        self.assertTrue(implementation.is_file())
        self.assertIn("Tstd    298.15 [K]", source.read_text(encoding="utf-8"))

    def test_every_transient_physical_parameter_is_in_source_table(self) -> None:
        with (ROOT / "parameters/hccb_p418_physical_parameter_sources.csv").open(
            newline="", encoding="utf-8-sig"
        ) as stream:
            ids = {row["parameter_id"] for row in csv.DictReader(stream)}
        self.assertTrue(set(P418_TRANSIENT_PARAMETER_IDS).issubset(ids))
        self.assertTrue({"P428", "P429", "P430", "P431"}.issubset(ids))

    def test_spatiotemporal_architecture_matches_registered_mgnt_values(self) -> None:
        contract_path = (
            ROOT / "parameters/hccb_p418_mgnt_temporal_pino_contract.yaml"
        )
        contract = contract_path.read_text(encoding="utf-8")
        expected = {
            "hidden_dim": 64,
            "local_pre_iterations": 2,
            "physics_attention_blocks": 2,
            "local_post_iterations": 2,
            "physics_attention_heads": 4,
            "physics_slices": 128,
            "temporal_layers": 3,
            "temporal_heads": 1,
            "leaky_relu_negative_slope": 0.01,
        }
        self.assertEqual(FORMAL_ARCHITECTURE, expected)
        for text in (
            "local_pre_iterations: 2",
            "physics_attention_blocks: 2",
            "local_post_iterations: 2",
            "physics_attention_heads: 4",
            "physics_slices: 128",
            "hidden_dim: 64",
            "temporal_layers: 3",
            "temporal_heads: 1",
            "leaky_relu_negative_slope: 0.01",
            "output_time_count: 56",
            "all 56 physical output times",
        ):
            self.assertIn(text, contract)
        self.assertIn("Earlier 37-time calculations", contract)
        payload = yaml.safe_load(contract)
        plan = json.loads(
            (ROOT / payload["physical_inputs"]["transient_plan"]).read_text()
        )
        output_times: set[Decimal] = set()
        for segment in plan["numerical_time_design"]["field_write_schedule"]:
            start = Decimal(str(segment["start_s"]))
            end = Decimal(str(segment["end_s"]))
            interval = Decimal(str(segment["interval_s"]))
            value = start
            while value <= end:
                output_times.add(value)
                value += interval
        self.assertEqual(len(output_times), 56)
        self.assertEqual(
            payload["p418_transient_representation"]["output_time_count"],
            len(output_times),
        )

    def test_temporal_diffusion_keeps_registered_transolver_and_refiner_values(self) -> None:
        payload = json.loads(
            (ROOT / "parameters/hccb_p418_ai_architecture_sources.json").read_text()
        )
        transolver = next(
            row for row in payload["architectures"] if row["name"] == "Transolver"
        )["source_settings"]
        refiner = next(
            row
            for row in payload["architectures"]
            if row["name"].startswith("PDE-Refiner")
        )["source_settings"]
        actual = FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE
        self.assertEqual(actual["hidden_dim"], transolver["hidden_size"])
        self.assertEqual(actual["spatial_layers"], transolver["layers"])
        self.assertEqual(actual["spatial_attention_heads"], transolver["attention_heads"])
        self.assertEqual(actual["physics_slices"], transolver["physics_slices"])
        self.assertEqual(actual["num_refinement_steps"], refiner["num_refinement_steps"])
        self.assertEqual(
            actual["minimum_noise_standard_deviation"],
            refiner["minimum_noise_standard_deviation"],
        )

    def test_chinese_model_note_uses_formal_56_time_resource_measurements(self) -> None:
        note = (
            ROOT / "parameters/hccb_p418_transient_model_comparison_CN.md"
        ).read_text(encoding="utf-8")
        for value in ("12.88 s", "15.13 s", "7.74 s", "14.03 s"):
            self.assertIn(value, note)
        self.assertNotIn("用时9.10 s", note)
        self.assertNotIn("用时12.49 s", note)


if __name__ == "__main__":
    unittest.main()
