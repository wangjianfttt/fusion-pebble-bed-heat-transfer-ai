#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_p418_fused_chain_table import (  # noqa: E402
    collect_records,
    render_table,
    render_text,
)


def payload(split_name: str, strict: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "status": "completed_p418_fused_chain_comparison",
        "split_name": split_name,
        "test_curve_count": 4,
        "exact_initial_graph_transformer_solid_temperature_RMSE_K": 2.0,
        "steady_PINN_initial_graph_transformer_solid_temperature_RMSE_K": 5.0,
        "steady_PINN_initial_graph_transformer_energy_RMSE": 0.4,
        "fused_diffusion_solid_temperature_RMSE_K": 4.0,
        "fused_diffusion_energy_RMSE": 0.3,
        "strict_end_to_end_group": None,
        "registered_steady_PINN_unique_endpoint_count": 6,
        "complete_chain_timing": {
            "warm_start_complete_chain_inference_seconds_per_curve": 2.0,
            "cold_start_complete_chain_inference_seconds_per_curve": 2.5,
        },
        "complete_chain_model_cost": {
            "complete_chain_model_parameter_count": 505,
            "complete_chain_training_seconds": 7200.0,
        },
    }
    if strict:
        result["strict_end_to_end_group"] = {
            "curve_count": 2,
            "exact_initial_mean_solid_temperature_RMSE_K": 3.0,
            "steady_PINN_initial_mean_solid_temperature_RMSE_K": 6.0,
            "fused_diffusion_solid_temperature_RMSE_K": 5.0,
            "steady_PINN_initial_graph_transformer_energy_RMSE": 0.5,
            "fused_diffusion_energy_RMSE": 0.4,
        }
    return result


class FusedChainTableTest(unittest.TestCase):
    def write(self, root: Path, split: str, data: dict[str, object]) -> None:
        path = root / split / "fused_chain_summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_strict_and_all_trajectory_rows_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = ("direction_down_test", "pair_disjoint_stress_test")
            self.write(root, splits[0], payload(splits[0], strict=False))
            self.write(root, splits[1], payload(splits[1], strict=True))
            records = collect_records(root, splits)
            self.assertEqual(len(records), 3)
            strict = [
                record
                for record in records
                if record["subset"] == "both steady endpoints unseen"
            ]
            self.assertEqual(len(strict), 1)
            self.assertEqual(strict[0]["curve_count"], 2)
            self.assertTrue(strict[0]["diffusion_improves_temperature_and_energy"])
            latex = render_table(records)
            self.assertIn("state-assisted reference", latex)
            self.assertIn("both steady endpoints unseen", latex)
            self.assertIn("Endpoint-pair split", latex)
            self.assertIn("Warm / cold", latex)
            self.assertIn("Unique endpoints", latex)
            text = render_text(records)
            self.assertIn("deployable steady-PINN--graph--Transformer chain", text)
            self.assertIn("2~K", text)
            self.assertIn("range to 5~K", text)
            self.assertIn("gave 4~K", text)
            self.assertIn("were 6 and 5~K", text)
            self.assertIn("Diffusion improved both quantities in 1 of 1", text)
            self.assertIn("projection-aware energy RMSEs", text)
            self.assertNotIn("registered trajectory", text)

    def test_missing_strict_result_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = "direction_down_test"
            self.write(root, split, payload(split, strict=False))
            with self.assertRaisesRegex(ValueError, "no strict end-to-end"):
                collect_records(root, (split,))

    def test_incomplete_summary_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = "pair_disjoint_stress_test"
            data = payload(split, strict=True)
            data["status"] = "draft"
            self.write(root, split, data)
            with self.assertRaisesRegex(ValueError, "incomplete fused-chain"):
                collect_records(root, (split,))


if __name__ == "__main__":
    unittest.main()
