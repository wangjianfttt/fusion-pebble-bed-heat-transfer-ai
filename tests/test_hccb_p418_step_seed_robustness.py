#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/summarize_hccb_p418_step_seed_robustness.py"
SPLITS = ROOT / "parameters/hccb_p418_step_response_splits.json"
SPLIT_NAME = "pair_disjoint_stress_test"
SEEDS = (20260717, 20260718, 20260719)


class P418StepSeedRobustnessTest(unittest.TestCase):
    def test_summarizes_three_training_seeds_and_rejects_wrong_seed(self) -> None:
        split = json.loads(SPLITS.read_text(encoding="utf-8"))["splits"][SPLIT_NAME]
        split_ids = {role: list(map(str, split[role])) for role in ("train", "validation", "test")}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for offset, seed in enumerate(SEEDS):
                suffix = "" if seed == SEEDS[0] else f"_seed{seed}"
                values = {
                    "observable": 2.0 + offset,
                    "data_only": 5.0 + offset,
                    "physics": 3.0 + offset,
                    "low_rank": 2.5 + offset,
                    "diffusion": 2.2 + offset,
                }
                summaries = {
                    f"transformer_{SPLIT_NAME}{suffix}": {
                        "status": "completed_p418_physical_step_response_transformer_formal",
                        "split_name": SPLIT_NAME,
                        "seed": seed,
                        "split_case_ids": split_ids,
                        "new_physical_parameters": [],
                        "test_mean_rmse_by_target": {
                            "outlet_temperature_K": values["observable"]
                        },
                    },
                    f"regional_graph_transformer_bounded_data_only_{SPLIT_NAME}{suffix}": {
                        "status": "completed_p418_spatiotemporal_regional_operator",
                        "split_name": SPLIT_NAME,
                        "seed": seed,
                        "split_case_ids": split_ids,
                        "new_physical_parameters": [],
                        "metrics": {
                            "test": {"solid_temperature_RMSE_K": values["data_only"]}
                        },
                    },
                    f"regional_graph_transformer_bounded_physics_{SPLIT_NAME}{suffix}": {
                        "status": "completed_p418_spatiotemporal_regional_operator",
                        "split_name": SPLIT_NAME,
                        "seed": seed,
                        "split_case_ids": split_ids,
                        "new_physical_parameters": [],
                        "metrics": {
                            "test": {"solid_temperature_RMSE_K": values["physics"]}
                        },
                    },
                    f"low_rank_temperature_residual_{SPLIT_NAME}{suffix}": {
                        "status": "completed_p418_low_rank_temperature_residual",
                        "split_name": SPLIT_NAME,
                        "upstream_training_seed": seed,
                        "split_case_ids": split_ids,
                        "new_physical_parameters": [],
                        "metrics": {
                            "test": {"solid_temperature_RMSE_K": values["low_rank"]}
                        },
                    },
                    f"temporal_diffusion_{SPLIT_NAME}{suffix}": {
                        "status": "completed_p418_temporal_temperature_diffusion",
                        "split_name": SPLIT_NAME,
                        "seed": seed,
                        "upstream_training_seed": seed,
                        "split_case_ids": split_ids,
                        "new_physical_parameters": [],
                        "metrics": {
                            "test": {
                                "diffusion_refined_solid_temperature_RMSE_K": values[
                                    "diffusion"
                                ]
                            }
                        },
                    },
                }
                for directory, summary in summaries.items():
                    path = root / directory
                    path.mkdir()
                    (path / "summary.json").write_text(
                        json.dumps(summary), encoding="utf-8"
                    )

            output = root / "seed_robustness"
            command = [
                "python3",
                str(SCRIPT),
                "--result-dir",
                str(root),
                "--splits",
                str(SPLITS),
                "--split-name",
                SPLIT_NAME,
                "--primary-seed",
                str(SEEDS[0]),
                "--seeds",
                *map(str, SEEDS),
                "--output-dir",
                str(output),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["seeds"], list(SEEDS))
            self.assertEqual(len(summary["models"]), 5)
            rows = list(csv.DictReader((output / "seed_summary.csv").open(encoding="utf-8")))
            physics = next(row for row in rows if row["model"] == "graph_transformer_energy_flux")
            self.assertEqual(physics["seed_count"], "3")
            self.assertAlmostEqual(float(physics["mean_K"]), 4.0)
            self.assertAlmostEqual(float(physics["sample_std_K"]), 1.0)

            transformer = (
                root / f"transformer_{SPLIT_NAME}" / "summary.json"
            )
            payload = json.loads(transformer.read_text(encoding="utf-8"))
            payload["status"] = "incomplete"
            transformer.write_text(json.dumps(payload), encoding="utf-8")
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("unfinished", rejected.stderr)
            payload["status"] = "completed_p418_physical_step_response_transformer_formal"
            payload["test_mean_rmse_by_target"]["outlet_temperature_K"] = float("nan")
            transformer.write_text(json.dumps(payload), encoding="utf-8")
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("invalid error value", rejected.stderr)
            payload["test_mean_rmse_by_target"]["outlet_temperature_K"] = 2.0
            transformer.write_text(json.dumps(payload), encoding="utf-8")

            bad = root / f"temporal_diffusion_{SPLIT_NAME}_seed{SEEDS[-1]}" / "summary.json"
            payload = json.loads(bad.read_text(encoding="utf-8"))
            payload["upstream_training_seed"] = 1
            bad.write_text(json.dumps(payload), encoding="utf-8")
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("different seeds", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
