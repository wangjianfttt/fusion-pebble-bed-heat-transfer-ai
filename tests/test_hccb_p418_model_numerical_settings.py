#!/usr/bin/env python3

from __future__ import annotations

import ast
import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "parameters/hccb_p418_model_numerical_settings.csv"


def literal_assignment(path: Path, name: str) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                value = ast.literal_eval(node.value)
                if not isinstance(value, dict):
                    raise AssertionError(f"{name} in {path} is not a dictionary")
                return value
    raise AssertionError(f"{name} is missing from {path}")


class P418ModelNumericalSettingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with TABLE.open(newline="", encoding="utf-8") as handle:
            cls.rows = list(csv.DictReader(handle))
        cls.by_key = {(row["model"], row["setting"]): row for row in cls.rows}

    def test_sources_and_implementations_exist(self) -> None:
        self.assertGreaterEqual(len(self.rows), 40)
        allowed = {
            "published_architecture",
            "published_component_adaptation",
            "official_code_architecture",
            "official_code_constant",
            "official_code_training",
            "published_algorithm",
            "finite_volume_definition",
            "data_derived",
            "measured_compute_setting",
            "predeclared_baseline",
            "predeclared_numerical_scan",
            "predeclared_selection_rule",
            "problem_geometry",
            "project_adaptation",
            "source_backed_output_parameterization",
            "official_OpenFOAM13_software_constant",
        }
        for row in self.rows:
            self.assertIn(row["setting_type"], allowed)
            self.assertEqual(row["is_physical_parameter"], "no")
            self.assertTrue(row["primary_source"])
            for value in row["source_path"].split(";"):
                self.assertTrue((ROOT / value).exists(), value)
            for value in row["implementation_path"].split(";"):
                self.assertTrue((ROOT / value).is_file(), value)
            self.assertTrue(row["explanation_cn"])

    def test_formal_constants_match_the_source_table(self) -> None:
        observable_architecture = literal_assignment(
            ROOT / "code/train_hccb_p418_transient_observable_transformer.py",
            "FORMAL_ARCHITECTURE",
        )
        observable_training = literal_assignment(
            ROOT / "code/train_hccb_p418_transient_observable_transformer.py",
            "FORMAL_TRAINING",
        )
        graph_architecture = literal_assignment(
            ROOT / "code/hccb_p418_spatiotemporal_regional_operator.py",
            "FORMAL_ARCHITECTURE",
        )
        graph_training = literal_assignment(
            ROOT / "code/train_hccb_p418_spatiotemporal_regional_operator.py",
            "FORMAL_TRAINING",
        )
        diffusion_architecture = literal_assignment(
            ROOT / "code/hccb_p418_temporal_temperature_diffusion.py",
            "FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE",
        )
        diffusion_training = literal_assignment(
            ROOT / "code/train_hccb_p418_temporal_temperature_diffusion.py",
            "FORMAL_TRAINING",
        )
        expected = {
            ("工程量时间Transformer", "hidden_width"): observable_architecture["d_model"],
            ("工程量时间Transformer", "layers"): observable_architecture["layers"],
            ("工程量时间Transformer", "attention_heads"): observable_architecture["heads"],
            ("工程量时间Transformer", "epochs"): observable_training["epochs"],
            ("工程量时间Transformer", "effective_batch_size"): observable_training["batch_size"],
            ("工程量时间Transformer", "learning_rate"): observable_training["learning_rate"],
            ("工程量时间Transformer", "weight_decay"): observable_training["weight_decay"],
            ("图-Transformer", "hidden_width"): graph_architecture["hidden_dim"],
            ("图-Transformer", "preprocessor_mpnn_iterations"): graph_architecture["local_pre_iterations"],
            ("图-Transformer", "physics_attention_blocks"): graph_architecture["physics_attention_blocks"],
            ("图-Transformer", "refinement_mpnn_iterations"): graph_architecture["local_post_iterations"],
            ("图-Transformer", "physics_attention_heads"): graph_architecture["physics_attention_heads"],
            ("图-Transformer", "physics_slices"): graph_architecture["physics_slices"],
            ("图-Transformer", "temporal_layers"): graph_architecture["temporal_layers"],
            ("图-Transformer", "temporal_heads"): graph_architecture["temporal_heads"],
            ("图-Transformer", "epochs"): graph_training["epochs"],
            ("图-Transformer", "learning_rate"): graph_training["learning_rate"],
            ("图-Transformer", "weight_decay"): graph_training["weight_decay"],
            ("图-Transformer", "temperature_data_weight"): graph_training["data_weight"],
            ("图-Transformer", "edge_flux_weight"): graph_training["edge_flux_weight"],
            ("图-Transformer", "energy_balance_weight"): graph_training["energy_weight"],
            ("扩散剩余误差修正", "hidden_width"): diffusion_architecture["hidden_dim"],
            ("扩散剩余误差修正", "spatial_layers"): diffusion_architecture["spatial_layers"],
            ("扩散剩余误差修正", "spatial_attention_heads"): diffusion_architecture["spatial_attention_heads"],
            ("扩散剩余误差修正", "physics_slices"): diffusion_architecture["physics_slices"],
            ("扩散剩余误差修正", "temporal_layers"): diffusion_architecture["temporal_layers"],
            ("扩散剩余误差修正", "temporal_heads"): diffusion_architecture["temporal_heads"],
            ("扩散剩余误差修正", "num_refinement_steps"): diffusion_architecture["num_refinement_steps"],
            ("扩散剩余误差修正", "minimum_noise_standard_deviation"): diffusion_architecture["minimum_noise_standard_deviation"],
            ("扩散剩余误差修正", "ema_decay"): diffusion_training["ema_decay"],
            ("扩散剩余误差修正", "epochs"): diffusion_training["epochs"],
            ("扩散剩余误差修正", "effective_batch_size"): diffusion_training["batch_size"],
            ("扩散剩余误差修正", "microbatch_size"): diffusion_training["microbatch_size"],
            ("扩散剩余误差修正", "activation_precision"): diffusion_training["activation_precision"],
        }
        for key, value in expected.items():
            self.assertIn(key, self.by_key)
            actual = self.by_key[key]["value"]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.assertAlmostEqual(float(actual), float(value))
            else:
                self.assertEqual(actual, str(value))


if __name__ == "__main__":
    unittest.main()
