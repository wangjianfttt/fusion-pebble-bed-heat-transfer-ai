#!/usr/bin/env python3

from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "parameters/hccb_p418_ai_architecture_sources.json"
MODEL = ROOT / "code/hccb_p418_parametric_regional_operator.py"
PINN_MODEL = ROOT / "code/hccb_p418_coordinate_pinn.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keyword_defaults(path: Path, class_name: str) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__":
                    return {
                        argument.arg: ast.literal_eval(default)
                        for argument, default in zip(item.args.kwonlyargs, item.args.kw_defaults)
                        if default is not None
                    }
    raise AssertionError(f"{class_name}.__init__ was not found")


class P418AIArchitectureSourcesTest(unittest.TestCase):
    def test_direct_hccb_pinn_precedents_are_complete_and_bounded(self) -> None:
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        precedents = {
            entry["doi"]: entry
            for entry in registry["domain_specific_pinn_precedents"]
        }
        self.assertEqual(
            set(precedents),
            {
                "10.3303/CET24114068",
                "10.1016/j.ijheatmasstransfer.2025.126970",
            },
        )
        expected_unreported = {
            "network depth",
            "network width",
            "learning rate",
            "batch size",
            "random seed",
        }
        for doi, entry in precedents.items():
            self.assertTrue(entry["paper_url"].startswith(("http://", "https://")))
            self.assertTrue(expected_unreported.issubset(entry["unreported_settings"]))
            self.assertIn(
                "a source of new pebble-bed physical parameters",
                entry["not_used_as"],
            )
            for path_key, hash_key in (
                ("local_source_pdf", "local_source_pdf_sha256"),
                ("local_text", "local_text_sha256"),
            ):
                source = ROOT / entry[path_key]
                self.assertTrue(source.is_file(), f"{doi}: missing {path_key}")
                self.assertEqual(sha256(source), entry[hash_key])
            if entry.get("local_method_note"):
                self.assertTrue((ROOT / entry["local_method_note"]).is_file())

    def test_archived_source_hashes_and_primary_defaults_match(self) -> None:
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        architectures = {entry["name"]: entry for entry in registry["architectures"]}

        rigno = architectures["RIGNO-style regional graph operator"]
        transolver = architectures["Transolver"]
        pde_refiner = architectures["PDE-Refiner-style diffusion refinement"]
        pinn = architectures["PINO-paper coordinate PINN control"]
        temporal = architectures["Temporal Transformer trajectory operator"]
        spatiotemporal = architectures["Published-component spatiotemporal regional operator"]
        dmdc = architectures["Volume-weighted DMDc baseline"]
        low_rank = architectures["Snapshot-POD low-rank temperature-residual correction"]
        selection = registry["architecture_selection_boundary"]
        self.assertEqual(
            transolver["paper_url"], "https://proceedings.mlr.press/v235/wu24r.html"
        )
        self.assertEqual(
            {entry["method"] for entry in selection["not_added_now"]},
            {
                "SPINN",
                "LE-PDE",
                "Radon neural operator",
                "Porous-DeepONet and physics-informed DeepONet",
                "Voxel CNN/U-Net porous-field predictors",
                "UNO-PINO/FNO family",
                "Graph-Physics temporal correction and local multi-node prediction",
            },
        )
        self.assertTrue(all(entry["reason"] for entry in selection["not_added_now"]))
        repeat_rule = registry["training_repeat_rule"]
        self.assertEqual(repeat_rule["strict_split"], "pair_disjoint_stress_test")
        self.assertEqual(len(repeat_rule["strict_split_seeds"]), 3)
        self.assertEqual(
            sha256(ROOT / repeat_rule["summary_script"]),
            repeat_rule["summary_script_sha256"],
        )
        energy_evaluation = registry["common_transient_energy_evaluation"]
        self.assertEqual(energy_evaluation["device"], "CPU")
        self.assertEqual(
            sha256(ROOT / energy_evaluation["script"]),
            energy_evaluation["script_sha256"],
        )
        self.assertEqual(sha256(ROOT / rigno["local_source"]), rigno["local_source_sha256"])
        self.assertEqual(
            sha256(ROOT / transolver["local_source"]),
            transolver["local_source_sha256"],
        )
        self.assertEqual(
            sha256(ROOT / pde_refiner["local_source"]),
            pde_refiner["local_source_sha256"],
        )
        self.assertEqual(sha256(ROOT / pinn["local_source"]), pinn["local_source_sha256"])
        self.assertEqual(sha256(ROOT / pinn["local_config"]), pinn["local_config_sha256"])
        self.assertEqual(
            sha256(ROOT / temporal["implementation"]), temporal["implementation_sha256"]
        )
        self.assertEqual(
            sha256(ROOT / temporal["physical_step_exporter"]),
            temporal["physical_step_exporter_sha256"],
        )
        self.assertEqual(
            sha256(ROOT / temporal["physical_step_runner"]),
            temporal["physical_step_runner_sha256"],
        )
        self.assertEqual(
            sha256(ROOT / "code/hccb_p418_spatiotemporal_regional_operator.py"),
            spatiotemporal["implementation_model_sha256"],
        )
        self.assertEqual(
            sha256(ROOT / "code/train_hccb_p418_spatiotemporal_regional_operator.py"),
            spatiotemporal["implementation_trainer_sha256"],
        )
        self.assertEqual(
            sha256(ROOT / spatiotemporal["source_contract"]),
            spatiotemporal["source_contract_sha256"],
        )
        fully_coupled = spatiotemporal["fully_coupled_extension"]
        self.assertEqual(
            fully_coupled["architecture_revision"],
            "p418_fully_coupled_oriented_initial_face_flux_context_v2",
        )
        self.assertEqual(
            fully_coupled["initial_internal_face_flux_message"],
            "owner:+m_dot;neighbour:-m_dot",
        )
        self.assertEqual(
            sha256(ROOT / fully_coupled["implementation"]),
            spatiotemporal["fully_coupled_extension_implementation_sha256"],
        )
        component_sources = {
            item["component"]: item for item in spatiotemporal["published_component_sources"]
        }
        self.assertEqual(len(component_sources), 5)
        self.assertEqual(
            component_sources["global Physics-Attention over learnable physical slices"]["paper_url"],
            "https://proceedings.mlr.press/v235/wu24r.html",
        )
        self.assertIn(
            "NeurIPS 2017",
            component_sources["temporal self-attention encoder"]["venue"],
        )
        self.assertEqual(
            component_sources[
                "adjacency-masked graph Transformer tested on large three-dimensional CFD meshes"
            ]["paper_url"],
            "https://arxiv.org/abs/2508.18051",
        )
        self.assertEqual(
            spatiotemporal["numerical_settings_reference"]["doi"],
            "10.48550/arXiv.2601.23177",
        )
        self.assertEqual(
            sha256(ROOT / spatiotemporal["numerical_settings_reference"]["local_source_pdf"]),
            spatiotemporal["numerical_settings_reference"]["local_source_sha256"],
        )
        gpu = spatiotemporal["gpu_efficiency_result"]
        repeated = json.loads((ROOT / gpu["repeated_result"]).read_text(encoding="utf-8"))
        factorized = json.loads((ROOT / gpu["factorized_result"]).read_text(encoding="utf-8"))
        cpu_gpu = json.loads((ROOT / gpu["cpu_gpu_result"]).read_text(encoding="utf-8"))
        self.assertEqual(gpu["nodes"], repeated["nodes"])
        self.assertEqual(gpu["edges"], repeated["edges"])
        self.assertEqual(gpu["time_points"], repeated["time_points"])
        self.assertAlmostEqual(gpu["repeated_query_seconds"], repeated["elapsed_seconds"])
        self.assertAlmostEqual(gpu["factorized_static_seconds"], factorized["elapsed_seconds"])
        self.assertAlmostEqual(
            gpu["factorized_gpu_vs_cpu8_speedup"], cpu_gpu["gpu_update_speedup"]
        )
        self.assertEqual(
            sha256(ROOT / dmdc["implementation"]), dmdc["implementation_sha256"]
        )
        self.assertEqual(
            sha256(ROOT / low_rank["implementation"]),
            low_rank["implementation_sha256"],
        )
        self.assertEqual(low_rank["doi"], "10.1090/qam/910462")
        self.assertIn("validation", low_rank["source_settings"]["rank_selection"])
        self.assertIn("zero at t=0", low_rank["source_settings"]["initial_condition"])
        for filename, key in (
            ("code/hccb_p418_regional_diffusion_refiner.py", "regional_model_sha256"),
            ("code/train_hccb_p418_regional_diffusion_refiner.py", "regional_trainer_sha256"),
            ("code/hccb_p418_temporal_temperature_diffusion.py", "temporal_model_sha256"),
            ("code/train_hccb_p418_temporal_temperature_diffusion.py", "temporal_trainer_sha256"),
        ):
            self.assertEqual(sha256(ROOT / filename), pde_refiner[key])
        self.assertEqual(
            sha256(ROOT / pde_refiner["resource_test_script"]),
            pde_refiner["resource_test_script_sha256"],
        )
        diffusion_gpu = pde_refiner["actual_graph_gpu_resource_result"]
        diffusion_float = json.loads(
            (ROOT / diffusion_gpu["float32_result"]).read_text(encoding="utf-8")
        )
        diffusion_bfloat = json.loads(
            (ROOT / diffusion_gpu["bfloat16_result"]).read_text(encoding="utf-8")
        )
        self.assertEqual(diffusion_float["nodes"], diffusion_bfloat["nodes"])
        self.assertEqual(diffusion_float["time_points"], diffusion_bfloat["time_points"])
        self.assertGreater(
            diffusion_float["peak_gpu_GB"], diffusion_bfloat["peak_gpu_GB"]
        )
        self.assertEqual(diffusion_bfloat["activation_precision"], "bfloat16")
        self.assertEqual(pde_refiner["source_settings"]["effective_batch_size"], 8)
        self.assertEqual(pde_refiner["source_settings"]["curve_microbatch_size"], 1)

        defaults = keyword_defaults(MODEL, "HCCBP418ParametricRegionalOperator")
        self.assertEqual(defaults["hidden_dim"], rigno["source_settings"]["node_latent_size"])
        self.assertEqual(defaults["processor_steps"], rigno["source_settings"]["processor_steps"])
        self.assertEqual(defaults["active_levels"], rigno["source_settings"]["regional_mesh_levels"])
        self.assertEqual(defaults["attention_heads"], transolver["source_settings"]["attention_heads"])
        self.assertEqual(defaults["physics_slices"], transolver["source_settings"]["physics_slices"])
        pinn_defaults = keyword_defaults(PINN_MODEL, "HCCBP418CoordinatePINNOperator")
        self.assertEqual(pinn_defaults["hidden_dim"], pinn["source_settings"]["hidden_width"])
        self.assertEqual(pinn_defaults["hidden_layers"], pinn["source_settings"]["hidden_layers"])


if __name__ == "__main__":
    unittest.main()
