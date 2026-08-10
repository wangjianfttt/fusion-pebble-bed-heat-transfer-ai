import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_p418_manuscript_values import build, write_macros  # noqa: E402


class P418ManuscriptValuesTest(unittest.TestCase):
    def write(self, root: Path, relative: str, payload: dict) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    def test_values_are_read_from_result_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(
                root,
                "results/hccb_p418_sourceflow_runtime_progress.json",
                {
                    "status": "P418 matrix runtime progress",
                    "completed_cases": 4,
                    "total_cases": 60,
                },
            )
            self.write(
                root,
                "results/hccb_p418_actual_spatiotemporal_operator_56time_gpu_factorized/summary.json",
                {
                    "status": "formal_actual_graph_model_and_transient_physics_backward_passed",
                    "nodes": 100,
                    "edges": 250,
                    "time_points": 56,
                    "model_parameter_count": 42,
                    "peak_gpu_GB": 1.25,
                },
            )
            self.write(
                root,
                "results/hccb_p418_actual_interface_coupling/summary.json",
                {
                    "all_conditions": {
                        "interface_pair_count": 55,
                        "maximum_flux_sum_over_global_interface_flux": 2.0e-8,
                    }
                },
            )
            self.write(
                root,
                "results/hccb_p418_velocity_step_time_scales/summary.json",
                {
                    "resolved_local_crop_crossing_times_s": [0.01, 0.05],
                    "velocity_basis": "source_channel_area_preserving_pore_boundary_velocity",
                    "particle_radial_conduction_scale_s": {
                        "minimum": 0.4,
                        "maximum": 0.6,
                    },
                },
            )
            self.write(
                root,
                "results/hccb_p418_sourceflow_preflight/formal_60_input_summary.json",
                {
                    "status": "hccb_p418_60_actual_case_inputs_verified",
                    "cases": [{"mesh_triangulated_porosity": 0.39}],
                },
            )

            values = build(root)
            self.assertEqual(values["SteadyProgressText"], "4/60")
            self.assertEqual(values["RegionalNodes"], "100")
            self.assertEqual(values["InterfacePairCount"], "55")
            self.assertEqual(values["MeshPorosity"], "0.390000")

            output = root / "manuscript/generated_results.tex"
            write_macros(values, output)
            text = output.read_text(encoding="utf-8")
            self.assertIn("\\newcommand{\\RegionalEdges}{250}", text)
            self.assertNotIn("pending formal calculation", text)
            self.assertNotIn("\\PendingFormalResult", text)

    def test_corrected_matrix_markers_override_an_old_progress_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(
                root,
                "results/hccb_p418_sourceflow_runtime_progress.json",
                {
                    "status": "P418 matrix runtime progress",
                    "completed_cases": 29,
                    "total_cases": 60,
                },
            )
            self.write(
                root,
                "results/hccb_p418_actual_spatiotemporal_operator_56time_gpu_factorized/summary.json",
                {
                    "status": "formal_actual_graph_model_and_transient_physics_backward_passed",
                    "nodes": 100,
                    "edges": 250,
                    "time_points": 56,
                    "model_parameter_count": 42,
                    "peak_gpu_GB": 1.25,
                },
            )
            self.write(
                root,
                "results/hccb_p418_actual_interface_coupling/summary.json",
                {
                    "all_conditions": {
                        "interface_pair_count": 55,
                        "maximum_flux_sum_over_global_interface_flux": 2.0e-8,
                    }
                },
            )
            self.write(
                root,
                "results/hccb_p418_velocity_step_time_scales/summary.json",
                {
                    "resolved_local_crop_crossing_times_s": [0.01, 0.05],
                    "velocity_basis": "source_channel_area_preserving_pore_boundary_velocity",
                    "particle_radial_conduction_scale_s": {
                        "minimum": 0.4,
                        "maximum": 0.6,
                    },
                },
            )
            self.write(
                root,
                "results/hccb_p418_sourceflow_preflight/formal_60_input_summary.json",
                {
                    "status": "hccb_p418_60_actual_case_inputs_verified",
                    "cases": [{"mesh_triangulated_porosity": 0.39}],
                },
            )
            matrix = root / "hccb_dense_cht_p418_60_sourceflow_r3"
            for index in range(60):
                case = matrix / f"u{index:02d}_T300_q4p85"
                case.mkdir(parents=True)
                if index < 5:
                    (case / "formal_sample_complete.json").write_text("{}\n")
            self.assertEqual(build(root)["SteadyProgressText"], "5/60")

    def test_verified_training_coverage_overrides_sparse_local_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(
                root,
                "results/hccb_p418_training_data_coverage_partial/summary.json",
                {
                    "completed_case_count": 40,
                    "expected_case_count": 60,
                },
            )
            self.write(
                root,
                "results/hccb_p418_sourceflow_runtime_progress.json",
                {
                    "status": "P418 matrix runtime progress",
                    "completed_cases": 1,
                    "total_cases": 60,
                },
            )
            self.write(
                root,
                "results/hccb_p418_actual_spatiotemporal_operator_56time_gpu_factorized/summary.json",
                {
                    "status": "formal_actual_graph_model_and_transient_physics_backward_passed",
                    "nodes": 100,
                    "edges": 250,
                    "time_points": 56,
                    "model_parameter_count": 42,
                    "peak_gpu_GB": 1.25,
                },
            )
            self.write(
                root,
                "results/hccb_p418_actual_interface_coupling/summary.json",
                {
                    "all_conditions": {
                        "interface_pair_count": 55,
                        "maximum_flux_sum_over_global_interface_flux": 2.0e-8,
                    }
                },
            )
            self.write(
                root,
                "results/hccb_p418_velocity_step_time_scales/summary.json",
                {
                    "resolved_local_crop_crossing_times_s": [0.01, 0.05],
                    "velocity_basis": "source_channel_area_preserving_pore_boundary_velocity",
                    "particle_radial_conduction_scale_s": {
                        "minimum": 0.4,
                        "maximum": 0.6,
                    },
                },
            )
            self.write(
                root,
                "results/hccb_p418_sourceflow_preflight/formal_60_input_summary.json",
                {
                    "status": "hccb_p418_60_actual_case_inputs_verified",
                    "cases": [{"mesh_triangulated_porosity": 0.39}],
                },
            )

            self.assertEqual(build(root)["SteadyProgressText"], "40/60")


if __name__ == "__main__":
    unittest.main()
