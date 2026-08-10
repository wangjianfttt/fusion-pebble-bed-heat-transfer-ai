import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/run_hccb_p418_poststeady_pipeline.sh"


class P418PoststeadyPipelineTest(unittest.TestCase):
    def test_time_step_comparison_precedes_formal_histories(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        consistency = text.index("compare_hccb_p418_preflight_formal_case.py")
        mesh_sensitivity = text.index("run_hccb_p418_mesh_sensitivity.sh")
        steady_postprocess = text.index("run_hccb_p418_60_postprocess.sh")
        self.assertLess(consistency, mesh_sensitivity)
        self.assertLess(mesh_sensitivity, steady_postprocess)
        sensitivity = text.index("run_hccb_p418_thermal_timestep_sensitivity.sh")
        selection = text.index("build_hccb_p418_selected_timestep_plan.py")
        table = text.index("build_hccb_p418_timestep_table.py")
        formal = text.index("run_hccb_p418_step_responses.sh")
        self.assertLess(sensitivity, selection)
        self.assertLess(selection, table)
        self.assertLess(table, formal)
        self.assertIn('PLAN="${SELECTED_PLAN}"', text)
        self.assertIn(
            'STEP_CONCURRENT_CASES=${STEP_CONCURRENT_CASES:-${CONCURRENT_CASES:-1}}',
            text,
        )
        self.assertIn(
            'CONCURRENT_CASES="${STEP_CONCURRENT_CASES}"',
            text,
        )
        self.assertIn('PLAN="${SELECTED_PLAN}" RUN_MODEL_TRAINING=0', text)
        self.assertNotIn("CONCURRENT_CASES=2", text)
        self.assertIn('completed=$(find "${MATRIX_ROOT}"', text)
        self.assertIn('"new_physical_parameters": []', text)
        self.assertIn(
            "DEVICE=cuda GRAPH_MICROBATCH_SIZE=1 TRANSOLVER_MICROBATCH_SIZE=1",
            text,
        )
        self.assertIn(
            "STEADY_RESULT_NAMESPACE=${STEADY_RESULT_NAMESPACE:-hccb_p418_60_corrected_20260731}",
            text,
        )
        self.assertIn('RESULT_NAMESPACE="${STEADY_RESULT_NAMESPACE}"', text)
        self.assertIn('COMPARISON_OUTPUT_DIR="${STEADY_COMPARISON_DIR}"', text)
        self.assertIn("--result-namespace", text)
        self.assertNotIn(
            "${RESULT_ROOT}/hccb_p418_60_model_comparison_100epoch",
            text,
        )
        for split_name in (
            "interleaved_all_ranges",
            "temperature_extrapolation",
            "velocity_extrapolation",
            "heat_source_interpolation",
            "heat_source_extrapolation",
        ):
            self.assertIn(split_name, text)

    def test_learning_curve_runs_after_primary_models_and_physical_steps(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        first_wait = text.index('wait "${steady_model_pid}"')
        seed_repeat = text.index("run_hccb_p418_steady_seed_robustness.sh")
        epoch_followup = text.index("run_hccb_p418_steady_epoch_followup.sh")
        chain_selection = text.index("select_hccb_p418_steady_chain_source.py")
        learning_curve = text.index("run_hccb_p418_steady_learning_curve.sh")
        self.assertLess(first_wait, learning_curve)
        self.assertLess(first_wait, seed_repeat)
        self.assertLess(seed_repeat, epoch_followup)
        cross_sources = text.index("build_hccb_p418_cross_packing_model_sources.py")
        self.assertLess(epoch_followup, cross_sources)
        chained = text.index("run_hccb_p418_chained_initial_state_evaluation.sh")
        self.assertLess(first_wait, chained)
        self.assertLess(epoch_followup, chain_selection)
        self.assertLess(chain_selection, chained)
        step_calls = []
        position = 0
        while True:
            try:
                position = text.index("run_hccb_p418_step_responses.sh", position)
            except ValueError:
                break
            step_calls.append(position)
            position += 1
        self.assertEqual(len(step_calls), 2)
        self.assertLess(step_calls[0], first_wait)
        self.assertLess(first_wait, step_calls[1])
        self.assertLess(seed_repeat, step_calls[1])
        self.assertLess(epoch_followup, step_calls[1])
        self.assertLess(step_calls[1], chained)
        first_step_context = text[max(0, step_calls[0] - 350) : step_calls[0]]
        second_step_context = text[max(0, step_calls[1] - 350) : step_calls[1]]
        self.assertIn("RUN_MODEL_TRAINING=0", first_step_context)
        self.assertIn("RUN_MODEL_TRAINING=1", second_step_context)
        chained_script = (
            ROOT / "code/run_hccb_p418_chained_initial_state_evaluation.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("build_hccb_p418_fused_chain_table.py", chained_script)
        self.assertIn("generated_fused_chain_results.tex", chained_script)
        step_script = (ROOT / "code/run_hccb_p418_step_responses.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("build_hccb_p418_transient_cost_table.py", step_script)
        self.assertIn("generated_transient_cost.tex", step_script)
        self.assertIn("build_hccb_p418_transient_performance_table.py", step_script)
        self.assertIn("generated_transient_performance.tex", step_script)
        self.assertIn("transient_cost_manuscript_table_sha256", text)
        self.assertIn("formal_result_files", text)
        self.assertIn("preflight_formal_consistency", text)
        self.assertIn("physical_and_model_source_summary", text)
        self.assertIn("physical_and_model_source_text", text)
        self.assertIn("steady_final_window_summary", text)
        self.assertIn("steady_final_window_text", text)
        self.assertIn("steady_hotspot_summary", text)
        self.assertIn("steady_hotspot_csv", text)
        self.assertIn("steady_hotspot_movements_csv", text)
        self.assertIn("mesh_sensitivity_summary", text)
        self.assertIn("mesh_sensitivity_gci", text)
        self.assertIn("mesh_sensitivity_table", text)
        self.assertIn("native_cell_performance_table", text)
        self.assertIn("steady_result_text", text)
        self.assertIn("thermal_regime_split_coverage", text)
        self.assertIn("dimensionless_heat_summary", text)
        self.assertIn("pressure_correlation_summary", text)
        self.assertIn("same_source_correlation_text", text)
        self.assertIn("transition_temperature_coverage", text)
        self.assertIn("transient_performance_table", text)
        self.assertIn("transient_performance_summary", text)
        self.assertIn("generated_transition_temperature_coverage.tex", text)
        self.assertIn("steady_learning_curve", text)
        self.assertIn(
            'STEADY_SUMMARY="${STEADY_CHAIN_SUMMARY}"',
            text,
        )
        self.assertLess(chained, learning_curve)

    def test_solver_iteration_transformer_is_not_in_the_paper_pipeline(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("run_hccb_p418_60_transient_models.sh", text)
        self.assertNotIn("relaxation_pid", text)

    def test_mesh_sensitivity_uses_current_source_flow_and_200_seconds(self) -> None:
        text = (ROOT / "code/run_hccb_p418_mesh_sensitivity.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("END_TIME=${END_TIME:-200}", text)
        self.assertIn("source_channel_volume_flow_preserved", text)
        self.assertIn("pore * fraction", text)
        self.assertIn("mesh_case_is_current", text)
        self.assertIn("build_hccb_p418_mesh_sensitivity_table.py", text)
        self.assertIn("generated_mesh_sensitivity.tex", text)


if __name__ == "__main__":
    unittest.main()
