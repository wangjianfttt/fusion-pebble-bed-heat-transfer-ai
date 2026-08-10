#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "code/run_hccb_p418_60_model_comparison.sh"
SUMMARY = ROOT / "code/summarize_hccb_p418_60_model_comparison.py"
CONVERGENCE = ROOT / "code/assess_hccb_p418_training_convergence.py"
PLOT = ROOT / "code/plot_hccb_p418_steady_engineering_comparison.py"
PAPER_PLOT = ROOT / "code/plot_hccb_p418_steady_model_comparison.py"
PERFORMANCE_TABLE = ROOT / "code/build_hccb_p418_steady_performance_table.py"
THERMAL_REGIME = ROOT / "code/summarize_hccb_p418_thermal_regime_model_errors.py"
THERMAL_COVERAGE = ROOT / "code/summarize_hccb_p418_thermal_regime_split_coverage.py"
NATIVE_EVALUATOR = ROOT / "code/evaluate_hccb_p418_native_cell_prediction.py"
NATIVE_SUMMARY = ROOT / "code/summarize_hccb_p418_native_cell_predictions.py"


class P418ModelComparisonRunnerTest(unittest.TestCase):
    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)

    def test_common_physics_inputs_are_used(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        for required in (
            "train_hccb_p418_regional_response_surface.py",
            "validate_hccb_p418_steady_comparison_inputs.py",
            'common_input_check.json',
            "--state-targets",
            "--mass-targets",
            "--energy-targets",
            "--training-statistics",
            "--comparison-epochs",
            "check_hccb_p418_steady_result_current.py",
            "--split-file",
            "ARCHITECTURES=${ARCHITECTURES:-pinn_data_only pinn graph transolver}",
            "RESULT_NAMESPACE=${RESULT_NAMESPACE:-hccb_p418_60}",
            "COMPARISON_OUTPUT_DIR=${COMPARISON_OUTPUT_DIR:-${ROOT}/results/${RESULT_NAMESPACE}_model_comparison_",
            '--result-prefix "${RESULT_NAMESPACE}"',
            "assess_hccb_p418_training_convergence.py",
            "hccb_p418_ai_architecture_sources.json",
            "plot_hccb_p418_steady_engineering_comparison.py",
            "plot_hccb_p418_steady_model_comparison.py",
            "build_hccb_p418_steady_performance_table.py",
            "summarize_hccb_p418_thermal_regime_model_errors.py",
            "summarize_hccb_p418_thermal_regime_split_coverage.py",
            '"${RESULT_PREFIX}_completed_physics/completed_case_physics.csv"',
            "GRAPH_MICROBATCH_SIZE=${GRAPH_MICROBATCH_SIZE:-1}",
            "TRANSOLVER_MICROBATCH_SIZE=${TRANSOLVER_MICROBATCH_SIZE:-1}",
            "MODEL_SEED=${MODEL_SEED:-20260717}",
            "FORMAL_PAPER_OUTPUTS=${FORMAL_PAPER_OUTPUTS:-auto}",
            "EXPECTED_CASES=${EXPECTED_CASES:-60}",
            "DATASET_INDEX=${DATASET_INDEX:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3_dataset/dataset_index.json}",
            "SUBFACE_GEOMETRY=${SUBFACE_GEOMETRY:-${ROOT}/results/hccb_p418_subface_residual_geometry_r2/subface_residual_geometry.npz}",
            "evaluate_hccb_p418_native_cell_prediction.py",
            "summarize_hccb_p418_native_cell_predictions.py",
            "native_cell_${architecture}_interleaved_all_ranges",
            '--expected-cases "${EXPECTED_CASES}"',
            "skip formal 5x5 paper figure and table",
            '--training-seed "${MODEL_SEED}"',
            '--seed "${MODEL_SEED}"',
            '--microbatch-size "${GRAPH_MICROBATCH_SIZE}"',
            '--microbatch-size "${TRANSOLVER_MICROBATCH_SIZE}"',
            "training_checkpoint.pt",
            "resume interrupted",
            "--resume",
            "PARALLEL_RESULT_POLL_SECONDS",
            "parallel_chain_pid.txt",
            "active_training_pids_for_output",
            'pathlib.Path("/proc").glob("[0-9]*")',
            'arg == "--output-dir"',
            "wait for active training pid(s)",
        ):
            self.assertIn(required, text)

    def test_summary_script_compiles(self) -> None:
        subprocess.run(
            ["python3", "-m", "py_compile", str(SUMMARY)], check=True
        )
        subprocess.run(
            ["python3", "-m", "py_compile", str(CONVERGENCE)], check=True
        )
        subprocess.run(["python3", "-m", "py_compile", str(PLOT)], check=True)
        subprocess.run(["python3", "-m", "py_compile", str(PAPER_PLOT)], check=True)
        subprocess.run(["python3", "-m", "py_compile", str(PERFORMANCE_TABLE)], check=True)
        subprocess.run(
            ["python3", "-m", "py_compile", str(THERMAL_REGIME)], check=True
        )
        subprocess.run(
            ["python3", "-m", "py_compile", str(THERMAL_COVERAGE)], check=True
        )
        subprocess.run(
            ["python3", "-m", "py_compile", str(NATIVE_EVALUATOR)], check=True
        )
        subprocess.run(
            ["python3", "-m", "py_compile", str(NATIVE_SUMMARY)], check=True
        )


if __name__ == "__main__":
    unittest.main()
