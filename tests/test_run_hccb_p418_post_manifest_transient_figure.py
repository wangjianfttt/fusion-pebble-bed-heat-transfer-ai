from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "run_hccb_p418_post_manifest_transient_figure.sh"


def test_post_manifest_transient_figure_waits_for_executor() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'while kill -0 "${WAIT_PID}"' in text
    assert 'while kill -0 "${WAIT_VALIDATION_PID}"' in text
    assert 'sleep "${POLL_SECONDS}"' in text


def test_post_manifest_transient_figure_requires_final_summary_and_marker() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "completed_p418_physical_step_model_comparison" in text
    assert "generated_transient_model_comparison_validated.tex" in text
    assert "pair_disjoint_stress_test" in text
    assert "new_physical_parameter_values_added" in text
    assert "complete_formal_p418_transient_model_comparison_figure" in text


def test_post_manifest_transient_figure_runs_only_plotting_code() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "plot_hccb_p418_transient_model_comparison.py" in text
    assert "foamMultiRun" not in text
    assert "train_hccb" not in text
