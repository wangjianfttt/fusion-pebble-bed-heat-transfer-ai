from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/run_hccb_p418_post_manifest_manuscript_finalization.sh"


def test_finalizer_waits_for_models_and_both_final_figures() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "WAIT_PIDS" in text
    assert "model_comparison/summary.json" in text
    assert "generated_transient_model_comparison_validated.tex" in text
    assert "generated_openfoam_model_field_comparison_validated.tex" in text
    assert "complete_formal_p418_transient_model_comparison_figure" in text
    assert "complete_same_scale_openfoam_model_field_comparison" in text
    assert 'selection_data_role") != "validation"' in text
    assert 'display_data_role") != "test"' in text


def test_finalizer_builds_and_checks_the_complete_manuscript() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    remaining = text.index("run_hccb_p418_remaining_validation_chain.sh")
    refresh = text.index("run_hccb_p418_manuscript_refresh.sh")
    validation = text.index("completed_p418_formal_manuscript_refresh")
    assert remaining < refresh < validation
    assert "completed_p418_remaining_validation_chain" in text
    assert "REMAINING_VALIDATION_LOCK" in text
    assert "while ! flock -n 8" in text
    assert text.count("run_hccb_p418_remaining_validation_chain.sh") == 1
    assert "CUDA_VISIBLE_DEVICES" in text
    assert "BUILD_SUPPLEMENT" in text
    assert ".post_manifest_manuscript_finalization.lock" in text
    assert "foamMultiRun" not in text
    assert "mpirun" not in text
