from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/run_hccb_p418_corrected_steady_when_ready.sh"


def test_waiter_requires_both_corrected_models_and_runs_only_postprocessing() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "required_methods=(graph transolver)" in text
    assert "train_regional_predictions.npz" in text
    assert "validation_regional_predictions.npz" in text
    assert "test_regional_predictions.npz" in text
    assert "run_hccb_p418_corrected_steady_comparison.sh" in text
    assert "corrected_steady_result_assembly_complete" in text
    assert "result_count" in text
    assert "train_hccb_p418" not in text
    assert "foamMultiRun" not in text
    assert "mpirun" not in text


def test_waiter_uses_lock_and_does_not_overwrite_completed_assembly() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert ".corrected_steady_postprocess.lock" in text
    assert "mkdir \"${LOCK}\"" in text
    first_existing_check = text.index(
        "if [[ -s ${COMPARISON_DIR}/corrected_result_assembly.json ]]"
    )
    model_wait = text.index("while :; do")
    assert first_existing_check < model_wait
