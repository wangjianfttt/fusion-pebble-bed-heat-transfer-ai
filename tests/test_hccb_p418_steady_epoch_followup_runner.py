from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_followup_runner_uses_source_plan_and_physical_comparison() -> None:
    text = (ROOT / "code/run_hccb_p418_steady_epoch_followup.sh").read_text(
        encoding="utf-8"
    )
    for required in (
        "build_hccb_p418_epoch_followup_plan.py",
        "training_convergence.json",
        "train_hccb_p418_conservative_mixed_operator.py",
        "check_hccb_p418_steady_result_current.py",
        "compare_hccb_p418_epoch_followup.py",
        "GRAPH_MICROBATCH_SIZE",
        "TRANSOLVER_MICROBATCH_SIZE",
        "MODEL_SEED=${MODEL_SEED:-20260717}",
        '--training-seed "${MODEL_SEED}"',
        '--seed "${MODEL_SEED}"',
    ):
        assert required in text


def test_poststeady_pipeline_runs_epoch_followup() -> None:
    text = (ROOT / "code/run_hccb_p418_poststeady_pipeline.sh").read_text(
        encoding="utf-8"
    )
    assert "run_hccb_p418_steady_epoch_followup.sh" in text
    assert "wait \"${steady_model_pid}\"" in text
    assert "run_hccb_p418_steady_seed_robustness.sh" in text
