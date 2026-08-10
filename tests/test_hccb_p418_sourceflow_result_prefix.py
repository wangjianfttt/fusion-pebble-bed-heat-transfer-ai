from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


FORMAL_ROUTE_FILES = (
    "README.md",
    "code/run_hccb_p418_60_postprocess.sh",
    "code/run_hccb_p418_steady_epoch_followup.sh",
    "code/run_hccb_p418_steady_seed_robustness.sh",
    "code/run_hccb_p418_steady_loss_weight_sensitivity.sh",
    "code/run_hccb_p418_60_model_comparison.sh",
    "code/run_hccb_p418_experimental_comparison.sh",
    "code/run_hccb_p418_60_diffusion_refiner.sh",
    "code/run_hccb_p418_steady_learning_curve.sh",
    "code/run_hccb_p418_learned_model_experimental_comparison.sh",
    "code/run_hccb_p418_step_responses.sh",
    "code/run_hccb_p418_chained_initial_state_evaluation.sh",
    "code/run_hccb_p418_cross_packing_setup.sh",
    "code/run_hccb_p418_formal_calculations.sh",
    "code/run_hccb_p418_poststeady_pipeline.sh",
    "parameters/hccb_p418_cross_packing_model_protocol.json",
    "manuscript/result_source_map.csv",
)


def test_corrected_sourceflow_results_do_not_reuse_the_old_r2_prefix() -> None:
    combined = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in FORMAL_ROUTE_FILES
    )
    assert "hccb_p418_60_r2" not in combined
    assert "hccb_p418_60_sourceflow_r3" in combined
