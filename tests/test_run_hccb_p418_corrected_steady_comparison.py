from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/run_hccb_p418_corrected_steady_comparison.sh"


def test_corrected_comparison_reuses_results_without_training() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    assembly = text.index("assemble_hccb_p418_corrected_steady_results.py")
    comparison = text.index("summarize_hccb_p418_60_model_comparison.py")
    native = text.index("summarize_hccb_p418_native_cell_predictions.py")
    postprocess = text.index("run_hccb_p418_60_model_postprocess_only.sh")
    assert assembly < comparison < native < postprocess
    assert "hccb_p418_60_normfix_20260731" in text
    assert "hccb_p418_60_corrected_20260731" in text
    assert "heat_source_extrapolation" in text
    assert "train_hccb_p418_regional_response_surface.py" not in text
    assert "train_hccb_p418_conservative_mixed_operator.py" not in text
