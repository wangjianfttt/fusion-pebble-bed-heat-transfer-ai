from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/resume_hccb_p418_after_power_loss.sh"


def test_power_recovery_rejoins_the_complete_formal_route() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    matrix = text.index("run_hccb_dense_cht_p418_matrix_parallel.sh")
    complete_check = text.index("steady matrix incomplete")
    formal = text.index("run_hccb_p418_formal_calculations.sh")
    assert matrix < complete_check < formal
    assert "run_hccb_p418_poststeady_pipeline.sh" not in text
    assert 'P418_PYTHON=${P418_PYTHON:-/data2/wangjian/venv/bin/python3}' in text
    assert "EXECUTE=1" in text
    assert 'NP_PER_CASE="${NP_PER_CASE}"' in text
    assert 'CONCURRENT_CASES="${CONCURRENT_CASES}"' in text
