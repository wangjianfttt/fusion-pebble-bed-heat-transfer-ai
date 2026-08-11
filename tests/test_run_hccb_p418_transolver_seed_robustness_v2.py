from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "run_hccb_p418_transolver_seed_robustness_v2.sh"


def test_runner_uses_three_fixed_seeds_and_current_transolver_code():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "SEEDS=(20260717 20260718 20260719)" in text
    assert "--architecture transolver" in text
    assert "--epochs \"${EPOCHS}\"" in text
    assert "--device cpu" in text


def test_runner_does_not_overwrite_existing_results():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "refusing to replace existing path" in text
    assert "incomplete output already exists" in text
    assert "mv " not in text
    assert "rm " not in text


def test_runner_builds_one_common_summary_after_all_seeds_finish():
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'wait "${pid}"' in text
    assert "summarize_hccb_p418_steady_seed_robustness.py" in text
    assert "hccb_p418_60_steady_seed_robustness_100epoch" in text
    assert "generated_steady_seed_robustness_text.tex" in text
    assert '--text-output "${TEXT_OUTPUT}"' in text
