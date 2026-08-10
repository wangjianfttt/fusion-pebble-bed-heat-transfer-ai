from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/watch_hccb_p418_partial_physics.sh"


def test_partial_watcher_updates_scalar_and_full_field_tail_summary() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    assert "summarize_hccb_p418_completed_matrix_physics.py" in text
    assert "summarize_hccb_p418_formal_steady_tails.py" in text
    assert "--allow-partial" in text
    assert "completed > 4 ? completed - 4 : 0" in text
    assert "hccb_p418_sourceflow_partial_final_windows" in text


def test_readme_explains_partial_and_final_tail_commands() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "summarize_hccb_p418_formal_steady_tails.py" in text
    assert "--allow-partial" in text
    assert "60/60组都有第200次稳态非线性迭代" in text
    assert "其中16组保留了175--200次迭代" in text
    assert "这些编号是稳态迭代，不是物理秒" in text
