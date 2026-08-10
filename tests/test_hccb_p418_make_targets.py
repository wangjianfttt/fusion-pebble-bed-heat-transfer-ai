import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def make_dry_run(target: str) -> str:
    completed = subprocess.run(
        ["make", "-n", target],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def test_p418_make_targets_use_the_formal_pipeline() -> None:
    progress = make_dry_run("p418-progress")
    plan = make_dry_run("p418-formal-plan")
    formal = make_dry_run("p418-formal-run")
    refresh = make_dry_run("p418-manuscript-refresh")
    assert "report_hccb_p418_runtime_progress.py" in progress
    assert "hccb_dense_cht_p418_sourceflow_preflight" in progress
    assert "hccb_p418_sourceflow_watch_status.txt" in progress
    assert "hccb_p418_sourceflow_runtime_progress.json" in progress
    assert "EXECUTE=0" in plan and "run_hccb_p418_formal_calculations.sh" in plan
    assert "EXECUTE=1" in formal and "run_hccb_p418_formal_calculations.sh" in formal
    assert "P418_PYTHON=" in plan
    assert "P418_PYTHON=" in formal
    assert "run_hccb_p418_manuscript_refresh.sh" in refresh


def test_readme_describes_current_two_packing_scope() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "make p418-progress" in readme
    assert "make p418-formal-run" in readme
    assert "当前P418结果只对应一套固定颗粒装填" not in readme
    assert "seed101和seed202" in readme
    assert "60+9=69" in readme
    assert "seed303" not in readme
    assert "60/60组都有第200次稳态非线性迭代" in readme
    assert "其中16组保留了175--200次迭代" in readme
