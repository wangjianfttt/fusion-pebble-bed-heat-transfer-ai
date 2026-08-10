from pathlib import Path
import os
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "run_hccb_p418_continue_after_seed101.sh"


def test_shell_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_dry_run_starts_nothing(tmp_path: Path):
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env={"ROOT": str(tmp_path), "EXECUTE": "0", "PATH": os.environ["PATH"]},
        check=True,
        capture_output=True,
        text=True,
    )
    assert "physical thermal steps" in result.stdout
    assert "dry run only" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_continuation_waits_for_60_and_calls_formal_route():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "formal_sample_complete.json" in text
    assert 'if [[ ${completed} -eq 60 ]]' in text
    assert "run_hccb_dense_cht_p418_matrix_parallel.sh" in text
    assert "run_hccb_p418_formal_calculations.sh" in text
    assert 'ROOT="${ROOT}" P418_PYTHON="${P418_PYTHON}" EXECUTE=1' in text
    assert 'P418_PYTHON="${P418_PYTHON}"' in text
    assert 'export PATH="$(dirname "${P418_PYTHON}"):${PATH}"' in text
    assert "import numpy, pandas, scipy, sklearn, torch" in text
