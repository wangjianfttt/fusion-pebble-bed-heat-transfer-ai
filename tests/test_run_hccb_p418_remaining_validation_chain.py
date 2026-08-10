from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/run_hccb_p418_remaining_validation_chain.sh"


def test_remaining_validation_chain_is_serial_and_model_only() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    steady_seeds = text.index("run_hccb_p418_steady_seed_robustness.sh")
    steady_curve = text.index("run_hccb_p418_steady_learning_curve.sh")
    transient_curve = text.index("run_hccb_p418_transient_learning_curve.sh")
    selected_outputs = text.index("rerun_hccb_p418_post_selection_outputs.py")
    record = text.index('"status": "completed_p418_remaining_validation_chain"')
    assert steady_seeds < steady_curve < transient_curve < selected_outputs < record
    assert "CUDA_VISIBLE_DEVICES" in text
    assert "taskset -c" in text
    assert "json_has_status" in text
    assert "formal training manifest must contain exactly 75 jobs" in text
    assert "follow-up training was not started" in text
    assert text.index("while kill -0") < text.index("formal 75-job chain complete")
    assert "foamMultiRun" not in text
    assert "chtMultiRegionFoam" not in text
    assert '"openfoam_solver_started": False' in text
    assert '"new_physical_parameters": []' in text
