from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_observation_duration_uses_completed_fine_history(tmp_path: Path) -> None:
    output = tmp_path / "duration"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "code/analyze_hccb_p418_observation_duration.py"),
            "--output-dir",
            str(output),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed_p418_observation_duration_analysis"
    assert len(summary["signals"]) == 4
    assert summary["new_physical_parameters"] == []
    assert 7.0 < summary["representative_slowest_time_constant_s"] < 9.0
    assert (
        summary["conservative_minimum_velocity_time_to_0p1_percent_s"] < 200.0
    )
    assert (
        summary["single_exponential_remaining_fraction_at_candidate_duration"]
        < 1.0e-4
    )
    assert (output / "tail_fit_windows.csv").exists()
    assert (output / "观察时长说明_CN.md").exists()
