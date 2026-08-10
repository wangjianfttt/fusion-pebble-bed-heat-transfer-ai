from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "build_hccb_p418_partial_cross_packing_text.py"
)


def payload() -> dict[str, object]:
    return {
        "status": "partial_seed101_seed202_integral_response_comparison",
        "accepted_common_case_count": 7,
        "failed_seed202_case_count": 2,
        "failed_seed202_cases": [
            "u0p25_T900_q4p85",
            "u0p25_T900_q8p85",
        ],
        "complete_nine_case_comparison": False,
        "metric_summary": {
            "outlet_temperature_K": {
                "mean_absolute_relative_change_percent": 0.524,
                "maximum_absolute_relative_change_percent": 0.661,
            },
            "maximum_solid_temperature_K": {
                "mean_absolute_relative_change_percent": 0.087,
                "maximum_absolute_relative_change_percent": 0.305,
            },
            "pressure_drop_Pa": {
                "mean_absolute_relative_change_percent": 16.467,
                "relative_change_percent_range": [14.67, 17.99],
            },
        },
    }


def run_builder(tmp_path: Path, data: dict[str, object]) -> subprocess.CompletedProcess[str]:
    source = tmp_path / "summary.json"
    output = tmp_path / "result.tex"
    source.write_text(json.dumps(data), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--summary",
            str(source),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_writes_explicit_partial_comparison(tmp_path: Path) -> None:
    completed = run_builder(tmp_path, payload())
    assert completed.returncode == 0, completed.stderr
    text = (tmp_path / "result.tex").read_text(encoding="utf-8")
    assert "Seven operating conditions" in text
    assert "not a complete nine-condition" in text
    assert "14.67--17.99" in text


def test_rejects_complete_nine_case_label(tmp_path: Path) -> None:
    data = payload()
    data["complete_nine_case_comparison"] = True
    completed = run_builder(tmp_path, data)
    assert completed.returncode != 0


def test_writes_complete_nine_case_comparison(tmp_path: Path) -> None:
    data = payload()
    data["status"] = "completed_seed101_seed202_integral_response_comparison"
    data["accepted_common_case_count"] = 9
    data["failed_seed202_case_count"] = 0
    data["failed_seed202_cases"] = []
    data["complete_nine_case_comparison"] = True
    completed = run_builder(tmp_path, data)
    assert completed.returncode == 0, completed.stderr
    text = (tmp_path / "result.tex").read_text(encoding="utf-8")
    assert "All nine tested operating conditions" in text
    assert "registered interior point" not in text
    assert "complete nine-condition" not in text
