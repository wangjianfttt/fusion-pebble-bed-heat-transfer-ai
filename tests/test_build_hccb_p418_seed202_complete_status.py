from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_seed202_complete_status.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_builds_complete_status_from_verified_case(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix"
    case = matrix / "case_a"
    case.mkdir(parents=True)
    summary = {
        "solver_finished": True,
        "reported_iteration": 200.0,
        "all_reported_values_are_finite": True,
        "flow": {"relative_mass_difference": 1e-8, "pressure_drop_Pa": 12.0},
        "temperature": {"outlet_average_K": 510.0, "solid_maximum_K": 620.0},
        "heat_balance": {"relative_energy_difference": 2e-5},
    }
    summary_path = case / "cht_result_summary_200.json"
    marker_path = case / "formal_sample_complete.json"
    heat_path = case / "boundary_heat_flows_200.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    marker_path.write_text('{"status":"complete"}', encoding="utf-8")
    heat_path.write_text('{"status":"complete"}', encoding="utf-8")
    recovery = {
        "status": "seed202_schema3_matrix_ready",
        "missing_conditions": [],
        "sha_mismatches": [],
        "checks": {"old_failed_14356_6_14356_7_excluded": True},
        "cases": [
            {
                "condition_id": "case_a",
                "formal_source_job": "job1",
                "result_summary_sha256": digest(summary_path),
                "marker_sha256": digest(marker_path),
                "boundary_heat_flows_sha256": digest(heat_path),
            }
        ],
    }
    recovery_path = tmp_path / "recovery.json"
    output = tmp_path / "status.json"
    recovery_path.write_text(json.dumps(recovery), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--matrix-root",
            str(matrix),
            "--recovery-record",
            str(recovery_path),
            "--output",
            str(output),
            "--expected-cases",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["accepted_case_count"] == 1
    assert payload["failed_case_count"] == 0
    assert payload["accepted_cases"][0]["pressure_drop_Pa"] == 12.0
