from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_completed_boundary_heat_summary.py"


def write_case(
    root: Path,
    case_id: str,
    generated: float,
    *,
    result_case_name: str | None = None,
) -> None:
    case = root / case_id
    case.mkdir(parents=True)
    (case / "formal_sample_complete.json").write_text(
        json.dumps(
            {
                "condition_id": case_id,
                "time": "200",
                "solver_finished": True,
            }
        ),
        encoding="utf-8",
    )
    (case / "cht_result_summary_200.json").write_text(
        json.dumps(
            {
                "case": str(case.with_name(result_case_name or case_id)),
                "solver_finished": True,
                "heat_balance": {
                    "solid_generated_power_W": generated,
                    "all_fluid_boundary_conductive_heat_flows_W": {
                        "inlet": -1.0,
                        "outlet": -0.1,
                        "coolingWall": 0.2,
                        "fluid_to_solid": 1.5,
                    },
                    "all_solid_boundary_conductive_heat_flows_W": {
                        "inlet": 0.0,
                        "outlet": 0.0,
                        "coolingWall": 0.5,
                        "solid_to_fluid": -1.5,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_builds_integral_boundary_heat_summary(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix"
    write_case(matrix, "case_a", 1.0)
    write_case(matrix, "case_b", 1.0)
    output = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--matrix-root",
            str(matrix),
            "--output",
            str(output),
            "--expected-case-count",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "p418_completed_boundary_heat_summary_ready"
    assert payload["case_count"] == 2
    assert payload["maximum_interface_pair_difference_W"] == 0.0
    assert payload["maximum_solid_balance_relative"] == 0.0
    assert payload["new_physical_parameters"] == []


def test_rejects_wrong_completed_case_count(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix"
    write_case(matrix, "case_a", 1.0)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--matrix-root",
            str(matrix),
            "--output",
            str(tmp_path / "summary.json"),
            "--expected-case-count",
            "2",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "completed case count 1 != 2" in completed.stderr


def test_accepts_numeric_archive_run_suffix(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix"
    write_case(
        matrix,
        "u0p10_T300_q8p85",
        1.0,
        result_case_name="u0p10_T300_q8p85_12921",
    )
    output = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--matrix-root",
            str(matrix),
            "--output",
            str(output),
            "--expected-case-count",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(output.read_text(encoding="utf-8"))["case_count"] == 1


def test_rejects_nonnumeric_archive_run_suffix(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix"
    write_case(
        matrix,
        "u0p10_T300_q8p85",
        1.0,
        result_case_name="u0p10_T300_q8p85_other",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--matrix-root",
            str(matrix),
            "--output",
            str(tmp_path / "summary.json"),
            "--expected-case-count",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "result summary case mismatch" in completed.stderr
