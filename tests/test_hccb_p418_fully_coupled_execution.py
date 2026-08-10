from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from compare_hccb_p418_fixed_and_fully_coupled_steps import (  # noqa: E402
    DEFAULT_SIGNALS,
    compare,
)
from verify_hccb_p418_fully_coupled_timestep_summary import (  # noqa: E402
    verify_summary,
)


RUNNER = ROOT / "code/run_hccb_p418_fully_coupled_step_responses.sh"
TIMESTEP_RUNNER = (
    ROOT / "code/run_hccb_p418_fully_coupled_timestep_sensitivity.sh"
)
TIMESTEP_CONFIG = (
    ROOT / "parameters/hccb_p418_fully_coupled_timestep_sensitivity.json"
)
FULL_PLAN = ROOT / "parameters/hccb_p418_fully_coupled_step_plan.json"
FINALIZER = ROOT / "code/finalize_hccb_p418_fully_coupled_step_response.sh"


def write_observables(path: Path, offset: float) -> None:
    time = np.asarray([[0.0, 1.0, 2.0]], dtype=np.float64)
    values = np.empty((1, 3, len(DEFAULT_SIGNALS)), dtype=np.float64)
    for index in range(len(DEFAULT_SIGNALS)):
        values[0, :, index] = offset + index + np.asarray([0.0, 1.0, 3.0])
    np.savez_compressed(
        path,
        case_id=np.asarray(["velocity_up_T500_q6p85"], dtype=object),
        complete=np.asarray([True]),
        time_s=time,
        time_mask=np.asarray([[True, True, True]]),
        values=values,
        signal_names=np.asarray(DEFAULT_SIGNALS, dtype=object),
    )


def test_fully_coupled_runner_defaults_to_plan_only() -> None:
    completed = subprocess.run(
        ["bash", str(RUNNER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "no OpenFOAM command was started" in completed.stdout
    assert "To run after all 60 steady endpoints are complete" in completed.stdout


def test_fully_coupled_timestep_runner_defaults_to_plan_only() -> None:
    completed = subprocess.run(
        ["bash", str(TIMESTEP_RUNNER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "no OpenFOAM command was started" in completed.stdout
    assert "sequence=velocity_up_T500_q6p85" in completed.stdout
    assert "initial_delta_t_s=4e-05,2e-05,1e-05" in completed.stdout


def test_fully_coupled_runner_obeys_workstation_pause_marker(tmp_path: Path) -> None:
    marker = tmp_path / "pause"
    marker.write_text("pause\n", encoding="utf-8")
    environment = {
        **os.environ,
        "ROOT": str(ROOT),
        "EXECUTE": "1",
        "PAUSE_MARKER": str(marker),
    }
    completed = subprocess.run(
        ["bash", str(RUNNER)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 3
    assert "paused for cloud migration" in completed.stderr


def test_fully_coupled_shell_entries_have_valid_bash_syntax() -> None:
    for path in (RUNNER, TIMESTEP_RUNNER, FINALIZER):
        completed = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr


def write_timestep_summary(path: Path) -> dict[str, object]:
    config = json.loads(TIMESTEP_CONFIG.read_text(encoding="utf-8"))
    summary = {
        "status": "completed_p418_fully_coupled_timestep_sensitivity",
        "analysis_kind": "fully_coupled_flow_heat",
        "sequence_id": config["sequence_id"],
        "delta_t_s": sorted(config["delta_t_s"], reverse=True),
        "formal_selection_rule": config["formal_selection_rule"],
        "selected_delta_t_s": min(config["delta_t_s"]),
        "selected_time_step_schedule": config["formal_time_step_schedule"],
        "new_physical_parameters": [],
    }
    path.write_text(json.dumps(summary), encoding="utf-8")
    return summary


def test_timestep_summary_must_match_formal_velocity_step(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    write_timestep_summary(path)
    result = verify_summary(path, TIMESTEP_CONFIG, FULL_PLAN)
    assert result["sequence_id"] == "velocity_up_T500_q6p85"
    assert result["selected_delta_t_s"] == 1.0e-5
    assert result["new_physical_parameters"] == []


def test_timestep_summary_rejects_another_sequence(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    summary = write_timestep_summary(path)
    summary["sequence_id"] = "velocity_down_T500_q6p85"
    path.write_text(json.dumps(summary), encoding="utf-8")
    try:
        verify_summary(path, TIMESTEP_CONFIG, FULL_PLAN)
    except ValueError as error:
        assert "another sequence" in str(error)
    else:
        raise AssertionError("a different time-step sequence was accepted")


def test_formal_runner_rejects_wrong_timestep_summary_before_openfoam(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.json"
    summary = write_timestep_summary(path)
    summary["sequence_id"] = "velocity_down_T500_q6p85"
    path.write_text(json.dumps(summary), encoding="utf-8")
    environment = {
        **os.environ,
        "ROOT": str(ROOT),
        "EXECUTE": "1",
        "PAUSE_MARKER": str(tmp_path / "no_pause_marker"),
        "TIMESTEP_SUMMARY": str(path),
    }
    completed = subprocess.run(
        ["bash", str(RUNNER)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "another sequence" in completed.stderr
    assert "foamMultiRun" not in completed.stdout


def test_fixed_vs_fully_coupled_comparison_uses_matching_sequences(
    tmp_path: Path,
) -> None:
    fixed = tmp_path / "fixed.npz"
    coupled = tmp_path / "coupled.npz"
    write_observables(fixed, offset=1.0)
    write_observables(coupled, offset=0.0)

    rows, summary = compare(fixed, coupled, DEFAULT_SIGNALS)

    assert len(rows) == len(DEFAULT_SIGNALS)
    assert summary["sequence_count"] == 1
    assert summary["new_physical_parameters"] == []
    assert all(row["maximum_absolute_difference"] == 1.0 for row in rows)
    assert set(summary["aggregate_by_signal"]) == set(DEFAULT_SIGNALS)
    assert all(
        item["trajectory_count"] == 1
        for item in summary["aggregate_by_signal"].values()
    )
    assert "No fitted acceptance percentage" in summary["comparison_rule"]


def test_fixed_vs_fully_coupled_comparison_rejects_different_sequences(
    tmp_path: Path,
) -> None:
    fixed = tmp_path / "fixed.npz"
    coupled = tmp_path / "coupled.npz"
    write_observables(fixed, offset=1.0)
    write_observables(coupled, offset=0.0)
    data = np.load(coupled, allow_pickle=True)
    changed = {name: data[name] for name in data.files}
    changed["case_id"] = np.asarray(["different_sequence"], dtype=object)
    np.savez_compressed(coupled, **changed)

    try:
        compare(fixed, coupled, DEFAULT_SIGNALS)
    except ValueError as error:
        assert "same sequence ids" in str(error)
    else:
        raise AssertionError("different sequence ids were accepted")
