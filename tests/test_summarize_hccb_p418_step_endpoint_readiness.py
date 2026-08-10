#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from summarize_hccb_p418_step_endpoint_readiness import (  # noqa: E402
    build_summary,
    endpoint_state,
    write_outputs,
)


def write_marker(root: Path, condition_id: str, *, finished: bool = True) -> None:
    case = root / condition_id
    case.mkdir(parents=True)
    sample = case / "training_sample_200_schema3/fields_and_topology.npz"
    sample.parent.mkdir()
    sample.write_bytes(b"sample")
    (case / "formal_sample_complete.json").write_text(
        json.dumps(
            {
                "time": "200",
                "solver_time_semantics": "steady_iteration_index",
                "physical_time_s": None,
                "solver_finished": finished,
                "relative_mass_difference": 1.0e-8,
                "relative_energy_difference": 2.0e-5,
                "training_sample": str(sample),
                "training_sample_schema_version": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_endpoint_requires_successful_solver_and_existing_schema3_sample(tmp_path: Path) -> None:
    write_marker(tmp_path, "ready")
    write_marker(tmp_path, "failed", finished=False)
    ready = endpoint_state(tmp_path, "ready")
    assert ready["ready"] is True
    assert ready["steady_iteration"] == 200
    assert ready["solver_time_semantics"] == "steady_iteration_index"
    assert ready["physical_time_s"] is None
    assert "time_s" not in ready
    failed = endpoint_state(tmp_path, "failed")
    assert failed["ready"] is False
    assert "not marked as finished" in str(failed["reason"])
    assert endpoint_state(tmp_path, "missing")["ready"] is False


def test_endpoint_rejects_nonsteady_time_semantics(tmp_path: Path) -> None:
    write_marker(tmp_path, "transient")
    marker = tmp_path / "transient" / "formal_sample_complete.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["solver_time_semantics"] = "physical_time_s"
    marker.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    state = endpoint_state(tmp_path, "transient")
    assert state["ready"] is False
    assert "not marked as a steady-iteration result" in str(state["reason"])


def test_summary_counts_unique_endpoints_and_ready_sequences(tmp_path: Path) -> None:
    write_marker(tmp_path, "a")
    write_marker(tmp_path, "b")
    plan = {
        "source_doi": "doi",
        "sequences": [
            {
                "sequence_id": "a_to_b",
                "family": "temperature",
                "source_condition_id": "a",
                "target_condition_id": "b",
            },
            {
                "sequence_id": "b_to_c",
                "family": "temperature",
                "source_condition_id": "b",
                "target_condition_id": "c",
            },
        ],
    }
    summary = build_summary(tmp_path, plan)
    assert summary["unique_endpoint_count"] == 3
    assert summary["ready_endpoint_count"] == 2
    assert summary["ready_sequence_count"] == 1
    assert summary["waiting_sequence_count"] == 1
    output = tmp_path / "out"
    write_outputs(summary, output)
    assert (output / "summary.json").is_file()
    assert "a_to_b" in (output / "P418_热阶跃端点准备情况_CN.md").read_text(encoding="utf-8")
