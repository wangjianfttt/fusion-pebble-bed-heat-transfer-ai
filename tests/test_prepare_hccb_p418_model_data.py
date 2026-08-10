from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from prepare_hccb_p418_model_data import build_plan  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(tmp_path: Path, complete: bool) -> Path:
    steady_ids = ["case_a", "case_b"]
    sequence_ids = ["step_a", "step_b"]
    steady_root = tmp_path / "steady"
    write_json(
        steady_root / "matrix_manifest.json",
        {"published_conditions": [{"condition_id": value} for value in steady_ids]},
    )
    plan = {
        "sequences": [{"sequence_id": value} for value in sequence_ids],
    }
    write_json(tmp_path / "fixed_plan.json", plan)
    write_json(tmp_path / "coupled_plan.json", plan)
    for index, identifier in enumerate(steady_ids):
        case = steady_root / identifier
        case.mkdir(parents=True)
        if complete or index == 0:
            write_json(case / "formal_sample_complete.json", {})
    for root_name, marker in (
        ("fixed", "step_response_complete.json"),
        ("coupled", "fully_coupled_step_response_complete.json"),
    ):
        for index, identifier in enumerate(sequence_ids):
            case = tmp_path / root_name / identifier
            case.mkdir(parents=True)
            if complete or index == 0:
                write_json(case / marker, {})
    for path in (
        tmp_path / "steady_dataset/dataset_index.json",
        tmp_path / "steady_dataset/shared_mesh_topology.npz",
        tmp_path / "subface.npz",
        tmp_path / "regional.npz",
        tmp_path / "model_geometry.npz",
        tmp_path / "steady_summary.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ready")
    config = {
        "new_physical_parameters": [],
        "steady": {
            "source_root": "steady",
            "completion_marker": "formal_sample_complete.json",
            "expected_case_count": 2,
            "postprocess_runner": "code/run_hccb_p418_60_postprocess.sh",
            "dataset_root": "steady_dataset",
            "postprocess_summary": "steady_summary.json",
        },
        "fixed_hydrodynamics_thermal_steps": {
            "source_root": "fixed",
            "completion_marker": "step_response_complete.json",
            "plan": "fixed_plan.json",
            "integrated_observable_history_kind": "physical_step_response",
            "result_root": "fixed_results",
        },
        "fully_coupled_flow_heat_steps": {
            "source_root": "coupled",
            "completion_marker": "fully_coupled_step_response_complete.json",
            "plan": "coupled_plan.json",
            "integrated_observable_history_kind": "fully_coupled_flow_heat_response",
            "result_root": "coupled_results",
        },
        "shared_inputs": {
            "shared_topology": "steady_dataset/shared_mesh_topology.npz",
            "steady_dataset_index": "steady_dataset/dataset_index.json",
            "subface_geometry": "subface.npz",
            "regional_topology": "regional.npz",
            "model_geometry": "model_geometry.npz",
        },
        "integrated_observable_exporter": "code/export_hccb_p418_transient_observables.py",
        "regional_sequence_exporter": "code/export_hccb_p418_step_regional_sequences.py",
    }
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    return config_path


def test_incomplete_data_are_reported_without_training_commands(tmp_path: Path) -> None:
    plan = build_plan(tmp_path, fixture(tmp_path, complete=False))
    assert plan["status"] == "p418_model_data_waiting_for_openfoam_outputs"
    assert plan["states"]["steady"]["completed_count"] == 1
    assert plan["states"]["fixed_hydrodynamics_thermal_steps"]["completed_count"] == 1
    assert plan["states"]["fully_coupled_flow_heat_steps"]["completed_count"] == 1
    assert not any(plan["stage_ready_to_run"].values())
    assert plan["starts_model_training"] is False


def test_complete_data_create_three_data_only_stages(tmp_path: Path) -> None:
    plan = build_plan(tmp_path, fixture(tmp_path, complete=True))
    assert all(plan["stage_ready_to_run"].values())
    assert plan["fixed_and_fully_coupled_endpoint_pairs_identical"] is True
    assert plan["new_physical_parameters"] == []
    for records in plan["commands"].values():
        for record in records:
            assert record["starts_model_training"] is False
            assert not any(Path(token).name.startswith("train_") for token in record["argv"])


def test_pipeline_rejects_different_step_pairs(tmp_path: Path) -> None:
    config_path = fixture(tmp_path, complete=True)
    write_json(tmp_path / "coupled_plan.json", {"sequences": [{"sequence_id": "other"}]})
    try:
        build_plan(tmp_path, config_path)
    except ValueError as error:
        assert "same endpoint pairs" in str(error)
    else:
        raise AssertionError("different fixed and fully coupled step pairs were accepted")


def test_pipeline_rejects_same_id_with_different_endpoint(tmp_path: Path) -> None:
    config_path = fixture(tmp_path, complete=True)
    write_json(
        tmp_path / "fixed_plan.json",
        {
            "sequences": [
                {"sequence_id": "step_a", "source_condition_id": "a", "target_condition_id": "b"},
                {"sequence_id": "step_b", "source_condition_id": "b", "target_condition_id": "c"},
            ]
        },
    )
    write_json(
        tmp_path / "coupled_plan.json",
        {
            "sequences": [
                {"sequence_id": "step_a", "source_condition_id": "a", "target_condition_id": "b"},
                {"sequence_id": "step_b", "source_condition_id": "b", "target_condition_id": "d"},
            ]
        },
    )
    try:
        build_plan(tmp_path, config_path)
    except ValueError as error:
        assert "same endpoint pairs" in str(error)
    else:
        raise AssertionError("different source-target endpoints were accepted")
