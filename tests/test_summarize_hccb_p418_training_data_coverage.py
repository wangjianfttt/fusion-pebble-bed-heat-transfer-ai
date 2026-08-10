from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "summarize_hccb_p418_training_data_coverage.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_arrays() -> dict[str, np.ndarray]:
    return {
        "fluid_cell_centroid_m": np.zeros((2, 3)),
        "fluid_cell_volume_m3": np.ones(2),
        "fluid_internal_face_owner": np.array([0]),
        "fluid_internal_face_neighbour": np.array([1]),
        "solid_cell_centroid_m": np.ones((1, 3)),
        "solid_cell_volume_m3": np.ones(1),
        "solid_internal_face_owner": np.array([], dtype=np.int64),
        "solid_internal_face_neighbour": np.array([], dtype=np.int64),
        "fluid_boundary_face_owner": np.array([0, 1]),
        "fluid_boundary_face_patch": np.array([0, 1]),
        "fluid_boundary_face_centroid_m": np.zeros((2, 3)),
        "fluid_boundary_face_area_vector_outward_m2": np.ones((2, 3)),
        "fluid_boundary_face_area_m2": np.ones(2),
        "fluid_boundary_velocity_value_mask": np.ones(2, dtype=bool),
        "fluid_boundary_pressure_value_mask": np.ones(2, dtype=bool),
        "fluid_boundary_temperature_value_mask": np.ones(2, dtype=bool),
        "fluid_boundary_density_value_mask": np.ones(2, dtype=bool),
        "fluid_boundary_mass_flow_value_mask": np.ones(2, dtype=bool),
        "solid_boundary_face_owner": np.array([0]),
        "solid_boundary_face_patch": np.array([0]),
        "solid_boundary_face_centroid_m": np.zeros((1, 3)),
        "solid_boundary_face_area_vector_outward_m2": np.ones((1, 3)),
        "solid_boundary_face_area_m2": np.ones(1),
        "solid_boundary_temperature_value_mask": np.ones(1, dtype=bool),
        "interface_fluid_cell": np.array([1]),
        "interface_solid_cell": np.array([0]),
        "interface_face_centroid_m": np.zeros((1, 3)),
        "interface_area_vector_fluid_outward_m2": np.ones((1, 3)),
        "interface_face_area_m2": np.ones(1),
        "fluid_velocity_m_s": np.ones((2, 3)),
        "fluid_pressure_Pa": np.array([120010.0, 120000.0]),
        "fluid_temperature_K": np.array([300.0, 400.0]),
        "fluid_density_kg_m3": np.ones(2),
        "fluid_internal_face_mass_flow_kg_s": np.ones(1),
        "fluid_boundary_velocity_m_s": np.ones((2, 3)),
        "fluid_boundary_pressure_Pa": np.array([120010.0, 120000.0]),
        "fluid_boundary_temperature_K": np.array([300.0, 400.0]),
        "fluid_boundary_density_kg_m3": np.ones(2),
        "fluid_boundary_face_mass_flow_kg_s": np.array([-1.0, 1.0]),
        "solid_temperature_K": np.array([500.0]),
        "solid_boundary_temperature_K": np.array([500.0]),
    }


def write_registered_case(matrix: Path, condition_id: str, *, complete: bool) -> None:
    case = matrix / condition_id
    case.mkdir(parents=True)
    physical = {
        "operating_condition_id": condition_id,
        "inlet_velocity_m_s": 0.1,
        "inlet_temperature_K": 300.0,
        "solid_heat_source_W_m3": 4.85e6,
        "outlet_pressure_Pa": 120000.0,
        "cooling_wall_temperature_K": 635.0,
    }
    (case / "cht_smoke_metadata.json").write_text(
        json.dumps(physical) + "\n", encoding="utf-8"
    )
    if not complete:
        return

    sample_dir = case / "training_sample_200_schema3"
    sample_dir.mkdir()
    arrays = sample_arrays()
    sample = sample_dir / "fields_and_topology.npz"
    np.savez_compressed(sample, **arrays)
    sample_digest = digest(sample)
    metadata = {
        "schema_version": 3,
        "sample_id": f"{condition_id}_t200",
        "time": "200",
        "fluid_cells": 2,
        "solid_cells": 1,
        "interface_faces": 1,
        "fluid_patch_names": ["inlet", "outlet"],
        "solid_patch_names": ["coolingWall"],
        "physical_conditions": physical,
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "array_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
        "sample_sha256": sample_digest,
    }
    (sample_dir / "metadata.json").write_text(
        json.dumps(metadata) + "\n", encoding="utf-8"
    )
    marker = {
        "condition_id": condition_id,
        "time": "200",
        "steady_iteration_end": 200,
        "solver_time_semantics": "steady_iteration_index",
        "physical_time_s": None,
        "solver_finished": True,
        "relative_mass_difference": 1.0e-8,
        "relative_energy_difference": 2.0e-5,
        "training_sample": str(sample),
        "training_sample_sha256": sample_digest,
        "training_sample_schema_version": 3,
    }
    tail = case / "steady_final_window_175_to_200.json"
    tail.write_text(
        json.dumps(
            {
                "status": "formal_steady_final_window_measured",
                "condition_id": condition_id,
                "window_s": [175.0, 200.0],
                "engineering_changes": {
                    "outlet_temperature_K": {"start": 1, "end": 1, "absolute_change": 0, "relative_change": 0},
                    "solid_maximum_temperature_K": {"start": 1, "end": 1, "absolute_change": 0, "relative_change": 0},
                    "cooling_wall_power_W": {"start": 1, "end": 1, "absolute_change": 0, "relative_change": 0},
                    "outlet_enthalpy_flow_W": {"start": 1, "end": 1, "absolute_change": 0, "relative_change": 0},
                    "pressure_drop_Pa": {"start": 1, "end": 1, "absolute_change": 0, "relative_change": 0},
                    "final_relative_mass_difference": 0,
                },
                "full_field_available": True,
                "full_field_changes": [
                    {
                        "field_name": name,
                        "absolute_rms_change": 0,
                        "maximum_absolute_change": 0,
                        "relative_rms_change": 0,
                    }
                    for name in (
                        "fluid_temperature",
                        "solid_temperature",
                        "fluid_velocity_components",
                        "fluid_pressure",
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    marker.update(
        {
            "steady_final_window_status": "formal_steady_final_window_measured",
            "steady_final_window_summary": str(tail),
            "steady_final_window_summary_sha256": digest(tail),
            "steady_final_window_full_field_available": True,
        }
    )
    (case / "formal_sample_complete.json").write_text(
        json.dumps(marker) + "\n", encoding="utf-8"
    )
    result = {
        "solver_finished": True,
        "physical_conditions": {
            key: physical[key]
            for key in (
                "inlet_velocity_m_s",
                "inlet_temperature_K",
                "solid_heat_source_W_m3",
                "cooling_wall_temperature_K",
            )
        },
        "flow": {
            "pressure_drop_Pa": 10.0,
            "relative_mass_difference": 1.0e-8,
        },
        "temperature": {
            "outlet_average_K": 400.0,
            "solid_maximum_K": 500.0,
        },
        "heat_balance": {
            "solid_generated_power_W": 1.0,
            "cooling_wall_heat_flow_W": 0.5,
            "inlet_enthalpy_flow_W": -0.2,
            "outlet_enthalpy_flow_W": 0.7,
            "relative_energy_difference": 2.0e-5,
        },
        "all_reported_values_are_finite": True,
    }
    (case / "cht_result_summary_200.json").write_text(
        json.dumps(result) + "\n", encoding="utf-8"
    )


def run_checker(matrix: Path, output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--matrix-root",
            str(matrix),
            "--output-dir",
            str(output),
            "--expected-case-count",
            "2",
            *extra,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_partial_report_lists_available_fields_and_missing_case(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix"
    write_registered_case(matrix, "case_a", complete=True)
    write_registered_case(matrix, "case_b", complete=False)
    write_registered_case(
        matrix,
        "case_a.precomputed_input_cloud_recovery",
        complete=False,
    )
    output = tmp_path / "out"

    completed = run_checker(matrix, output, "--allow-partial", "--verify-file-checksums")

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "p418_partial_training_data_coverage_ready"
    assert summary["completed_case_count"] == 1
    assert summary["missing_condition_ids"] == ["case_b"]
    assert summary["solver_time_semantics"] == "steady_iteration_index"
    assert summary["physical_time_s"] is None
    assert summary["steady_iteration_column"] == "steady_iteration"
    assert "solid_hotspot_location_m" in summary["derived_from_predicted_fields"]
    assert "cooling_wall_face_heat_W" in summary["separate_postprocess_targets"]
    chinese = (output / "P418_训练数据字段完整性_CN.md").read_text(
        encoding="utf-8"
    )
    assert "1/2" in chinese
    assert "不是物理秒" in chinese
    assert "颗粒最高温度及位置" in chinese
    assert "P070" in chinese and "P071" in chinese
    rows = (output / "case_training_data_coverage.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert "steady_iteration" in rows[0]
    assert "solver_time_semantics" in rows[0]
    assert "physical_time_s" in rows[0]
    assert ",time," not in f",{rows[0]},"


def test_strict_mode_refuses_incomplete_matrix(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix"
    write_registered_case(matrix, "case_a", complete=True)
    write_registered_case(matrix, "case_b", complete=False)

    completed = run_checker(matrix, tmp_path / "out")

    assert completed.returncode != 0
    assert "completed formal cases 1/2" in completed.stderr
