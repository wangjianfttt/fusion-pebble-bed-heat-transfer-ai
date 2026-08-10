from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from check_hccb_p418_final_scientific_requirements import (  # noqa: E402
    fully_coupled_scope_limit_complete,
    json_matches,
    summarize_remaining_dependencies,
    steady_extrapolation_evidence,
    steady_physics_evidence,
    three_mesh_sensitivity_evidence,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_current_steady_physics_path_is_recognized(tmp_path: Path) -> None:
    path = tmp_path / "results/hccb_p418_sourceflow_complete_physics_60/summary.json"
    write_json(
        path,
        {
            "status": "completed_p418_case_physics_summarized",
            "completed_case_count": 60,
            "maximum_relative_mass_difference": 1.0e-8,
            "maximum_relative_energy_difference": 5.0e-5,
            "new_physical_parameters": [],
        },
    )

    complete, selected = steady_physics_evidence(tmp_path)

    assert complete is True
    assert selected == path


def test_all_three_extrapolation_splits_are_required(tmp_path: Path) -> None:
    path = (
        tmp_path
        / "results/hccb_p418_60_corrected_20260731_model_comparison_100epoch"
        / "corrected_result_assembly.json"
    )
    methods = ["response_surface", "pinn", "pinn_data_only", "graph", "transolver"]
    records = [
        {"split": split, "method": method}
        for split in (
            "temperature_extrapolation",
            "velocity_extrapolation",
            "heat_source_extrapolation",
        )
        for method in methods
    ]
    write_json(
        path,
        {
            "status": "corrected_steady_result_assembly_complete",
            "records": records,
            "new_physical_parameters": [],
        },
    )

    complete, selected = steady_extrapolation_evidence(tmp_path)
    assert complete is True
    assert selected == path

    write_json(
        path,
        {
            "status": "corrected_steady_result_assembly_complete",
            "records": [row for row in records if row["split"] != "velocity_extrapolation"],
            "new_physical_parameters": [],
        },
    )
    complete, _ = steady_extrapolation_evidence(tmp_path)
    assert complete is False


def test_three_independent_coupled_failures_resolve_scope(tmp_path: Path) -> None:
    path = tmp_path / "scope.json"
    write_json(
        path,
        {
            "status": "P418_SCOPE_LIMITS_EVIDENCE_SYNCED",
            "records": [
                {
                    "status": "failed_solver_exit_propagated",
                    "slurm_state": "FAILED",
                    "job_id": str(job_id),
                }
                for job_id in (14721, 14722, 14723)
            ],
        },
    )
    assert fully_coupled_scope_limit_complete(path) is True

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"] = payload["records"][:2]
    write_json(path, payload)
    assert fully_coupled_scope_limit_complete(path) is False


def test_remaining_work_has_no_hidden_physics_gap() -> None:
    requirements = [
        {"group": "三维物理计算", "name": "60个稳态工况", "complete": True},
        {
            "group": "模型比较",
            "name": "瞬态DMDc、三种图模型和扩散修正比较",
            "complete": False,
        },
        {"group": "论文成品", "name": "瞬态模型图", "complete": False},
        {"group": "论文成品", "name": "英文论文PDF", "complete": False},
    ]

    summary = summarize_remaining_dependencies(requirements)

    assert summary["unfinished_count"] == 3
    assert summary["unfinished_physics_calculations"] == []
    assert summary["waiting_for_model_chain"] == [
        "瞬态DMDc、三种图模型和扩散修正比较"
    ]
    assert summary["generated_after_model_chain"] == ["瞬态模型图", "英文论文PDF"]
    assert summary["other_unfinished_items"] == []


def test_preselection_summary_is_not_a_final_result(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    write_json(
        path,
        {
            "status": "completed_p418_physical_step_model_comparison",
            "strict_split_loss_balancing_stage": "registered_preselection",
        },
    )
    assert not json_matches(
        path,
        status="completed_p418_physical_step_model_comparison",
        strict_split_loss_balancing_stage="validation_selected",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["strict_split_loss_balancing_stage"] = "validation_selected"
    write_json(path, payload)
    assert json_matches(
        path,
        status="completed_p418_physical_step_model_comparison",
        strict_split_loss_balancing_stage="validation_selected",
    )


def test_three_mesh_requires_formal_summary_and_source_tables(tmp_path: Path) -> None:
    root = tmp_path / "results/hccb_p418_three_mesh_cht_sensitivity"
    summary = root / "summary.json"
    write_json(
        summary,
        {
            "status": "completed_three_mesh_p418_cht_comparison",
            "new_physical_parameters": [],
            "mesh_levels": [
                {
                    "mesh_level": level,
                    "fluid_cells": fluid,
                    "solid_cells": solid,
                    "total_cells": fluid + solid,
                    "fluid_basic_check_passes": True,
                    "solid_basic_check_passes": True,
                }
                for level, fluid, solid in (
                    ("coarse", 160989, 200162),
                    ("medium", 432384, 515540),
                    ("fine", 858419, 1011645),
                )
            ],
            "grid_convergence": [
                {"metric": metric}
                for metric in (
                    "pressure_drop_Pa",
                    "outlet_temperature_change_K",
                    "solid_maximum_temperature_change_K",
                    "cooling_wall_heat_fraction",
                )
            ],
        },
    )
    complete, selected = three_mesh_sensitivity_evidence(tmp_path)
    assert complete is False
    assert selected == summary

    (root / "engineering_observables.csv").write_text("mesh,value\n", encoding="utf-8")
    (root / "mesh_gci.csv").write_text("metric,value\n", encoding="utf-8")
    write_json(
        root / "formal_recovery_verification.json",
        {
            "status": "verified_formal_p418_three_mesh_recovery",
            "file_sha256": {
                "summary.json": sha256(summary),
                "engineering_observables.csv": sha256(
                    root / "engineering_observables.csv"
                ),
                "mesh_gci.csv": sha256(root / "mesh_gci.csv"),
            },
            "checks": {"formal_files": True},
            "new_physical_parameters": [],
        },
    )
    complete, _ = three_mesh_sensitivity_evidence(tmp_path)
    assert complete is True

    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["mesh_levels"][0]["solid_basic_check_passes"] = False
    write_json(summary, payload)
    complete, _ = three_mesh_sensitivity_evidence(tmp_path)
    assert complete is False
