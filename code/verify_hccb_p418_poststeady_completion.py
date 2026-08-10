#!/usr/bin/env python3
"""Check that a post-steady completion record still describes existing results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_STATUS = "completed_p418_poststeady_heat_transfer_pipeline"
FILE_FIELDS = (
    ("selected_timestep_plan", "selected_timestep_plan_sha256"),
    (
        "thermal_timestep_sensitivity_summary",
        "thermal_timestep_sensitivity_summary_sha256",
    ),
    ("thermal_timestep_gci", "thermal_timestep_gci_sha256"),
    (
        "thermal_timestep_manuscript_table",
        "thermal_timestep_manuscript_table_sha256",
    ),
    ("cross_packing_seed101_model_sources", "cross_packing_seed101_model_sources_sha256"),
    ("steady_loss_weight_sensitivity", "steady_loss_weight_sensitivity_sha256"),
    ("fused_chain_manuscript_table", "fused_chain_manuscript_table_sha256"),
    (
        "fused_chain_manuscript_table_summary",
        "fused_chain_manuscript_table_summary_sha256",
    ),
    ("transient_cost_manuscript_table", "transient_cost_manuscript_table_sha256"),
    ("transient_cost_summary", "transient_cost_summary_sha256"),
)

FORMAL_RESULT_LABELS = {
    "preflight_formal_consistency",
    "physical_and_model_source_summary",
    "physical_and_model_source_text",
    "completed_physics_csv",
    "steady_hotspot_summary",
    "steady_hotspot_csv",
    "steady_hotspot_movements_csv",
    "steady_final_window_summary",
    "steady_final_window_text",
    "mesh_sensitivity_summary",
    "mesh_sensitivity_gci",
    "mesh_sensitivity_table",
    "dimensionless_heat_summary",
    "pressure_correlation_summary",
    "same_source_correlation_text",
    "physical_response_figure",
    "regional_fidelity_text",
    "steady_model_comparison_csv",
    "steady_model_comparison_figure",
    "steady_performance_table",
    "thermal_regime_split_coverage",
    "steady_result_text",
    "native_cell_model_comparison",
    "native_cell_performance_table",
    "steady_seed_robustness_table",
    "transient_model_metrics",
    "transient_performance_table",
    "transient_performance_summary",
    "transient_model_figure",
    "transient_result_text",
    "transition_temperature_coverage",
    "transition_temperature_coverage_text",
    "steady_learning_curve",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_record(record_path: Path) -> dict:
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    if payload.get("status") != EXPECTED_STATUS:
        raise ValueError("post-steady completion status is missing or outdated")
    delta_t = payload.get("selected_delta_t_s")
    if not isinstance(delta_t, (int, float)) or delta_t <= 0:
        raise ValueError("selected thermal time step is missing")
    if payload.get("new_physical_parameters") != []:
        raise ValueError("completion record unexpectedly adds physical parameters")

    checked = []
    for path_field, hash_field in FILE_FIELDS:
        result_path = Path(payload.get(path_field, ""))
        expected_hash = payload.get(hash_field)
        if not result_path.is_file():
            raise ValueError(f"recorded result is missing: {result_path}")
        actual_hash = sha256(result_path)
        if actual_hash != expected_hash:
            raise ValueError(f"recorded result changed: {result_path}")
        checked.append(
            {"field": path_field, "path": str(result_path.resolve()), "sha256": actual_hash}
        )

    formal_files = payload.get("formal_result_files")
    if not isinstance(formal_files, list):
        raise ValueError("principal formal-result file list is missing")
    labels = [item.get("label") for item in formal_files if isinstance(item, dict)]
    if len(labels) != len(formal_files) or set(labels) != FORMAL_RESULT_LABELS:
        raise ValueError("principal formal-result file list is incomplete or duplicated")
    for item in formal_files:
        result_path = Path(item.get("path", ""))
        expected_hash = item.get("sha256")
        if not result_path.is_file():
            raise ValueError(f"recorded result is missing: {result_path}")
        actual_hash = sha256(result_path)
        if actual_hash != expected_hash:
            raise ValueError(f"recorded result changed: {result_path}")
        checked.append(
            {
                "field": item["label"],
                "path": str(result_path.resolve()),
                "sha256": actual_hash,
            }
        )

    plan_path = Path(payload["selected_timestep_plan"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    numerical_time = plan.get("numerical_time_design", {})
    plan_delta_t = numerical_time.get("delta_t_s")
    if plan_delta_t != delta_t:
        raise ValueError("selected time step differs between the completion record and plan")
    time_step_schedule = payload.get("selected_time_step_schedule")
    if not isinstance(time_step_schedule, list) or not time_step_schedule:
        raise ValueError("selected staged time-step schedule is missing")
    if time_step_schedule != numerical_time.get("time_step_schedule"):
        raise ValueError("selected staged time-step schedule differs from the plan")
    field_write_schedule = payload.get("selected_field_write_schedule")
    if not isinstance(field_write_schedule, list) or not field_write_schedule:
        raise ValueError("selected field-write schedule is missing")
    if field_write_schedule != numerical_time.get("field_write_schedule"):
        raise ValueError("selected field-write schedule differs from the plan")

    summary_path = Path(payload["thermal_timestep_sensitivity_summary"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed_p418_thermal_timestep_sensitivity":
        raise ValueError("time-step comparison is incomplete")
    if summary.get("formal_selection_rule") != "finest_completed_predeclared_step":
        raise ValueError("time-step comparison does not use the declared finest-step rule")
    if summary.get("selected_delta_t_s") != delta_t:
        raise ValueError("selected time step differs between the completion record and comparison")
    if summary.get("new_physical_parameters") != []:
        raise ValueError("time-step comparison unexpectedly adds physical parameters")

    return {
        "status": "p418_poststeady_completion_is_current",
        "record": str(record_path.resolve()),
        "selected_delta_t_s": float(delta_t),
        "selected_time_step_schedule": time_step_schedule,
        "selected_field_write_schedule": field_write_schedule,
        "checked_results": checked,
        "new_physical_parameters": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_record(args.record)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
