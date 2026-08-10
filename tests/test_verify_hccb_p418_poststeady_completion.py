import hashlib
import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/verify_hccb_p418_poststeady_completion.py"
FORMAL = ROOT / "code/run_hccb_p418_formal_calculations.sh"
FORMAL_LABELS = (
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
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_record(tmp_path: Path) -> Path:
    selected_plan = tmp_path / "formal_step_plan.json"
    timestep_summary = tmp_path / "thermal_timestep_sensitivity.json"
    timestep_gci = tmp_path / "thermal_timestep_gci.csv"
    timestep_table = tmp_path / "generated_timestep_sensitivity.tex"
    sources = tmp_path / "model_sources.json"
    sensitivity = tmp_path / "loss_weight_summary.json"
    fused_table = tmp_path / "generated_fused_chain_results.tex"
    fused_summary = tmp_path / "fused_chain_table_summary.json"
    transient_cost_table = tmp_path / "generated_transient_cost.tex"
    transient_cost_summary = tmp_path / "transient_cost_table.json"
    time_step_schedule = [
        {"start_s": 0.0, "end_s": 0.1, "delta_t_s": 1.0e-5},
        {"start_s": 0.1, "end_s": 1.0, "delta_t_s": 5.0e-4},
        {"start_s": 1.0, "end_s": 25.0, "delta_t_s": 1.0e-2},
        {"start_s": 25.0, "end_s": 300.0, "delta_t_s": 0.125},
    ]
    field_write_schedule = [
        {"start_s": 0.0, "end_s": 0.1, "interval_s": 0.005},
        {"start_s": 0.1, "end_s": 1.0, "interval_s": 0.1},
        {"start_s": 1.0, "end_s": 5.0, "interval_s": 0.4},
        {"start_s": 5.0, "end_s": 25.0, "interval_s": 4.0},
        {"start_s": 25.0, "end_s": 300.0, "interval_s": 25.0},
    ]
    selected_plan.write_text(
        json.dumps(
            {
                "numerical_time_design": {
                    "delta_t_s": 1.0e-5,
                    "time_step_schedule": time_step_schedule,
                    "field_write_schedule": field_write_schedule,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    timestep_summary.write_text(
        json.dumps(
            {
                "status": "completed_p418_thermal_timestep_sensitivity",
                "formal_selection_rule": "finest_completed_predeclared_step",
                "selected_delta_t_s": 1.0e-5,
                "new_physical_parameters": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    timestep_gci.write_text(
        "signal,quantity,fine_gci_fraction\nT,endpoint,0.01\n", encoding="utf-8"
    )
    timestep_table.write_text(
        "\\begin{table*}time step\\end{table*}\n", encoding="utf-8"
    )
    sources.write_text('{"status":"sources"}\n', encoding="utf-8")
    sensitivity.write_text('{"status":"sensitivity"}\n', encoding="utf-8")
    fused_table.write_text("\\begin{table*}\\end{table*}\n", encoding="utf-8")
    fused_summary.write_text('{"status":"fused-table"}\n', encoding="utf-8")
    transient_cost_table.write_text(
        "\\begin{table*}cost\\end{table*}\n", encoding="utf-8"
    )
    transient_cost_summary.write_text('{"status":"transient-cost"}\n', encoding="utf-8")
    formal_files = []
    for label in FORMAL_LABELS:
        path = tmp_path / f"{label}.dat"
        path.write_text(f"{label}\n", encoding="utf-8")
        formal_files.append(
            {"label": label, "path": str(path), "sha256": digest(path)}
        )
    record = tmp_path / "complete.json"
    record.write_text(
        json.dumps(
            {
                "status": "completed_p418_poststeady_heat_transfer_pipeline",
                "selected_delta_t_s": 1.0e-5,
                "selected_time_step_schedule": time_step_schedule,
                "selected_field_write_schedule": field_write_schedule,
                "selected_timestep_plan": str(selected_plan),
                "selected_timestep_plan_sha256": digest(selected_plan),
                "thermal_timestep_sensitivity_summary": str(timestep_summary),
                "thermal_timestep_sensitivity_summary_sha256": digest(timestep_summary),
                "thermal_timestep_gci": str(timestep_gci),
                "thermal_timestep_gci_sha256": digest(timestep_gci),
                "thermal_timestep_manuscript_table": str(timestep_table),
                "thermal_timestep_manuscript_table_sha256": digest(timestep_table),
                "cross_packing_seed101_model_sources": str(sources),
                "cross_packing_seed101_model_sources_sha256": digest(sources),
                "steady_loss_weight_sensitivity": str(sensitivity),
                "steady_loss_weight_sensitivity_sha256": digest(sensitivity),
                "fused_chain_manuscript_table": str(fused_table),
                "fused_chain_manuscript_table_sha256": digest(fused_table),
                "fused_chain_manuscript_table_summary": str(fused_summary),
                "fused_chain_manuscript_table_summary_sha256": digest(fused_summary),
                "transient_cost_manuscript_table": str(transient_cost_table),
                "transient_cost_manuscript_table_sha256": digest(transient_cost_table),
                "transient_cost_summary": str(transient_cost_summary),
                "transient_cost_summary_sha256": digest(transient_cost_summary),
                "formal_result_files": formal_files,
                "new_physical_parameters": [],
            }
        ),
        encoding="utf-8",
    )
    return record


def test_current_record_passes(tmp_path: Path) -> None:
    record = make_record(tmp_path)
    output = tmp_path / "checked.json"
    completed = subprocess.run(
        ["python3", str(SCRIPT), "--record", str(record), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "p418_poststeady_completion_is_current"
    assert len(payload["checked_results"]) == 43
    assert output.is_file()


def test_changed_result_is_rejected(tmp_path: Path) -> None:
    record = make_record(tmp_path)
    payload = json.loads(record.read_text(encoding="utf-8"))
    Path(payload["steady_loss_weight_sensitivity"]).write_text("changed\n", encoding="utf-8")
    completed = subprocess.run(
        ["python3", str(SCRIPT), "--record", str(record)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "recorded result changed" in completed.stderr


def test_old_record_without_weight_sensitivity_is_rejected(tmp_path: Path) -> None:
    record = make_record(tmp_path)
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload.pop("steady_loss_weight_sensitivity")
    payload.pop("steady_loss_weight_sensitivity_sha256")
    record.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(["python3", str(SCRIPT), "--record", str(record)], check=True)


def test_old_record_without_fused_chain_table_is_rejected(tmp_path: Path) -> None:
    record = make_record(tmp_path)
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload.pop("fused_chain_manuscript_table")
    payload.pop("fused_chain_manuscript_table_sha256")
    record.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(["python3", str(SCRIPT), "--record", str(record)], check=True)


def test_old_record_without_timestep_gci_is_rejected(tmp_path: Path) -> None:
    record = make_record(tmp_path)
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload.pop("thermal_timestep_gci")
    payload.pop("thermal_timestep_gci_sha256")
    record.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(["python3", str(SCRIPT), "--record", str(record)], check=True)


def test_old_record_without_principal_results_is_rejected(tmp_path: Path) -> None:
    record = make_record(tmp_path)
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload.pop("formal_result_files")
    record.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(["python3", str(SCRIPT), "--record", str(record)], check=True)


def test_changed_principal_result_is_rejected(tmp_path: Path) -> None:
    record = make_record(tmp_path)
    payload = json.loads(record.read_text(encoding="utf-8"))
    Path(payload["formal_result_files"][0]["path"]).write_text(
        "changed\n", encoding="utf-8"
    )
    completed = subprocess.run(
        ["python3", str(SCRIPT), "--record", str(record)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "recorded result changed" in completed.stderr


def test_mismatched_selected_time_step_is_rejected(tmp_path: Path) -> None:
    record = make_record(tmp_path)
    payload = json.loads(record.read_text(encoding="utf-8"))
    plan = Path(payload["selected_timestep_plan"])
    plan.write_text(
        json.dumps({"numerical_time_design": {"delta_t_s": 0.5}}) + "\n",
        encoding="utf-8",
    )
    payload["selected_timestep_plan_sha256"] = digest(plan)
    record.write_text(json.dumps(payload), encoding="utf-8")
    completed = subprocess.run(
        ["python3", str(SCRIPT), "--record", str(record)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "selected time step differs" in completed.stderr


def test_old_record_without_staged_time_schedule_is_rejected(tmp_path: Path) -> None:
    record = make_record(tmp_path)
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload.pop("selected_time_step_schedule")
    record.write_text(json.dumps(payload), encoding="utf-8")
    completed = subprocess.run(
        ["python3", str(SCRIPT), "--record", str(record)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "staged time-step schedule is missing" in completed.stderr


def test_formal_route_checks_record_before_reuse() -> None:
    text = FORMAL.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(FORMAL)], check=True)
    verify = text.index("verify_hccb_p418_poststeady_completion.py")
    reuse = text.index("reuse completed seed101 post-steady calculations")
    assert verify < reuse
    assert "steady_loss_weight_sensitivity" in (
        ROOT / "code/run_hccb_p418_poststeady_pipeline.sh"
    ).read_text(encoding="utf-8")
