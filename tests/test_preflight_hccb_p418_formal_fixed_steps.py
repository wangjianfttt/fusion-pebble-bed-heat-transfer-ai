import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/preflight_hccb_p418_formal_fixed_steps.py"


def load_module():
    spec = importlib.util.spec_from_file_location("formal_step_preflight", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_current_formal_fixed_step_plan_passes_preflight():
    module = load_module()
    payload = module.build_preflight(
        ROOT / "parameters/hccb_p418_transient_step_plan.json",
        ROOT / "results/hccb_p418_step_endpoint_readiness_60/endpoint_field_check.json",
        ROOT / "results/hccb_p418_observation_duration/summary.json",
        ROOT
        / "results/hccb_p418_thermal_timestep_sensitivity/thermal_timestep_sensitivity.json",
    )
    assert payload["status"] == "ready_for_300_s_duration_approval_not_submitted"
    assert payload["sequence_count"] == 12
    assert payload["unique_endpoint_count"] == 11
    assert payload["duration_s"] == 300.0
    assert payload["full_field_snapshot_count"] == 56
    assert payload["formal_solver_submitted"] is False
    assert all(payload["checks"].values())


def test_current_plan_contains_no_new_physical_parameters():
    plan = json.loads(
        (ROOT / "parameters/hccb_p418_transient_step_plan.json").read_text(
            encoding="utf-8"
        )
    )
    assert plan["new_physical_parameters"] == []
    assert len(plan["sequences"]) == 12
