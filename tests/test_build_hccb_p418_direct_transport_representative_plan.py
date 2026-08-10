from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_direct_transport_representative_plan.py"


def test_builds_no_solver_representative_plan(tmp_path: Path) -> None:
    completed = subprocess.run(
        ["python3", str(SCRIPT), "--output-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    plan = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
    assert plan["openfoam_solver_started_by_this_plan"] is False
    assert plan["formal_solver_submission_approved"] is False
    assert plan["same_physics"]["new_fitted_parameters"] == []
    assert plan["same_physics"]["geometry_mesh_boundaries_operating_conditions_changed"] is False
    assert plan["stage_1_representative_smoke"]["target_end_time_s"] == 0.01
    assert plan["stage_1_representative_smoke"]["initial_delta_t_s"] == 1.0e-7
    assert plan["stage_1_representative_smoke"]["mpi_ranks"] == 32
    assert plan["stage_1_representative_smoke"]["fixed_flow_reference_time_points"] == 1001
    assert plan["stage_1_representative_smoke"]["fixed_flow_reference_interpolation_used"] is False
    assert plan["stage_2_manuscript_comparison"]["approved"] is False
    assert plan["stage_2_manuscript_comparison"]["automatic_submission"] is False
    assert plan["solver_submission_requires_exact_phrase"] == "批准短算"
    text = (tmp_path / "PLAN_CN.md").read_text(encoding="utf-8")
    assert "不能单独支撑完整热响应结论" in text
    assert "没有新拟合参数" in text


def test_direct_transport_uses_registered_equations() -> None:
    source = (
        ROOT
        / "solver_extensions"
        / "hccbHeliumTransport"
        / "hccbHeliumTransportI.H"
    ).read_text(encoding="utf-8")
    properties = (
        ROOT
        / "solver_extensions"
        / "hccbHeliumTransport"
        / "physicalProperties.example"
    ).read_text(encoding="utf-8")
    assert "viscosityCoefficient_" in source
    assert "conductivityPressureCoefficient_" in source
    assert "conductivityPressureTemperatureExponent_" in source
    assert "viscosityCoefficient                    0.4646;" in properties
    assert "viscosityTemperatureExponent            0.66;" in properties
    assert "conductivityCoefficient                 0.1448;" in properties
    assert "conductivityPressureCoefficient         2.5e-3;" in properties
    assert "conductivityPressureExponent            1.17;" in properties
    assert "conductivityPressureTemperatureExponent -1.85;" in properties
