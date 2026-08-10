import importlib.util
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLANNER = ROOT / "code/plan_hccb_p418_end_to_end_research.py"
FORMAL_RUNNER = ROOT / "code/run_hccb_p418_formal_calculations.sh"


def load_module():
    specification = importlib.util.spec_from_file_location("p418_plan", PLANNER)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def touch_markers(root: Path, relative_root: str, marker: str, count: int) -> None:
    for index in range(count):
        case = root / relative_root / f"case_{index:02d}"
        case.mkdir(parents=True, exist_ok=True)
        (case / marker).write_text("{}\n", encoding="utf-8")


def test_plan_reads_progress_without_starting_commands(tmp_path: Path) -> None:
    module = load_module()
    touch_markers(
        tmp_path,
        "hccb_dense_cht_p418_60_sourceflow_r3",
        "formal_sample_complete.json",
        14,
    )
    touch_markers(
        tmp_path,
        "hccb_p418_physical_steps_12",
        "step_response_complete.json",
        2,
    )
    plan = module.build_plan(tmp_path)
    assert plan["current_progress"]["steady"] == "14/60"
    assert plan["current_progress"]["fixed_hydrodynamics_steps"] == "2/12"
    assert plan["current_progress"]["fully_coupled_steps"] == "0/12"
    assert (
        plan["current_progress"]["high_re_fixed_vs_fully_coupled_comparison"]
        is False
    )
    assert plan["solver_or_training_started"] is False
    assert plan["new_physical_parameters"] == []
    assert len(plan["stages"]) == 12
    assert not (tmp_path / "results").exists()


def test_plan_uses_last_remote_status_when_large_fields_are_not_local(
    tmp_path: Path,
) -> None:
    status = tmp_path / "results/hccb_p418_model_data_preparation/summary.json"
    status.parent.mkdir(parents=True)
    status.write_text(
        """
{
  "states": {
    "steady": {"completed_count": 14},
    "fixed_hydrodynamics_thermal_steps": {"completed_count": 0},
    "fully_coupled_flow_heat_steps": {"completed_count": 0}
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    module = load_module()
    plan = module.build_plan(tmp_path)
    assert plan["current_progress"]["steady"] == "14/60"
    assert plan["current_progress"]["fixed_hydrodynamics_steps"] == "0/12"


def test_plan_reports_workstation_pause_marker(tmp_path: Path) -> None:
    marker = tmp_path / "control/PAUSE_NEW_P418_CASES_FOR_CLOUD_MIGRATION"
    marker.parent.mkdir(parents=True)
    marker.write_text("paused\n", encoding="utf-8")
    module = load_module()
    plan = module.build_plan(tmp_path)
    assert plan["workstation_pause_marker_present"] is True
    assert plan["stages"][1]["status"] == "等待易算云代表工况和统一提交"


def test_high_re_stages_follow_model_freezing() -> None:
    module = load_module()
    plan = module.build_plan(ROOT)
    names = [stage["name"] for stage in plan["stages"]]
    full_model = names.index("全耦合图--Transformer训练与损失组合选择")
    seed202 = names.index("seed202独立颗粒装填及模型选择")
    high_re_fixed = names.index("6组高雷诺数组合的固定流场独立测试")
    high_re_full = names.index("6组高雷诺数组合的全耦合独立测试")
    assert full_model < high_re_fixed
    assert seed202 < high_re_fixed
    assert high_re_fixed < high_re_full
    high_re_stage = plan["stages"][high_re_fixed]
    assert "只推理不再训练" in high_re_stage["compute"]


def test_formal_dry_run_calls_detailed_planner(tmp_path: Path) -> None:
    project = tmp_path / "project"
    code = project / "code"
    code.mkdir(parents=True)
    planner_copy = code / PLANNER.name
    planner_copy.write_bytes(PLANNER.read_bytes())
    environment = os.environ.copy()
    environment.update({"ROOT": str(project), "P418_PYTHON": "python3"})
    result = subprocess.run(
        ["bash", str(FORMAL_RUNNER)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "P418球床流动--换热与融合模型完整计算顺序" in result.stdout
    assert "本程序只读取现有文件，不启动求解或训练" in result.stdout
    assert "dry run only" in result.stdout
    assert list(project.iterdir()) == [code]
