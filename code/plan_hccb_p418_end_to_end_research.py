#!/usr/bin/env python3
"""Print the complete P418 calculation order without starting a solver or training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def count_markers(root: Path, relative_root: str, marker: str) -> int:
    base = root / relative_root
    if not base.is_dir():
        return 0
    return sum(1 for path in base.rglob(marker) if path.is_file())


def file_ready(root: Path, relative_path: str) -> bool:
    return (root / relative_path).is_file()


def recorded_data_count(root: Path, state_name: str) -> int:
    if state_name == "steady":
        coverage = (
            root
            / "results/hccb_p418_training_data_coverage_partial/summary.json"
        )
        if coverage.is_file():
            try:
                payload = json.loads(coverage.read_text(encoding="utf-8"))
                return int(payload["completed_case_count"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
    summary_path = (
        root / "results/hccb_p418_model_data_preparation/summary.json"
    )
    if not summary_path.is_file():
        return 0
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        return int(payload["states"][state_name]["completed_count"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 0


def stage_state(done: bool, prerequisites_done: bool) -> str:
    if done:
        return "已完成"
    if prerequisites_done:
        return "可在前序检查通过后运行"
    return "等待前序结果"


def build_plan(root: Path) -> dict[str, object]:
    workstation_pause_marker = file_ready(
        root, "control/PAUSE_NEW_P418_CASES_FOR_CLOUD_MIGRATION"
    )
    steady_count = count_markers(
        root,
        "hccb_dense_cht_p418_60_sourceflow_r3",
        "formal_sample_complete.json",
    )
    steady_count = max(steady_count, recorded_data_count(root, "steady"))
    fixed_count = count_markers(
        root,
        "hccb_p418_physical_steps_12",
        "step_response_complete.json",
    )
    fixed_count = max(
        fixed_count,
        recorded_data_count(root, "fixed_hydrodynamics_thermal_steps"),
    )
    fully_coupled_count = count_markers(
        root,
        "hccb_p418_fully_coupled_steps_12",
        "fully_coupled_step_response_complete.json",
    )
    fully_coupled_count = max(
        fully_coupled_count,
        recorded_data_count(root, "fully_coupled_flow_heat_steps"),
    )
    high_re_fixed_count = count_markers(
        root,
        "hccb_p418_high_re_independent_fixed_steps_6",
        "step_response_complete.json",
    )
    high_re_fully_coupled_count = count_markers(
        root,
        "hccb_p418_high_re_independent_fully_coupled_steps_6",
        "fully_coupled_step_response_complete.json",
    )

    parameters_ready = all(
        file_ready(root, path)
        for path in (
            "parameters/hccb_p418_physical_parameter_evidence_files.csv",
            "parameters/hccb_p418_ai_architecture_sources.json",
            "parameters/hccb_p418_transient_step_plan.json",
            "parameters/hccb_p418_fully_coupled_step_plan.json",
            "parameters/hccb_p418_high_re_independent_step_plan.json",
        )
    )
    steady_done = steady_count == 60
    fixed_done = (
        fixed_count == 12
        and file_ready(root, "results/hccb_p418_poststeady_pipeline_complete.json")
    )
    fully_coupled_timestep_done = file_ready(
        root,
        "results/hccb_p418_fully_coupled_timestep_sensitivity/"
        "fully_coupled_timestep_sensitivity.json",
    )
    fully_coupled_done = fully_coupled_count == 12
    fixed_vs_full_done = file_ready(
        root,
        "results/hccb_p418_fully_coupled_steps_12/"
        "fixed_vs_fully_coupled/summary.json",
    )
    full_model_selection_done = file_ready(
        root,
        "results/hccb_p418_fully_coupled_model_comparison/"
        "selected_loss_balancing_method.json",
    )
    full_model_final_done = any(
        (root / "results/hccb_p418_fully_coupled_model_comparison").glob(
            "*/final_summary.json"
        )
    )
    seed202_done = file_ready(
        root, "results/hccb_p418_cross_packing_seed202_complete.json"
    )
    architecture_frozen = file_ready(
        root,
        "results/hccb_p418_cross_packing_seed202_model_comparison/"
        "architecture_selection.json",
    )
    seed303_done = file_ready(
        root, "results/hccb_p418_cross_packing_seed303_complete.json"
    )
    seed303_final_done = any(
        (root / "results").glob(
            "hccb_p418_cross_packing_seed303_final_*/summary.json"
        )
    )
    high_re_fixed_done = high_re_fixed_count == 6
    high_re_full_done = high_re_fully_coupled_count == 6
    high_re_fixed_evaluation_done = file_ready(
        root,
        "results/hccb_p418_high_re_independent_fixed_model_evaluation/"
        "summary.json",
    )
    high_re_full_evaluation_done = file_ready(
        root,
        "results/hccb_p418_high_re_independent_fully_coupled_model_evaluation/"
        "summary.json",
    )
    high_re_comparison_done = file_ready(
        root,
        "results/hccb_p418_high_re_independent_model_comparison/summary.json",
    )
    high_re_fixed_complete = (
        high_re_fixed_done and high_re_fixed_evaluation_done
    )
    high_re_full_complete = (
        high_re_full_done
        and high_re_full_evaluation_done
        and high_re_comparison_done
    )
    formal_done = file_ready(
        root, "results/hccb_p418_formal_calculations_complete.json"
    )

    stages = [
        {
            "number": 1,
            "name": "文献参数、方程和模型设置复核",
            "status": stage_state(parameters_ready, True),
            "compute": "CPU，几分钟",
            "command": (
                "make p418-parameter-evidence && "
                "make p418-model-comparison-protocol"
            ),
            "output": "results/hccb_p418_source_summary.json",
            "why": "确认所有球床物性和边界条件都有文献来源。",
        },
        {
            "number": 2,
            "name": "seed101的60组稳态三维流动--换热计算",
            "status": (
                "等待易算云代表工况和统一提交"
                if workstation_pause_marker and not steady_done
                else stage_state(steady_done, parameters_ready)
            ),
            "progress": f"{steady_count}/60",
            "compute": "OpenFOAM，CPU并行",
            "command": "云端先跑1次和200次稳态迭代的代表工况，再提交剩余稳态数组任务",
            "output": "hccb_dense_cht_p418_60_sourceflow_r3/*/formal_sample_complete.json",
            "why": "给出稳态温度、速度、压力和局部热点的三维参考场。",
        },
        {
            "number": 3,
            "name": "固定流场近似下的12条物理热阶跃和主模型比较",
            "status": stage_state(fixed_done, steady_done),
            "progress": f"{fixed_count}/12",
            "compute": "OpenFOAM CPU + 单GPU模型训练",
            "command": "bash code/run_hccb_p418_poststeady_pipeline.sh",
            "output": "results/hccb_p418_poststeady_pipeline_complete.json",
            "why": "比较PINN、图--Transformer、POD、DMDc和扩散温度修正。",
        },
        {
            "number": 4,
            "name": "全流热耦合代表阶跃的时间步检查",
            "status": stage_state(fully_coupled_timestep_done, steady_done),
            "compute": "OpenFOAM，32个CPU核",
            "command": (
                "EXECUTE=1 bash "
                "code/run_hccb_p418_fully_coupled_timestep_sensitivity.sh"
            ),
            "output": (
                "results/hccb_p418_fully_coupled_timestep_sensitivity/"
                "fully_coupled_timestep_sensitivity.json"
            ),
            "why": "先确定全耦合曲线最初25 s所需时间步，再计算正式曲线。",
        },
        {
            "number": 5,
            "name": "全流热耦合的12条物理阶跃",
            "status": stage_state(
                fully_coupled_done, steady_done and fully_coupled_timestep_done
            ),
            "progress": f"{fully_coupled_count}/12",
            "compute": "OpenFOAM，CPU并行",
            "command": (
                "EXECUTE=1 bash code/run_hccb_p418_fully_coupled_step_responses.sh"
            ),
            "output": "results/hccb_p418_fully_coupled_steps_12",
            "why": "直接量化固定速度/压力场近似对换热瞬态造成的差别。",
        },
        {
            "number": 6,
            "name": "固定流场与全耦合曲线逐条比较",
            "status": stage_state(
                fixed_vs_full_done, fixed_done and fully_coupled_done
            ),
            "compute": "CPU，几分钟",
            "command": (
                "python3 code/compare_hccb_p418_fixed_and_fully_coupled_steps.py"
            ),
            "output": (
                "results/hccb_p418_fully_coupled_steps_12/"
                "fixed_vs_fully_coupled/summary.json"
            ),
            "why": "报告压降、出口温度、颗粒最高温度、壁面换热和能量差。",
        },
        {
            "number": 7,
            "name": "全耦合图--Transformer训练与损失组合选择",
            "status": stage_state(
                full_model_selection_done and full_model_final_done,
                fully_coupled_done,
            ),
            "compute": "单GPU",
            "command": (
                "EXECUTE=1 DEVICE=cuda bash "
                "code/run_hccb_p418_fully_coupled_model_stage.sh"
            ),
            "output": "results/hccb_p418_fully_coupled_model_comparison",
            "why": "只用训练和检查曲线选模型，独立测试曲线最后读取一次。",
        },
        {
            "number": 8,
            "name": "seed202独立颗粒装填及模型选择",
            "status": stage_state(seed202_done and architecture_frozen, steady_done),
            "compute": "OpenFOAM CPU + 单GPU",
            "command": "由code/run_hccb_p418_formal_calculations.sh统一调用",
            "output": (
                "results/hccb_p418_cross_packing_seed202_model_comparison/"
                "architecture_selection.json"
            ),
            "why": "在另一套颗粒排列上确定最终网络结构。",
        },
        {
            "number": 9,
            "name": "seed303零样本颗粒装填预测",
            "status": stage_state(
                seed303_done and seed303_final_done, architecture_frozen
            ),
            "compute": "OpenFOAM CPU + 单GPU推理",
            "command": "由code/run_hccb_p418_formal_calculations.sh统一调用",
            "output": "results/hccb_p418_cross_packing_seed303_final_*/summary.json",
            "why": "最终网络固定后，在第三套颗粒排列上做独立预测。",
        },
        {
            "number": 10,
            "name": "6组高雷诺数组合的固定流场独立测试",
            "status": stage_state(
                high_re_fixed_complete,
                fixed_done and architecture_frozen and full_model_final_done,
            ),
            "progress": f"{high_re_fixed_count}/6",
            "compute": "OpenFOAM CPU，模型只推理不再训练",
            "command": (
                "MODE=fixed EXECUTE=1 bash "
                "code/run_hccb_p418_high_re_independent_steps.sh"
            ),
            "output": (
                "results/hccb_p418_high_re_independent_fixed_steps_6；"
                "results/hccb_p418_high_re_independent_fixed_model_evaluation/"
                "summary.json"
            ),
            "why": "检查模型在P418原始工况范围高速端的组合迁移能力。",
        },
        {
            "number": 11,
            "name": "6组高雷诺数组合的全耦合独立测试",
            "status": stage_state(
                high_re_full_complete,
                high_re_fixed_done and fully_coupled_done and full_model_final_done,
            ),
            "progress": f"{high_re_fully_coupled_count}/6",
            "compute": "OpenFOAM CPU，模型只推理不再训练",
            "command": (
                "MODE=fully_coupled EXECUTE=1 bash "
                "code/run_hccb_p418_high_re_independent_steps.sh"
            ),
            "output": (
                "results/hccb_p418_high_re_independent_fully_coupled_steps_6；"
                "results/"
                "hccb_p418_high_re_independent_fully_coupled_model_evaluation/"
                "summary.json；results/"
                "hccb_p418_high_re_independent_model_comparison/summary.json"
            ),
            "why": "同时检查高速端和固定流场近似，不让这6条曲线参与模型选择。",
        },
        {
            "number": 12,
            "name": "结果图、论文表格和最终PDF",
            "status": stage_state(
                formal_done,
                seed303_final_done
                and full_model_final_done
                and high_re_fixed_complete
                and high_re_full_complete,
            ),
            "compute": "CPU，约十几分钟",
            "command": "make p418-manuscript-refresh",
            "output": "manuscript/main.pdf",
            "why": "所有数值均从程序输出写入图表，避免手工抄写。",
        },
    ]

    return {
        "title": "P418球床流动--换热与融合模型完整计算顺序",
        "project_root": str(root.resolve()),
        "mode": "只读取文件并打印计划，不启动OpenFOAM或模型训练",
        "workstation_pause_marker_present": workstation_pause_marker,
        "current_progress": {
            "steady": f"{steady_count}/60",
            "fixed_hydrodynamics_steps": f"{fixed_count}/12",
            "fully_coupled_steps": f"{fully_coupled_count}/12",
            "high_re_fixed_steps": f"{high_re_fixed_count}/6",
            "high_re_fully_coupled_steps": f"{high_re_fully_coupled_count}/6",
            "high_re_fixed_frozen_prediction": high_re_fixed_evaluation_done,
            "high_re_fully_coupled_frozen_prediction": (
                high_re_full_evaluation_done
            ),
            "high_re_fixed_vs_fully_coupled_comparison": high_re_comparison_done,
        },
        "stages": stages,
        "important_order": [
            "高雷诺数6组曲线只能在主模型、归一化、训练轮数和损失组合全部固定后读取。",
            "全耦合12条曲线先完成时间步检查，再与固定流场12条曲线逐条比较。",
            "seed303只能使用seed202已经固定的网络结构，不能反过来修改模型。",
        ],
        "new_physical_parameters": [],
        "solver_or_training_started": False,
    }


def print_text(plan: dict[str, object]) -> None:
    print(plan["title"])
    print(f"当前进度：{plan['current_progress']}")
    print("本程序只读取现有文件，不启动求解或训练。")
    if plan["workstation_pause_marker_present"]:
        print("工作站暂停新工况的标记仍在；正式计算等待易算云统一提交。")
    for stage in plan["stages"]:
        progress = f"（{stage['progress']}）" if "progress" in stage else ""
        print(f"{stage['number']:>2}. {stage['name']}{progress}")
        print(f"    状态：{stage['status']}")
        print(f"    计算：{stage['compute']}")
        print(f"    入口：{stage['command']}")
        print(f"    结果：{stage['output']}")
        print(f"    作用：{stage['why']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    plan = build_plan(args.project_root.resolve())
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.format == "json":
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    else:
        print_text(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
