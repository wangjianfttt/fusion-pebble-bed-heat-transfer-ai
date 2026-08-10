#!/usr/bin/env python3
"""Verify that the P418 PINN/operator/diffusion stages share one physical state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SHARED_STATE = [
    "Ux_m_s",
    "Uy_m_s",
    "Uz_m_s",
    "pressure_Pa",
    "temperature_K",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify(
    contract: dict[str, Any],
    graph: dict[str, Any],
    diffusion: dict[str, Any],
    diffusion_state: dict[str, Any],
    fully_coupled: dict[str, Any],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}

    checks["shared_state_order"] = contract["shared_state"]["channel_order"] == SHARED_STATE
    checks["steady_output_order"] = (
        contract["steady_stage"]["state_output_order"] == SHARED_STATE
    )
    checks["transient_output_order"] = (
        contract["physical_transient_stage"]["state_output_order"] == SHARED_STATE
    )
    checks["fully_coupled_output_order"] = (
        fully_coupled["predicted_state_channels"] == SHARED_STATE
    )
    checks["diffusion_corrects_temperature_only"] = (
        contract["diffusion_stage"]["corrected_channels"] == ["temperature_K"]
        and diffusion_state["corrected_state_channels"] == ["temperature"]
    )
    checks["diffusion_preserves_hydrodynamics"] = (
        contract["diffusion_stage"]["unchanged_channels"] == SHARED_STATE[:4]
        and diffusion_state["fixed_state_channels"]
        == ["velocity_x", "velocity_y", "velocity_z", "pressure"]
        and float(diffusion_state["maximum_absolute_fixed_hydrodynamic_change"]) == 0.0
    )
    checks["same_regional_graph"] = (
        int(graph["nodes"]) == int(diffusion["nodes"])
        and int(graph["edges"]) == int(diffusion["edges"])
        and graph["subface_geometry_sha256"] == diffusion["subface_geometry_sha256"]
        and graph["regional_topology_sha256"] == diffusion["regional_topology_sha256"]
    )
    checks["same_time_points"] = (
        int(graph["time_points"]) == int(diffusion["time_points"]) == 56
        and graph["step_plan_sha256"] == diffusion["step_plan_sha256"]
    )
    checks["graph_initial_state_preserved"] = (
        float(graph["initial_maximum_absolute_error"]) == 0.0
    )
    checks["graph_hydrodynamics_preserved"] = (
        float(graph["hydrodynamic_maximum_absolute_error"]) == 0.0
    )
    checks["graph_physics_active"] = graph["physics_mode"] == "energy_and_flux"
    checks["graph_forward_backward_finite"] = (
        bool(graph["loss_finite"])
        and bool(graph["all_gradients_present"])
        and bool(graph["all_gradients_finite"])
    )
    checks["diffusion_forward_backward_finite"] = (
        bool(diffusion["loss_finite"])
        and bool(diffusion["all_gradients_present"])
        and bool(diffusion["all_gradients_finite"])
    )
    checks["diffusion_temperature_energy_rule"] = bool(
        diffusion_state["all_checks_passed"]
    )
    checks["fully_coupled_face_flux_predicted"] = fully_coupled[
        "predicted_face_quantities"
    ] == [
        "oriented_internal_mass_flux_kg_s",
        "oriented_boundary_mass_flux_kg_s",
    ]
    checks["no_formal_training_hidden_in_checks"] = (
        not bool(fully_coupled["formal_training_started"])
        and not bool(preflight["full_training_can_start"])
    )
    checks["no_new_physical_parameters"] = all(
        not item.get("new_physical_parameters", [])
        for item in (contract, graph, diffusion, diffusion_state, fully_coupled)
    )

    failed = [name for name, passed in checks.items() if not passed]
    require(not failed, "end-to-end model interface checks failed: " + ", ".join(failed))

    steady = preflight["current_data"]["steady"]
    fixed = preflight["current_data"]["physical_transient"]
    coupled = preflight["current_data"]["fully_coupled_transient"]
    return {
        "status": "p418_end_to_end_model_interface_verified",
        "checks": checks,
        "shared_state_order": SHARED_STATE,
        "actual_regional_graph": {
            "nodes": int(graph["nodes"]),
            "edges": int(graph["edges"]),
            "time_points": int(graph["time_points"]),
            "subface_geometry_sha256": graph["subface_geometry_sha256"],
            "regional_topology_sha256": graph["regional_topology_sha256"],
            "step_plan_sha256": graph["step_plan_sha256"],
        },
        "graph_transformer": {
            "model_parameter_count": int(graph["model_parameter_count"]),
            "peak_gpu_GB": float(graph["peak_gpu_GB"]),
            "elapsed_seconds": float(graph["elapsed_seconds"]),
            "scope": "full-size forward/loss/backward program and memory check",
        },
        "diffusion_temperature_refiner": {
            "model_parameter_count": int(diffusion["model_parameter_count"]),
            "peak_gpu_GB": float(diffusion["peak_gpu_GB"]),
            "elapsed_seconds": float(diffusion["elapsed_seconds"]),
            "corrected_channel": "temperature_K",
            "scope": "full-size forward/loss/backward program and memory check",
        },
        "current_openfoam_data": {
            "steady": f"{steady['completed']}/{steady['required']}",
            "fixed_hydrodynamics_steps": f"{fixed['completed']}/{fixed['required']}",
            "fully_coupled_steps": f"{coupled['completed']}/{coupled['required']}",
        },
        "formal_accuracy_available": False,
        "interpretation": (
            "The three model stages use one state order, one regional graph and one "
            "56-time design. The diffusion stage changes temperature only. This proves "
            "the computational interface, not prediction accuracy; accuracy still "
            "requires the complete OpenFOAM histories."
        ),
        "new_physical_parameters": [],
    }


def chinese_summary(result: dict[str, Any]) -> str:
    graph = result["actual_regional_graph"]
    data = result["current_openfoam_data"]
    return (
        "# P418融合模型接口检查\n\n"
        "这项检查回答一个具体问题：PINN、图--Transformer和扩散修正是否在"
        "同一套物理量、同一张三维区域网格和同一批输出时刻上工作。\n\n"
        "## 已确认\n\n"
        f"- 共同状态顺序：`{', '.join(result['shared_state_order'])}`。\n"
        f"- 实际区域图：`{graph['nodes']}`个节点、`{graph['edges']}`条边。\n"
        f"- 图--Transformer与扩散模型都使用`{graph['time_points']}`个输出时刻，"
        "网格和时间计划的校验值相同。\n"
        "- 图--Transformer在固定流场热响应中只推进温度，初始温度严格等于"
        "源工况，速度和压力没有改变。\n"
        "- 扩散模型只修正温度残差，不改变速度、压力和面质量流；温度误差降低"
        "但能量关系变差时，不记为共同改善。\n"
        "- 全耦合扩展同时预测速度、压力、温度和面质量流，用来检验固定流场"
        "近似造成的偏差。\n"
        "- 本次没有增加任何球床物性或运行参数，也没有启动正式训练。\n\n"
        "## 当前数据\n\n"
        f"- 三维稳态：`{data['steady']}`。\n"
        f"- 固定流场热阶跃：`{data['fixed_hydrodynamics_steps']}`。\n"
        f"- 全耦合流动--换热阶跃：`{data['fully_coupled_steps']}`。\n\n"
        "因此，模型之间的程序连接已经在真实46,089节点网格上跑通；"
        "完整预测精度仍需等待OpenFOAM数据补齐后计算。\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "parameters/hccb_p418_fused_model_contract.json",
    )
    parser.add_argument(
        "--graph-summary",
        type=Path,
        default=ROOT
        / "results/hccb_p418_actual_spatiotemporal_operator_56time_gpu_factorized/summary.json",
    )
    parser.add_argument(
        "--diffusion-summary",
        type=Path,
        default=ROOT
        / "results/hccb_p418_actual_temporal_diffusion_56time_gpu_batch1_bfloat16_chunk2048/summary.json",
    )
    parser.add_argument(
        "--diffusion-state-summary",
        type=Path,
        default=ROOT / "results/hccb_p418_diffusion_physical_state/summary.json",
    )
    parser.add_argument(
        "--fully-coupled-summary",
        type=Path,
        default=ROOT / "results/hccb_p418_fully_coupled_training_interface/summary.json",
    )
    parser.add_argument(
        "--preflight-summary",
        type=Path,
        default=ROOT / "results/hccb_p418_fused_preflight/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/hccb_p418_end_to_end_model_interface",
    )
    args = parser.parse_args()

    result = verify(
        load_json(args.contract.resolve()),
        load_json(args.graph_summary.resolve()),
        load_json(args.diffusion_summary.resolve()),
        load_json(args.diffusion_state_summary.resolve()),
        load_json(args.fully_coupled_summary.resolve()),
        load_json(args.preflight_summary.resolve()),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "P418_融合模型接口检查_CN.md").write_text(
        chinese_summary(result), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
