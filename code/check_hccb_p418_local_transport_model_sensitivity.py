#!/usr/bin/env python3
"""Check that local P418 flow variables reach the intended model equations.

This is a deterministic software/physics-path check.  It does not train a
model and it does not report predictive accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from hccb_multiregion_steady_cht_residual import CoupledInterfaceMap, RegionMesh
from hccb_p418_fully_coupled_spatiotemporal_operator import (
    HCCBP418FullyCoupledRegionalOperator,
    P418FullyCoupledFluxGraph,
)
from hccb_p418_regional_cht_adapter import P418SubfaceGeometry
from hccb_p418_spatiotemporal_regional_operator import (
    HCCBP418SpatiotemporalRegionalOperator,
    P418ThermalStepRegionalGraph,
)
from hccb_p418_transient_regional_physics import (
    assemble_p418_transient_regional_residual,
)


DEFAULT_CONTRACT = ROOT / "parameters/hccb_p418_local_transport_model_contract.json"
DEFAULT_OUTPUT = ROOT / "results/hccb_p418_local_transport_model_sensitivity"
SEED = 20260725


def regional_graph(boundary_fraction: torch.Tensor | None = None):
    if boundary_fraction is None:
        boundary_fraction = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
    return P418ThermalStepRegionalGraph.from_tensors(
        centroid_m=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        volume_m3=torch.ones(4, dtype=torch.float32),
        node_type=torch.tensor([0, 0, 1, 1]),
        edge_source=torch.tensor([0, 1, 2, 3, 0, 2]),
        edge_target=torch.tensor([1, 0, 3, 2, 2, 0]),
        edge_kind=torch.tensor([0, 0, 1, 1, 2, 2]),
        edge_area_m2=torch.ones(6, dtype=torch.float32),
        edge_area_vector_m2=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        boundary_fraction=boundary_fraction,
    )


def subface_geometry() -> P418SubfaceGeometry:
    dtype = torch.float64
    fluid_mesh = RegionMesh(
        cell_centroid=torch.tensor(
            [[0.0, 0.0, 0.5], [0.0, 0.0, 1.5]], dtype=dtype
        ),
        cell_volume=torch.ones(2, dtype=dtype),
        internal_face_centroid=torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype),
        internal_area_vector=torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype),
        internal_owner=torch.tensor([0]),
        internal_neighbour=torch.tensor([1]),
        boundary_face_centroid=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0],
                [-0.5, 0.0, 0.5],
                [0.5, 0.0, 1.5],
                [0.0, 0.5, 0.5],
            ],
            dtype=dtype,
        ),
        boundary_area_vector=torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=dtype,
        ),
        boundary_owner=torch.tensor([0, 1, 0, 1, 0]),
    )
    solid_mesh = RegionMesh(
        cell_centroid=torch.tensor([[0.0, 1.0, 0.5]], dtype=dtype),
        cell_volume=torch.ones(1, dtype=dtype),
        internal_face_centroid=torch.empty((0, 3), dtype=dtype),
        internal_area_vector=torch.empty((0, 3), dtype=dtype),
        internal_owner=torch.empty(0, dtype=torch.long),
        internal_neighbour=torch.empty(0, dtype=torch.long),
        boundary_face_centroid=torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.5, 1.0, 0.5],
                [-0.5, 1.0, 0.5],
                [0.0, 0.5, 0.5],
            ],
            dtype=dtype,
        ),
        boundary_area_vector=torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=dtype,
        ),
        boundary_owner=torch.zeros(5, dtype=torch.long),
    )
    return P418SubfaceGeometry(
        fluid_mesh=fluid_mesh,
        solid_mesh=solid_mesh,
        interface=CoupledInterfaceMap(
            fluid_boundary_face=torch.tensor([4]),
            solid_boundary_face=torch.tensor([4]),
        ),
        fluid_boundary_patch=torch.arange(5),
        solid_boundary_patch=torch.arange(5),
        fluid_patch_names=(
            "inlet",
            "outlet",
            "coolingWall",
            "symmetryWalls",
            "fluid_to_solid",
        ),
        solid_patch_names=(
            "inlet",
            "outlet",
            "coolingWall",
            "symmetryWalls",
            "solid_to_fluid",
        ),
        fine_to_regional_global=np.array([0, 1, 2]),
        fluid_global_region=np.array([0, 1]),
        solid_global_region=np.array([2]),
    )


def fixed_operator_checks() -> dict[str, float | bool]:
    torch.manual_seed(SEED)
    graph = regional_graph()
    model = HCCBP418SpatiotemporalRegionalOperator(
        hidden_dim=8,
        local_pre_iterations=1,
        physics_attention_blocks=1,
        local_post_iterations=1,
        physics_attention_heads=2,
        physics_slices=4,
        temporal_layers=1,
        temporal_heads=1,
        temporal_node_chunk_size=2,
        spatial_temporal_mode="factorized_static_spatial",
        boundary_role_count=graph.boundary_role_count,
    ).eval()
    initial = torch.tensor(
        [
            [
                [0.20, 0.01, 0.00, 1.20, 0.30],
                [0.28, 0.02, 0.01, 1.10, 0.34],
                [0.00, 0.00, 0.00, 0.00, 0.55],
                [0.00, 0.00, 0.00, 0.00, 0.60],
            ]
        ],
        dtype=torch.float32,
    )
    condition = torch.zeros((1, 8), dtype=torch.float32)
    time = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    with torch.no_grad():
        baseline = model(initial, condition, time, graph)
        velocity = initial.clone()
        velocity[0, 0, 0] += 0.25
        velocity_result = model(velocity, condition, time, graph)
        pressure = initial.clone()
        pressure[0, 1, 3] += 0.40
        pressure_result = model(pressure, condition, time, graph)
        changed_boundary = graph.boundary_fraction.clone()
        changed_boundary[[0, 1]] = changed_boundary[[1, 0]]
        boundary_result = model(
            initial, condition, time, regional_graph(changed_boundary)
        )
    initial_exact = float((baseline[:, 0] - initial).abs().max())
    fixed_hydrodynamics = float(
        (
            baseline[..., :4]
            - initial[:, None, :, :4].expand(-1, len(time), -1, -1)
        )
        .abs()
        .max()
    )
    return {
        "initial_state_exact_max_abs": initial_exact,
        "fixed_hydrodynamics_max_abs": fixed_hydrodynamics,
        "velocity_to_final_temperature_max_abs": float(
            (velocity_result[:, -1, :, 4] - baseline[:, -1, :, 4]).abs().max()
        ),
        "pressure_to_final_temperature_max_abs": float(
            (pressure_result[:, -1, :, 4] - baseline[:, -1, :, 4]).abs().max()
        ),
        "boundary_role_to_final_temperature_max_abs": float(
            (boundary_result[:, -1, :, 4] - baseline[:, -1, :, 4]).abs().max()
        ),
    }


def fixed_flux_physics_checks() -> dict[str, float]:
    geometry = subface_geometry()
    time = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
    state = torch.zeros((1, 3, 3, 5), dtype=torch.float64)
    state[:, :, 0, 2] = 0.20
    state[:, :, 1, 2] = 0.18
    state[:, :, :2, 3] = 120000.0
    state[:, :, 0, 4] = torch.tensor([700.0, 705.0, 715.0])
    state[:, :, 1, 4] = torch.tensor([760.0, 762.0, 768.0])
    state[:, :, 2, 4] = torch.tensor([680.0, 686.0, 697.0])
    step = torch.tensor(
        [[0.20, 700.0, 4.85, 0.25, 900.0, 8.85, 120000.0, 635.0]],
        dtype=torch.float64,
    )
    internal = torch.tensor([[0.020]], dtype=torch.float64)
    boundary = torch.tensor(
        [[-0.020, 0.020, 0.0, 0.0, 0.0]], dtype=torch.float64
    )
    reference = assemble_p418_transient_regional_residual(
        geometry=geometry,
        step_condition=step,
        state_physical=state,
        time_s=time,
        fluid_internal_mass_flux_kg_s=internal,
        fluid_boundary_mass_flux_kg_s=boundary,
    )
    changed = assemble_p418_transient_regional_residual(
        geometry=geometry,
        step_condition=step,
        state_physical=state,
        time_s=time,
        fluid_internal_mass_flux_kg_s=1.2 * internal,
        fluid_boundary_mass_flux_kg_s=1.2 * boundary,
    )
    return {
        "mass_flux_to_fluid_energy_residual_max_abs_W_m3": float(
            (changed.fluid_energy_w_m3 - reference.fluid_energy_w_m3).abs().max()
        ),
        "mass_flux_to_mass_residual_max_abs_kg_m3_s": float(
            (changed.fluid_mass_kg_m3_s - reference.fluid_mass_kg_m3_s)
            .abs()
            .max()
        ),
        "mass_flux_to_internal_energy_flux_max_abs_W": float(
            (
                changed.fluid_internal_energy_flux_w
                - reference.fluid_internal_energy_flux_w
            )
            .abs()
            .max()
        ),
    }


def fully_coupled_checks() -> dict[str, float]:
    torch.manual_seed(SEED + 1)
    graph = regional_graph()
    flux_graph = P418FullyCoupledFluxGraph.from_tensors(
        internal_owner_global=torch.tensor([0]),
        internal_neighbour_global=torch.tensor([1]),
        internal_features=torch.zeros((1, 10), dtype=torch.float32),
        boundary_owner_global=torch.tensor([0, 1]),
        boundary_features=torch.zeros((2, 12), dtype=torch.float32),
        boundary_active=torch.tensor([True, True]),
        node_count=graph.node_count,
    )
    model = HCCBP418FullyCoupledRegionalOperator(
        hidden_dim=8,
        local_pre_iterations=1,
        physics_attention_blocks=1,
        local_post_iterations=1,
        physics_attention_heads=1,
        physics_slices=2,
        temporal_layers=1,
        temporal_heads=1,
        temporal_node_chunk_size=None,
        boundary_role_count=graph.boundary_role_count,
    ).eval()
    initial = torch.tensor(
        [
            [
                [0.20, 0.01, 0.00, 1.20, 0.30],
                [0.28, 0.02, 0.01, 1.10, 0.34],
                [0.00, 0.00, 0.00, 0.00, 0.55],
                [0.00, 0.00, 0.00, 0.00, 0.60],
            ]
        ],
        dtype=torch.float32,
    )
    internal = torch.tensor([[0.020]], dtype=torch.float32)
    boundary = torch.tensor([[-0.020, 0.020]], dtype=torch.float32)
    time = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    with torch.no_grad():
        result = model(
            initial,
            internal,
            boundary,
            torch.zeros((1, 8), dtype=torch.float32),
            time,
            graph,
            flux_graph,
        )
        changed_flux = model(
            initial,
            1.2 * internal,
            1.2 * boundary,
            torch.zeros((1, 8), dtype=torch.float32),
            time,
            graph,
            flux_graph,
        )
    return {
        "initial_state_exact_max_abs": float(
            (result.state[:, 0] - initial).abs().max()
        ),
        "initial_internal_flux_exact_max_abs": float(
            (result.internal_mass_flux[:, 0] - internal).abs().max()
        ),
        "initial_boundary_flux_exact_max_abs": float(
            (result.boundary_mass_flux[:, 0] - boundary).abs().max()
        ),
        "predicted_state_time_change_max_abs": float(
            (result.state[:, -1] - result.state[:, 0]).abs().max()
        ),
        "predicted_internal_flux_time_change_max_abs": float(
            (result.internal_mass_flux[:, -1] - result.internal_mass_flux[:, 0])
            .abs()
            .max()
        ),
        "initial_flux_to_predicted_flux_max_abs": float(
            (changed_flux.internal_mass_flux - result.internal_mass_flux)
            .abs()
            .max()
        ),
        "initial_flux_to_state_direct_max_abs": float(
            (changed_flux.state - result.state).abs().max()
        ),
    }


def render_cn(payload: dict[str, object]) -> str:
    fixed = payload["fixed_flow_operator"]
    flux = payload["fixed_flow_physics"]
    coupled = payload["fully_coupled_operator"]
    return "\n".join(
        [
            "# P418局部流场输入敏感性检查",
            "",
            "这一步不训练模型，也不评价预测精度。它只回答一个具体问题："
            "局部速度、压力、边界位置和面质量流是否真的进入当前PINN/Transformer计算链条。",
            "",
            "## 固定流场温度模型",
            "",
            f"- 初始状态在 `t=0` 的最大差值为 `{fixed['initial_state_exact_max_abs']:.3e}`；",
            f"- 全时间过程的速度和压力最大改动为 `{fixed['fixed_hydrodynamics_max_abs']:.3e}`，"
            "符合固定流场温度阶跃的定义；",
            f"- 改动局部速度后，末时刻温度输出最大变化为 "
            f"`{fixed['velocity_to_final_temperature_max_abs']:.3e}`；",
            f"- 改动局部压力后，末时刻温度输出最大变化为 "
            f"`{fixed['pressure_to_final_temperature_max_abs']:.3e}`；",
            f"- 交换入口/出口边界邻接比例后，末时刻温度输出最大变化为 "
            f"`{fixed['boundary_role_to_final_temperature_max_abs']:.3e}`。",
            "",
            "因此，局部三维速度、压力和边界位置不是只写在说明文件中，"
            "而是进入了图–Transformer的状态或几何编码。",
            "",
            "## 固定面质量流的物理作用",
            "",
            f"- 将面质量流整体提高20%后，流体能量方程最大变化为 "
            f"`{flux['mass_flux_to_fluid_energy_residual_max_abs_W_m3']:.3e} W/m3`；",
            f"- 质量方程最大变化为 "
            f"`{flux['mass_flux_to_mass_residual_max_abs_kg_m3_s']:.3e} kg/(m3 s)`；",
            f"- 内部面的能量流最大变化为 "
            f"`{flux['mass_flux_to_internal_energy_flux_max_abs_W']:.3e} W`。",
            "",
            "固定流场模型没有把每个面质量流再复制成一个网络特征。它把OpenFOAM给出的"
            "面质量流作为流体焓输运和质量守恒方程中的已知系数。这样既保留了局部流动作用，"
            "也符合“流场已经求好，只预测温度过程”的工况定义。",
            "",
            "## 全耦合模型",
            "",
            f"- `t=0` 的状态、内部面质量流和边界面质量流最大差值分别为 "
            f"`{coupled['initial_state_exact_max_abs']:.3e}`、"
            f"`{coupled['initial_internal_flux_exact_max_abs']:.3e}` 和 "
            f"`{coupled['initial_boundary_flux_exact_max_abs']:.3e}`；",
            f"- 随时间预测的状态和内部面质量流最大变化分别为 "
            f"`{coupled['predicted_state_time_change_max_abs']:.3e}` 和 "
            f"`{coupled['predicted_internal_flux_time_change_max_abs']:.3e}`；",
            f"- 改动初始面质量流后，预测面质量流最大变化为 "
            f"`{coupled['initial_flux_to_predicted_flux_max_abs']:.3e}`。",
            "",
            "全耦合模型中，初始面质量流既进入面流量分支，也按面方向传给相邻区域节点；"
            "区域状态与面流量随后还会由连续性、动量和能量方程共同联系。"
            f"本次未训练随机网络中，单独改面质量流对状态分支的直接变化为 "
            f"`{coupled['initial_flux_to_state_direct_max_abs']:.3e}`。该数值只证明输入路径存在，"
            "不是正式模型精度结论。",
            "",
            "## 结论",
            "",
            "当前接口没有发现“局部速度、压力或面质量流只写在文档里、实际没有进入计算”的问题。"
            "固定流场和全耦合模型采用了不同但与各自物理任务一致的输入方式。这项检查只证明输入"
            "路径和方程作用存在，不决定最终模型排名，也不要求据此新增全耦合求解；正式误差和效率"
            "比较以已完成的数据集和最终模型比较结果为准。",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    contract = json.loads(args.contract.resolve().read_text(encoding="utf-8"))
    fixed = fixed_operator_checks()
    flux = fixed_flux_physics_checks()
    coupled = fully_coupled_checks()
    checks = {
        "contract_introduces_no_physical_parameters": (
            contract.get("new_physical_parameters") == []
        ),
        "fixed_flow_initial_state_is_exact": (
            fixed["initial_state_exact_max_abs"] <= 1.0e-7
        ),
        "fixed_flow_hydrodynamics_remain_fixed": (
            fixed["fixed_hydrodynamics_max_abs"] <= 1.0e-7
        ),
        "local_velocity_reaches_temperature_branch": (
            fixed["velocity_to_final_temperature_max_abs"] > 1.0e-8
        ),
        "local_pressure_reaches_temperature_branch": (
            fixed["pressure_to_final_temperature_max_abs"] > 1.0e-8
        ),
        "boundary_role_reaches_temperature_branch": (
            fixed["boundary_role_to_final_temperature_max_abs"] > 1.0e-8
        ),
        "fixed_face_flux_reaches_energy_equation": (
            flux["mass_flux_to_fluid_energy_residual_max_abs_W_m3"] > 0.0
            and flux["mass_flux_to_internal_energy_flux_max_abs_W"] > 0.0
        ),
        "fully_coupled_initial_state_and_flux_are_exact": (
            max(
                coupled["initial_state_exact_max_abs"],
                coupled["initial_internal_flux_exact_max_abs"],
                coupled["initial_boundary_flux_exact_max_abs"],
            )
            <= 1.0e-7
        ),
        "fully_coupled_predicts_time_dependent_state_and_flux": (
            coupled["predicted_state_time_change_max_abs"] > 1.0e-8
            and coupled["predicted_internal_flux_time_change_max_abs"] > 1.0e-8
        ),
        "initial_face_flux_reaches_fully_coupled_flux_branch": (
            coupled["initial_flux_to_predicted_flux_max_abs"] > 1.0e-8
        ),
        "initial_face_flux_reaches_fully_coupled_state_branch": (
            coupled["initial_flux_to_state_direct_max_abs"] > 1.0e-8
        ),
    }
    payload: dict[str, object] = {
        "status": (
            "p418_local_transport_input_paths_confirmed"
            if all(checks.values())
            else "p418_local_transport_input_path_check_failed"
        ),
        "purpose": (
            "deterministic input-path and equation-sensitivity check; "
            "not model training or predictive validation"
        ),
        "seed": SEED,
        "new_physical_parameters": [],
        "physical_source_ids": contract["physical_source_ids"],
        "fixed_flow_operator": fixed,
        "fixed_flow_physics": flux,
        "fully_coupled_operator": coupled,
        "checks": checks,
        "interpretation": {
            "fixed_flow": (
                "regional U and p are direct state-encoder inputs; frozen face mass "
                "flux is a known coefficient in conservative mass/enthalpy transport"
            ),
            "fully_coupled": (
                "regional U,p,T and oriented initial face mass flux are encoded "
                "into adjacent nodes; the model predicts time-dependent U,p,T and "
                "face mass flux, coupled by continuity, momentum and energy losses"
            ),
        },
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "P418_局部流场输入敏感性_CN.md").write_text(
        render_cn(payload), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
