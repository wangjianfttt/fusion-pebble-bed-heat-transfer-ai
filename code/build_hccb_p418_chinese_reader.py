#!/usr/bin/env python3
"""Generate a concise Chinese reader from the accepted P418 results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


FINAL_STATUS = "complete_p418_final_manuscript_narrative"
MESH_STATUS = "completed_three_mesh_p418_cht_comparison"
STEADY_LABELS = {
    "response_surface": "响应面",
    "pinn_data_only": "纯数据 PINN",
    "pinn": "物理约束 PINN",
    "graph": "图算子",
    "transolver": "Physics-Attention 算子",
}
TRANSIENT_LABELS = {
    "initial_temperature_persistence": "初始温度场持续性基线",
    "dmdc": "连续时间 DMDc",
    "graph_transformer_data_only": "纯数据图 Transformer",
    "graph_transformer_energy_flux": "能量与热流约束图 Transformer",
    "graph_transformer_factorized_energy_flux": (
        "分解式能量与热流约束图 Transformer"
    ),
    "low_rank_residual_correction": "POD 低秩残差修正",
    "diffusion_residual_correction": "扩散残差修正",
}
HIGH_RE_LABELS = {
    "data_only": "纯数据图 Transformer",
    "physics_constrained": "物理约束图 Transformer",
    "factorized": "分解式物理约束图 Transformer",
}


def load_json(path: Path, expected_status: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 不是 JSON 对象")
    if payload.get("status") != expected_status:
        raise ValueError(
            f"{path} 状态不正确：{payload.get('status')}"
        )
    return payload


def finite(value: object, name: str, *, nonnegative: bool = True) -> float:
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{name} 不是有效数值：{value}")
    return result


def fmt(value: float) -> str:
    absolute = abs(value)
    if value == 0.0:
        return "0"
    if absolute < 0.01:
        return f"{value:.2e}"
    if absolute < 1.0:
        return f"{value:.3f}"
    if absolute < 10.0:
        return f"{value:.2f}"
    return f"{value:.1f}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mesh_lookup(mesh: dict) -> tuple[dict[str, dict], dict[str, dict]]:
    levels = mesh.get("mesh_levels")
    convergence = mesh.get("grid_convergence")
    if not isinstance(levels, list) or len(levels) != 3:
        raise ValueError("三网格结果必须恰好包含粗、中、细三档")
    if not isinstance(convergence, list):
        raise ValueError("三网格 GCI 结果缺失")
    level_lookup = {str(row.get("mesh_level")): row for row in levels}
    if set(level_lookup) != {"coarse", "medium", "fine"}:
        raise ValueError("粗、中、细网格不完整")
    for name, row in level_lookup.items():
        if (
            row.get("fluid_basic_check_passes") is not True
            or row.get("solid_basic_check_passes") is not True
        ):
            raise ValueError(f"{name} 网格未通过流体区和固体区基础检查")
    convergence_lookup = {
        str(row.get("metric")): row for row in convergence
    }
    required = {
        "pressure_drop_Pa",
        "outlet_temperature_change_K",
        "solid_maximum_temperature_change_K",
        "cooling_wall_heat_fraction",
    }
    if not required.issubset(convergence_lookup):
        raise ValueError("三网格工程量不完整")
    return level_lookup, convergence_lookup


def validate_final(final: dict) -> None:
    expected_counts = {
        "steady_case_count": 60,
        "fixed_flow_trajectory_count": 12,
        "high_re_curve_count": 6,
        "independent_packing_case_count": 9,
    }
    for key, expected in expected_counts.items():
        if int(final.get(key, -1)) != expected:
            raise ValueError(f"{key} 必须为 {expected}")
    if final.get("full_domain_solver_started") is not False:
        raise ValueError("全文域求解范围说明不一致")
    if final.get("fully_coupled_accuracy_claimed") is not False:
        raise ValueError("全耦合精度说明不一致")
    if final.get("new_physical_parameters") != []:
        raise ValueError("中文便读版不得引入新的物理参数")


def build_text(final: dict, mesh: dict) -> str:
    validate_final(final)
    levels, gci = mesh_lookup(mesh)
    if mesh.get("new_physical_parameters") != []:
        raise ValueError("三网格计算引入了新的物理参数")

    steady_model = str(final["steady_solid_temperature_leader"])
    transient_model = str(final["best_strict_transient_model"])
    high_re_model = str(final["best_high_re_model"])
    if steady_model not in STEADY_LABELS:
        raise ValueError(f"未知稳态模型：{steady_model}")
    if transient_model not in TRANSIENT_LABELS:
        raise ValueError(f"未知瞬态模型：{transient_model}")
    if high_re_model not in HIGH_RE_LABELS:
        raise ValueError(f"未知高流速模型：{high_re_model}")

    solid_range = final["solid_maximum_temperature_range_K"]
    reversal = final["wall_heat_reversal_temperature_range_K"]
    wall_counts = final["wall_heat_direction_case_counts"]
    robustness = final["robustness_and_learning_curve"]
    packing = final[
        "independent_packing_maximum_absolute_change_percent"
    ]
    external = final.get("external_consistency")
    if not isinstance(external, dict):
        raise ValueError("外部传热与压降对照结果缺失")
    if (
        external.get("used_in_p418_training") is not False
        or external.get("cellwise_validation_claimed") is not False
    ):
        raise ValueError("外部对照的用途说明不正确")
    external_hcpb_error = finite(
        external["hcpb_annulus_mean_absolute_relative_error_percent"],
        "HELOKA/HCPB Nusselt 平均相对误差",
    )
    external_pressure_error = finite(
        external["fixed_bed_pressure_median_absolute_relative_error_percent"],
        "1 mm 固定床压降中位相对误差",
    )
    transient_rmse = final["strict_transient_temperature_RMSE_K"]
    transient_energy = final[
        "strict_transient_projection_aware_energy_normalized_RMSE"
    ]
    diffusion_rmse = final["diffusion_temperature_RMSE_K"]
    diffusion_energy = final[
        "diffusion_projection_aware_energy_normalized_RMSE"
    ]
    diffusion_coverage = finite(
        final["diffusion_90pct_interval_coverage_fraction"],
        "扩散90%区间覆盖率",
    )
    diffusion_width = finite(
        final["diffusion_90pct_interval_mean_width_K"],
        "扩散90%区间平均宽度",
    )
    if not 0.0 <= diffusion_coverage <= 1.0 or diffusion_width < 0.0:
        raise ValueError("扩散不确定性结果不合理")
    joint = final["diffusion_joint_temperature_energy_improvement"]

    for name, values in (
        ("固体最高温度范围", solid_range),
        ("壁面换热反转温度范围", reversal),
        ("三条曲线训练误差范围", robustness[
            "transient_three_curve_RMSE_range_K"
        ]),
    ):
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError(f"{name} 不完整")

    mesh_cells = "、".join(
        (
            f"{name}网格 {int(levels[key]['fluid_cells']) + int(levels[key]['solid_cells']):,}"
            " 个单元"
        )
        for key, name in (
            ("coarse", "粗"),
            ("medium", "中"),
            ("fine", "细"),
        )
    )
    pressure_gci = 100.0 * finite(
        gci["pressure_drop_Pa"]["fine_gci_fraction"], "压降 GCI"
    )
    outlet_gci = 100.0 * finite(
        gci["outlet_temperature_change_K"]["fine_gci_fraction"],
        "出口温度变化 GCI",
    )
    solid_gci_abs = finite(
        gci["solid_maximum_temperature_change_K"]["fine_gci_absolute"],
        "固体最高温度变化 GCI",
    )
    wall_gci = 100.0 * finite(
        gci["cooling_wall_heat_fraction"]["fine_gci_fraction"],
        "壁面热流分数 GCI",
    )
    transient_triplet = "、".join(
        fmt(finite(transient_rmse[key], f"{key} 温度误差"))
        for key in (
            "graph_transformer_data_only",
            "graph_transformer_energy_flux",
            "graph_transformer_factorized_energy_flux",
        )
    )
    energy_triplet = "、".join(
        fmt(finite(transient_energy[key], f"{key} 能量差异"))
        for key in (
            "graph_transformer_data_only",
            "graph_transformer_energy_flux",
            "graph_transformer_factorized_energy_flux",
        )
    )
    diffusion_statement = (
        "扩散修正同时降低了温度误差和能量方程差异。"
        if joint
        else "扩散修正没有同时改善温度误差和能量方程差异，因此只作为误差权衡结果。"
    )
    diffusion_uncertainty_statement = (
        f"名义90%区间的实际覆盖率为{fmt(100.0 * diffusion_coverage)}%"
        f"，平均宽度为{fmt(diffusion_width)} K。"
        + (
            "该区间明显过窄，不能解读为已充分定标的不确定性。"
            if diffusion_coverage < 0.85
            else ""
        )
    )
    if transient_model == "initial_temperature_persistence":
        transient_cost_statement = (
            "在持续性基线、DMDc、三种图 Transformer、POD 低秩修正和"
            "扩散修正的统一比较中，温度误差最低的是"
            f"{TRANSIENT_LABELS[transient_model]}。它不需要训练，因此不报告"
            "训练成本回收曲线数。"
        )
    else:
        transient_cost_statement = (
            "在持续性基线、DMDc、三种图 Transformer、POD 低秩修正和"
            "扩散修正的统一比较中，温度误差最低的是"
            f"{TRANSIENT_LABELS[transient_model]}；相对 32 核 OpenFOAM 参考计算，"
            "其完整预测链加速 "
            f"{fmt(finite(final['best_strict_transient_speedup'], '瞬态加速比'))} 倍，"
            f"约预测 {int(final['best_strict_transient_break_even_curves'])} 条曲线后"
            "可抵消训练和参考数据成本。"
        )

    return f"""# P418 论文中文便读版

## 1. 研究问题

本文研究聚变堆固体增殖剂球床内的孔隙尺度共轭传热。氦气在球形陶瓷颗粒间流动，颗粒内部产生体积热源，同时冷却壁与流体、颗粒共同换热。研究重点不是让神经网络替代物理求解，而是判断在完整三维参考数据、严格外推划分和守恒约束下，PINN、图 Transformer 与扩散修正能否快速预测稳态和热阶跃后的温度场、壁面热流及热点位置。

## 2. 参考数据与计算范围

基准数据库包含 60 个三维稳态 OpenFOAM 共轭传热工况、12 条 0--300 s 固定流场热阶跃曲线、6 条独立高流速曲线，以及第二套球形颗粒装填中的 9 个匹配工况。模型训练和测试按完整工况及成对端点分开，测试工况没有拆散后混入训练集。三网格研究采用{mesh_cells}；三档流体区和固体区均通过基础网格检查。细网格 GCI 为：压降 {fmt(pressure_gci)}%，出口温度变化 {fmt(outlet_gci)}%，固体最高温度变化绝对值 {fmt(solid_gci_abs)} K，壁面热流分数 {fmt(wall_gci)}%。这些结果说明温度量已经较稳定，但压降和局部壁面热流仍对网格较敏感，因此文中分别报告热学误差与水力误差。

## 3. 球床传热规律

60 个稳态工况中的固体最高温度为 {fmt(finite(solid_range[0], '最低固体温度'))}--{fmt(finite(solid_range[1], '最高固体温度'))} K。壁面换热方向并不固定：{int(wall_counts['wall_to_fluid'])} 个工况由壁面向流体供热，{int(wall_counts['fluid_to_wall'])} 个工况由流体向壁面放热；方向反转出现在入口温度 {fmt(finite(reversal[0], '最低反转温度'))}--{fmt(finite(reversal[1], '最高反转温度'))} K。稳态相邻工况间参考热点的最大位移为 {fmt(1000.0 * finite(final['maximum_adjacent_reference_hotspot_distance_m'], '最大热点位移'))} mm。这说明平均温度误差较小并不等于壁面换热方向和热点位置都正确。

外部对照中，HELOKA/HCPB Nusselt 数的平均绝对相对误差为 {fmt(external_hcpb_error)}%，1 mm 固定床压力梯度的中位绝对相对误差为 {fmt(external_pressure_error)}%。PREMUX 和 TESOMEX 只用于检查整体温度变化是否合理，不作为当前三维网格逐单元精度证明，也没有用于训练模型。

## 4. 模型结果

五种稳态模型中，固体温度最坏工况归一化均方根误差最低的是{STEADY_LABELS[steady_model]}，误差为 {fmt(finite(final['steady_solid_temperature_worst_case_nrmse'], '稳态误差'))}。在端点成对独立的瞬态测试中，纯数据、能量与热流约束、分解式能量与热流约束三种图 Transformer 的固体温度 RMSE 依次为 {transient_triplet} K，对应的能量方程差异依次为 {energy_triplet}。{transient_cost_statement}

扩散修正使确定性温度 RMSE 从 {fmt(finite(diffusion_rmse['deterministic'], '确定性温度误差'))} K 变为 {fmt(finite(diffusion_rmse['refined'], '扩散温度误差'))} K，能量方程差异从 {fmt(finite(diffusion_energy['deterministic'], '确定性能量差异'))} 变为 {fmt(finite(diffusion_energy['refined'], '扩散能量差异'))}。{diffusion_statement}{diffusion_uncertainty_statement}在 6 条独立高流速曲线上，表现最好的冻结模型为{HIGH_RE_LABELS[high_re_model]}，流体和固体温度 RMSE 分别为 {fmt(finite(final['best_high_re_fluid_temperature_RMSE_K'], '高流速流体温度误差'))} K 和 {fmt(finite(final['best_high_re_solid_temperature_RMSE_K'], '高流速固体温度误差'))} K。

## 5. 随机初值、数据量与装填变化

三次随机初值计算中，稳态固体温度归一化误差的最大变异系数为 {fmt(finite(robustness['maximum_steady_seed_cv_percent'], '稳态随机初值变异系数'))}%，瞬态温度误差的最大标准差为 {fmt(finite(robustness['maximum_transient_seed_std_K'], '瞬态随机初值标准差'))} K。对于最终稳态模型，训练工况从 9 个增加到 36 个时，测试误差由 {fmt(finite(robustness['steady_learning_low_count_nrmse'], '少数据稳态误差'))} 降至 {fmt(finite(robustness['steady_learning_high_count_nrmse'], '多数据稳态误差'))}；瞬态训练从 3 条单方向曲线增加到 6 条双向曲线时，固体温度 RMSE 由 {fmt(finite(robustness['transient_three_curve_RMSE_range_K'][0], '三曲线最低误差'))}--{fmt(finite(robustness['transient_three_curve_RMSE_range_K'][1], '三曲线最高误差'))} K 变为 {fmt(finite(robustness['transient_six_curve_RMSE_K'], '六曲线误差'))} K。

第二套球形颗粒装填的 9 个匹配工况表明，出口温度和固体最高温度的最大变化分别为 {fmt(finite(packing['outlet_temperature'], '装填出口温度变化'))}% 和 {fmt(finite(packing['maximum_solid_temperature'], '装填最高温度变化'))}%，而压降最大变化达到 {fmt(finite(packing['pressure_drop'], '装填压降变化'))}%。因此，热学整体量对这两套装填相对稳定，但水力响应不能只用平均孔隙率代替真实颗粒结构。

## 6. 适用范围

本文结论适用于已经计算的局部静态球形颗粒床、文献给定工况范围以及固定流场下的热演化。全文域网格没有通过流体区基础检查，因此没有启动全文域稳态求解；全耦合启动计算在形成可用轨迹前越出了已登记的氦物性范围，因此不用于模型精度排名。移动颗粒、接触结构演化、中子加热分布和包层尺度流道仍需新的参考计算。

## 7. 结论

本文建立了一个以三维 OpenFOAM 数据为依据、按完整工况检验的球床传热降阶建模流程。结果表明，物理约束有助于同时检查温度、能量、壁面热流和热点，但并不保证单一温度指标必然优于纯数据模型；模型优劣必须在未见工况、高流速曲线和不同颗粒装填上分别验证。最终模型可用于已验证范围内的快速热场预测和参数研究，不能外推为全文域或全耦合启动过程的精度证明。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final-narrative-summary", type=Path, required=True
    )
    parser.add_argument(
        "--mesh-sensitivity-summary", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())

    final = load_json(args.final_narrative_summary, FINAL_STATUS)
    mesh = load_json(args.mesh_sensitivity_summary, MESH_STATUS)
    text = build_text(final, mesh).strip() + "\n"
    if len(text) > 9000:
        raise ValueError("中文便读版过长")
    forbidden = ("尚待", "待完成", "完成后", "预计结果")
    if any(token in text for token in forbidden):
        raise ValueError("中文便读版仍含未完成措辞")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    payload = {
        "status": "completed_p418_concise_chinese_reader",
        "character_count": len(text),
        "section_count": text.count("\n## "),
        "steady_case_count": 60,
        "fixed_flow_trajectory_count": 12,
        "high_re_curve_count": 6,
        "independent_packing_case_count": 9,
        "full_domain_solver_started": False,
        "fully_coupled_accuracy_claimed": False,
        "new_physical_parameters": [],
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "source_files": [
            {
                "path": str(args.final_narrative_summary),
                "sha256": sha256(args.final_narrative_summary),
            },
            {
                "path": str(args.mesh_sensitivity_summary),
                "sha256": sha256(args.mesh_sensitivity_summary),
            },
        ],
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
