#!/usr/bin/env python3
"""Check the separate high-Re P418 trajectory-combination test plan."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHANGED_KEYS = (
    "inlet_velocity_m_s",
    "inlet_temperature_K",
    "solid_heat_source_MW_m3",
)


def read_json(path: Path) -> dict:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def condition_table(path: Path) -> dict[str, dict[str, float]]:
    with path.resolve().open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "condition_id",
        *CHANGED_KEYS,
        "particle_reynolds_inlet",
        "prandtl",
        "particle_peclet_inlet",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("dimensionless condition table lacks required columns")
    return {
        row["condition_id"]: {
            key: float(row[key])
            for key in CHANGED_KEYS
            + ("particle_reynolds_inlet", "prandtl", "particle_peclet_inlet")
        }
        for row in rows
    }


def unordered_pair(row: dict) -> tuple[str, str]:
    return tuple(sorted((row["source_condition_id"], row["target_condition_id"])))


def validate_model_use(plan: dict) -> None:
    if plan.get("data_role") != "frozen_model_independent_test_only":
        raise ValueError("high-Re data role changed")
    use = plan.get("model_use", {})
    forbidden = (
        "allowed_for_training",
        "allowed_for_normalization",
        "allowed_for_checkpoint_selection",
        "allowed_for_architecture_selection",
        "allowed_for_loss_weight_selection",
        "allowed_for_pod_basis",
        "allowed_for_diffusion_training",
    )
    if any(use.get(key) is not False for key in forbidden):
        raise ValueError("a high-Re curve was allowed to influence model fitting")


def verify(
    fixed_path: Path,
    coupled_path: Path,
    main_path: Path,
    conditions_path: Path,
) -> dict[str, object]:
    fixed = read_json(fixed_path)
    coupled = read_json(coupled_path)
    main = read_json(main_path)
    conditions = condition_table(conditions_path)

    if fixed.get("analysis_kind") != "independent_high_re_test":
        raise ValueError("fixed high-Re analysis kind changed")
    if coupled.get("analysis_kind") != "independent_high_re_test":
        raise ValueError("fully coupled high-Re analysis kind changed")
    if fixed.get("new_physical_parameters") != [] or coupled.get("new_physical_parameters") != []:
        raise ValueError("high-Re plans introduce new physical parameters")
    validate_model_use(fixed)
    validate_model_use(coupled)
    sequences = fixed["sequences"]
    if len(sequences) != 6 or coupled["sequences"] != sequences:
        raise ValueError("fixed and fully coupled plans must share six curves")
    ids = [row["sequence_id"] for row in sequences]
    if len(set(ids)) != len(ids):
        raise ValueError("high-Re sequence ids are not unique")
    main_ids = {row["sequence_id"] for row in main["sequences"]}
    if main_ids.intersection(ids):
        raise ValueError("high-Re sequence id overlaps the main twelve curves")
    main_pairs = {unordered_pair(row) for row in main["sequences"]}
    if any(unordered_pair(row) in main_pairs for row in sequences):
        raise ValueError("high-Re endpoint pair duplicates a main trajectory")

    families = Counter(row["family"] for row in sequences)
    if families != {
        "inlet_velocity_step": 2,
        "inlet_temperature_step": 2,
        "solid_heat_source_step": 2,
    }:
        raise ValueError(f"unexpected high-Re family balance: {families}")

    endpoint_rows: list[dict[str, object]] = []
    for row in sequences:
        source_id = row["source_condition_id"]
        target_id = row["target_condition_id"]
        if source_id not in conditions or target_id not in conditions:
            raise ValueError(f"{row['sequence_id']} uses an endpoint outside the P418 table")
        source = conditions[source_id]
        target = conditions[target_id]
        changed = [
            key for key in CHANGED_KEYS if source[key] != target[key]
        ]
        if len(changed) != 1:
            raise ValueError(f"{row['sequence_id']} changes {changed}")
        endpoint_rows.append(
            {
                "sequence_id": row["sequence_id"],
                "family": row["family"],
                "changed_input": changed[0],
                "source_condition_id": source_id,
                "target_condition_id": target_id,
                "source_Re": source["particle_reynolds_inlet"],
                "target_Re": target["particle_reynolds_inlet"],
                "source_Pr": source["prandtl"],
                "target_Pr": target["prandtl"],
                "source_Pe": source["particle_peclet_inlet"],
                "target_Pe": target["particle_peclet_inlet"],
            }
        )

    all_re = [value for row in endpoint_rows for value in (row["source_Re"], row["target_Re"])]
    all_pr = [value for row in endpoint_rows for value in (row["source_Pr"], row["target_Pr"])]
    all_pe = [value for row in endpoint_rows for value in (row["source_Pe"], row["target_Pe"])]
    high_re_curves = [
        row for row in endpoint_rows if max(row["source_Re"], row["target_Re"]) >= 2.4
    ]
    if len(high_re_curves) != 6:
        raise ValueError("every independent curve must reach the P418 high-inlet-Re endpoint")

    return {
        "status": "p418_high_re_independent_plan_verified_not_run",
        "sequence_count": len(sequences),
        "family_counts": dict(families),
        "pair_disjoint_from_main_twelve": True,
        "frozen_model_test_only": True,
        "new_physical_parameters": [],
        "inlet_dimensionless_range": {
            "particle_reynolds_min": min(all_re),
            "particle_reynolds_max": max(all_re),
            "prandtl_min": min(all_pr),
            "prandtl_max": max(all_pr),
            "particle_peclet_min": min(all_pe),
            "particle_peclet_max": max(all_pe),
        },
        "curves": endpoint_rows,
        "openfoam_calculation_started": False,
        "model_training_started": False,
    }


def write_outputs(summary: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    limits = summary["inlet_dimensionless_range"]
    lines = [
        "# P418高流速工况组合独立测试",
        "",
        "## 为什么增加这组测试",
        "",
        "原12条热阶跃可以比较三类操作量，但其严格端点对分离测试主要位于较低入口Re范围。这里另列6条曲线，用P418原有工况端点检验模型在高流速工况组合下的温度和换热预测。",
        "",
        "## 使用限制",
        "",
        "- 这6条曲线不参加训练；",
        "- 不用它们计算归一化量；",
        "- 不用它们选择网络、训练轮数、损失权重、POD阶数或扩散模型；",
        "- 只在主模型确定后进行一次独立预测；",
        "- 部分端点状态在原12条曲线中出现过，因此这是新轨迹组合测试，不表述为所有端点都从未出现。",
        "",
        "## 物理范围",
        "",
        (
            f"6条曲线覆盖入口颗粒Re `{limits['particle_reynolds_min']:.4f}--"
            f"{limits['particle_reynolds_max']:.4f}`、Pr `{limits['prandtl_min']:.4f}--"
            f"{limits['prandtl_max']:.4f}`、Pe `{limits['particle_peclet_min']:.4f}--"
            f"{limits['particle_peclet_max']:.4f}`。这些量由已登记文献物性和P418工况计算，"
            "不是新增输入参数，也不替代孔道内部局部Re。"
        ),
        "",
        "## 曲线",
        "",
        "| 曲线 | 类型 | 源工况 | 目标工况 | 源Re | 目标Re |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in summary["curves"]:
        lines.append(
            "| {sequence_id} | {family} | {source_condition_id} | "
            "{target_condition_id} | {source_Re:.4f} | {target_Re:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "当前只完成了工况设计和程序检查，OpenFOAM曲线尚未计算，不能提前报告模型精度。",
        ]
    )
    (output_dir / "P418_高流速独立测试_CN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixed-plan",
        type=Path,
        default=ROOT / "parameters/hccb_p418_high_re_independent_step_plan.json",
    )
    parser.add_argument(
        "--coupled-plan",
        type=Path,
        default=ROOT
        / "parameters/hccb_p418_high_re_independent_fully_coupled_step_plan.json",
    )
    parser.add_argument(
        "--main-plan",
        type=Path,
        default=ROOT / "parameters/hccb_p418_transient_step_plan.json",
    )
    parser.add_argument(
        "--conditions",
        type=Path,
        default=ROOT
        / "results/hccb_p418_inlet_dimensionless_envelope/inlet_dimensionless_conditions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/hccb_p418_high_re_independent_plan",
    )
    args = parser.parse_args()
    summary = verify(
        args.fixed_plan,
        args.coupled_plan,
        args.main_plan,
        args.conditions,
    )
    write_outputs(summary, args.output_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
