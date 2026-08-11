#!/usr/bin/env python3
"""Summarize how physical parameters and numerical settings enter P418."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

LITERATURE_OR_OFFICIAL = {
    "published_architecture",
    "published_component_adaptation",
    "official_code_architecture",
    "official_code_constant",
    "official_code_training",
    "published_algorithm",
    "source_backed_output_parameterization",
    "official_OpenFOAM13_software_constant",
}
CASE_DERIVED = {
    "data_derived",
    "measured_compute_setting",
    "problem_geometry",
    "finite_volume_definition",
}
PREDECLARED_PROJECT = {
    "predeclared_baseline",
    "predeclared_numerical_scan",
    "predeclared_selection_rule",
    "project_adaptation",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def build(
    root: Path, require_local_source_files: bool = True
) -> tuple[dict[str, object], str]:
    physical = read_csv(root / "parameters/hccb_p418_physical_parameter_sources.csv")
    equations = read_csv(root / "parameters/hccb_p418_equation_input_map.csv")
    numerical = read_csv(root / "parameters/hccb_p418_model_numerical_settings.csv")
    observables = read_csv(root / "parameters/hccb_p418_experimental_observable_matrix.csv")
    observation_sources = json.loads(
        (root / "parameters/hccb_p418_experimental_observation_sources.json").read_text(
            encoding="utf-8"
        )
    )
    observation_parameter_ids = {
        parameter_id
        for family in observation_sources["observation_families"]
        for parameter_id in family["source_parameter_ids"]
    }
    physical_ids = {row["parameter_id"] for row in physical}
    equation_ids = {
        item.strip()
        for row in equations
        for item in row["文献参数编号"].split(";")
        if item.strip()
    }
    unused_physical = sorted(physical_ids - equation_ids)
    unknown_equation_ids = sorted(equation_ids - physical_ids)
    numerical_types = Counter(row["setting_type"] for row in numerical)
    classified_types = LITERATURE_OR_OFFICIAL | CASE_DERIVED | PREDECLARED_PROJECT
    unknown_setting_types = sorted(set(numerical_types) - classified_types)
    template_root = root / "experimental_data_templates"
    template_rows = {
        path.name: len(read_csv(path)) for path in sorted(template_root.glob("*.csv"))
    }
    all_templates_empty = all(count == 0 for count in template_rows.values())
    all_numerical_nonphysical = all(
        row["is_physical_parameter"] == "no" for row in numerical
    )
    source_paths_exist = all(
        all(
            bool(item.strip())
            and (
                (root / item.strip()).exists()
                if require_local_source_files
                else True
            )
            for item in row["source_path"].split(";")
        )
        for row in numerical
    )
    summary = {
        "status": "completed_p418_parameter_and_model_input_summary",
        "physical_parameter_count": len(physical),
        "physical_parameters_used_by_equations": len(physical_ids & equation_ids),
        "unused_physical_parameter_ids": unused_physical,
        "unknown_equation_parameter_ids": unknown_equation_ids,
        "equation_map_row_count": len(equations),
        "model_numerical_setting_count": len(numerical),
        "model_setting_type_counts": dict(sorted(numerical_types.items())),
        "literature_or_official_model_setting_count": sum(
            numerical_types[item] for item in LITERATURE_OR_OFFICIAL
        ),
        "case_or_data_derived_model_setting_count": sum(
            numerical_types[item] for item in CASE_DERIVED
        ),
        "predeclared_project_comparison_setting_count": sum(
            numerical_types[item] for item in PREDECLARED_PROJECT
        ),
        "unknown_model_setting_types": unknown_setting_types,
        "all_model_settings_are_nonphysical": all_numerical_nonphysical,
        "all_model_setting_source_paths_exist": source_paths_exist,
        "model_setting_source_verification_mode": (
            "local_files" if require_local_source_files else "registered_metadata"
        ),
        "experimental_observable_count": len(observables),
        "experimental_observation_source_count": len(observation_parameter_ids),
        "experimental_template_rows": template_rows,
        "experimental_templates_contain_no_measurements": all_templates_empty,
        "new_physical_parameters": [],
    }
    if (
        unused_physical
        or unknown_equation_ids
        or unknown_setting_types
        or not all_numerical_nonphysical
        or not source_paths_exist
        or not all_templates_empty
    ):
        raise ValueError(json.dumps(summary, ensure_ascii=False, indent=2))

    project_rows = [
        row for row in numerical if row["setting_type"] in PREDECLARED_PROJECT
    ]
    lines = [
        "# P418参数怎样进入物理计算和神经网络",
        "",
        "## 先说结论",
        "",
        f"- 球床几何、氦气物性、颗粒热物性和运行工况共有`{len(physical)}`项，全部有原文位置，并且全部进入方程、边界条件或几何生成程序。",
        f"- 神经网络和传统基线另有`{len(numerical)}`项数值设置。这些设置全部明确标为“非物理参数”，不能改变颗粒直径、孔隙率、导热率、热容、气体物性或运行工况。",
        f"- 其中`{summary['literature_or_official_model_setting_count']}`项来自论文、作者公开程序或文献支持的输出范围，`{summary['case_or_data_derived_model_setting_count']}`项由当前网格、显存实测或检查数据确定，`{summary['predeclared_project_comparison_setting_count']}`项是模型比较前预先写明的候选或选择方法。",
        f"- 实验接口覆盖`{len(observables)}`类测量量，相关文献参数登记为`{len(observation_parameter_ids)}`条。五张实验数据表目前都只有表头，没有填入人为测量值。",
        "",
        "## 为什么不能把所有数字都叫“文献参数”",
        "",
        "颗粒直径、孔隙率、氦气黏度、颗粒导热率和发热率必须来自目标材料及装置文献。网络批量大小、POD候选阶数和训练轮次不是材料常数：前两者受数据量和显存影响，训练轮次由检查工况选择。把这些数值伪装成“文献物性”反而不科学。本项目的做法是把物理参数锁定为文献值，把网络数值设置单独登记，并且不让独立测试工况参与选择。",
        "",
        "## 本项目预先写明的数值比较设置",
        "",
        "|模型|设置|采用方式|为什么不是物理参数|",
        "|---|---|---|---|",
    ]
    for row in project_rows:
        lines.append(
            f"|{row['model']}|{row['setting']} = `{row['value']}`|"
            f"{row['primary_source']}|{row['explanation_cn']}|"
        )
    lines.extend(
        [
            "",
            "## 正式计算时遵守的顺序",
            "",
            "1. 先用文献物理参数完成OpenFOAM三维参考计算。",
            "2. 只用训练工况计算归一化量和模型参数。",
            "3. 用检查工况选择训练轮次、POD/DMDc阶数或损失组合。",
            "4. 模型固定后再读取独立工况和独立颗粒排列。",
            "5. 实验数据只在真实传感器位置或边界积分量上比较；没有测量的数据保持空白。",
            "",
            f"详细的{len(physical)}项物理参数及原文位置见`parameters/HCCB_P418_PARAMETER_EVIDENCE_CN.md`；{len(numerical)}项模型设置见`parameters/hccb_p418_model_numerical_settings_CN.md`。",
            "",
        ]
    )
    return summary, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/hccb_p418_parameter_use",
    )
    args = parser.parse_args()
    summary, document = build(ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "P418_参数怎样进入研究_CN.md").write_text(
        document,
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
