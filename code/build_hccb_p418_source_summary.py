#!/usr/bin/env python3
"""Combine P418 physical and numerical sources into one readable summary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


MODEL_CN = {
    "PINO-paper coordinate PINN control": ("经典坐标PINN", "稳态物理约束对照模型"),
    "RIGNO-style regional graph operator": ("RIGNO式区域图神经算子", "非结构流体--颗粒区域稳态预测"),
    "Transolver": ("Transolver", "比较长距离区域热耦合的Transformer"),
    "Temporal Transformer trajectory operator": ("时间Transformer", "预测出口温度、壁面热功率等完整时间曲线"),
    "Published-component spatiotemporal regional operator": ("区域图--Transformer时空算子", "预测12条真实热阶跃的三维温度场"),
    "DPOT": ("DPOT自回归去噪Transformer", "保留为今后多堆积、大样本预训练候选"),
    "Volume-weighted DMDc baseline": ("体积加权DMDc", "传统线性瞬态降阶对照"),
    "Snapshot-POD low-rank temperature-residual correction": ("快照POD低秩残差修正", "使用少量空间模态修正确定性模型的温度误差"),
    "PDE-Refiner-style diffusion refinement": ("PDE-Refiner式扩散残差修正", "修正图--Transformer剩余温度误差并给出不确定范围"),
}

PRECEDENT_CN = {
    "10.3303/CET24114068": (
        "HCCB球床有效渗透率和导热系数反演",
        "二维坐标PINN，由孔隙尺度压力、速度和温度场反演宏观参数",
    ),
    "10.1016/j.ijheatmasstransfer.2025.126970": (
        "低流速球床有效扩散特性反演",
        "二维坐标PINN，由孔隙尺度浓度场反演有效扩散系数",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def esc(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def require_nonempty(
    rows: list[dict[str, str]], fields: tuple[str, ...], label: str
) -> None:
    for line_number, row in enumerate(rows, start=2):
        missing = [field for field in fields if not row.get(field, "").strip()]
        if missing:
            raise ValueError(
                f"{label} row {line_number} has empty fields: {', '.join(missing)}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--physical-sources",
        type=Path,
        default=Path("parameters/hccb_p418_physical_parameter_sources.csv"),
    )
    parser.add_argument(
        "--equation-map",
        type=Path,
        default=Path("parameters/hccb_p418_equation_input_map.csv"),
    )
    parser.add_argument(
        "--literature-manifest",
        type=Path,
        default=Path("parameters/literature_parameter_manifest.csv"),
    )
    parser.add_argument(
        "--actual-cases",
        type=Path,
        default=Path("results/hccb_p418_60_actual_case_input_check/summary.json"),
    )
    parser.add_argument(
        "--step-plan",
        type=Path,
        default=Path("parameters/hccb_p418_transient_step_plan.json"),
    )
    parser.add_argument(
        "--architecture-sources",
        type=Path,
        default=Path("parameters/hccb_p418_ai_architecture_sources.json"),
    )
    parser.add_argument(
        "--numerical-settings",
        type=Path,
        default=Path("parameters/hccb_p418_model_numerical_settings.csv"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/hccb_p418_source_summary.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("parameters/HCCB_P418_PARAMETER_AND_MODEL_SOURCES_CN.md"),
    )
    args = parser.parse_args()

    physical = csv_rows(args.physical_sources)
    equations = csv_rows(args.equation_map)
    literature = csv_rows(args.literature_manifest)
    actual = json.loads(args.actual_cases.read_text(encoding="utf-8"))
    step_plan = json.loads(args.step_plan.read_text(encoding="utf-8"))
    registry = json.loads(args.architecture_sources.read_text(encoding="utf-8"))
    numerical = csv_rows(args.numerical_settings)

    require_nonempty(
        physical,
        (
            "parameter_id",
            "物理量",
            "采用值或关系式",
            "单位",
            "在本研究中的用途",
            "文献",
            "链接或DOI",
            "原文位置说明",
        ),
        "physical source table",
    )
    require_nonempty(
        equations,
        (
            "类别",
            "物理量或方程",
            "符号或关系",
            "文献参数编号",
            "采用值或关系",
            "OpenFOAM位置",
            "Python实现",
            "在神经网络中的作用",
            "说明",
        ),
        "equation input table",
    )
    require_nonempty(
        numerical,
        (
            "model",
            "setting",
            "value",
            "setting_type",
            "primary_source",
            "source_path",
            "implementation_path",
            "is_physical_parameter",
            "explanation_cn",
        ),
        "model numerical setting table",
    )

    physical_ids = {row["parameter_id"] for row in physical}
    if len(physical_ids) != len(physical):
        raise ValueError("physical source table contains duplicate parameter ids")
    literature_by_id = {row["parameter_id"]: row for row in literature}
    if len(literature_by_id) != len(literature):
        raise ValueError("literature parameter manifest contains duplicate parameter ids")
    public_by_title: dict[str, str] = {}
    for row in literature:
        link = row["source_url_or_doi"]
        if link.startswith(("http://", "https://")):
            public_by_title.setdefault(row["source_title"], link)
    for row in physical:
        parameter_id = row["parameter_id"]
        manifest_row = literature_by_id.get(parameter_id)
        if manifest_row is None:
            raise ValueError(f"physical input is missing from literature manifest: {parameter_id}")
        if manifest_row["status"] != "extracted":
            raise ValueError(
                f"physical input is not an extracted literature value: {parameter_id} "
                f"({manifest_row['status']})"
            )
        if manifest_row["value"] != row["采用值或关系式"]:
            raise ValueError(f"physical input value differs from literature manifest: {parameter_id}")
        if manifest_row["unit"] != row["单位"]:
            raise ValueError(f"physical input unit differs from literature manifest: {parameter_id}")
        if manifest_row["source_title"] != row["文献"]:
            raise ValueError(f"physical input source title differs: {parameter_id}")
        link = row["链接或DOI"]
        row["public_link"] = (
            link if link.startswith(("http://", "https://")) else public_by_title.get(row["文献"], "")
        )
    if any(not row["public_link"] for row in physical):
        raise ValueError("one or more physical inputs lack a public source link")
    mapped_ids: set[str] = set()
    for row in equations:
        ids = {item.strip() for item in row["文献参数编号"].split(";") if item.strip()}
        if not ids or not ids.issubset(physical_ids):
            raise ValueError(f"equation source differs: {row['物理量或方程']}")
        mapped_ids.update(ids)
    if mapped_ids != physical_ids:
        missing = sorted(physical_ids - mapped_ids)
        extra = sorted(mapped_ids - physical_ids)
        raise ValueError(
            f"physical/equation parameter coverage differs; missing={missing}, extra={extra}"
        )
    if not set(actual["physical_parameter_ids"]).issubset(physical_ids):
        raise ValueError("actual OpenFOAM cases use an unlisted physical parameter")
    if actual["case_count"] != 60:
        raise ValueError("actual P418 matrix is not the complete 60-condition matrix")
    if actual["new_fitted_physical_parameters"]:
        raise ValueError("actual cases contain newly fitted physical parameters")
    if not actual["all_openfoam_dictionary_values_match_registered_sources"]:
        raise ValueError("one or more OpenFOAM dictionaries differ from the source table")
    if not actual.get("all_operating_points_are_exact_P418_values"):
        raise ValueError("one or more OpenFOAM operating points differ from P418")
    if not actual.get("all_cases_share_one_fixed_mesh"):
        raise ValueError("the 60 OpenFOAM cases do not share one fixed mesh")
    geometry = actual.get("geometry_sources")
    if not isinstance(geometry, dict):
        raise ValueError("actual OpenFOAM input result lacks the geometry-source check")
    if not geometry.get("all_published_geometry_and_meshing_inputs_match"):
        raise ValueError("one or more packing or meshing inputs differ from the source table")
    if not geometry.get("fine_local_crop_is_a_computed_geometry_result"):
        raise ValueError("the fine local crop is not identified as a computed geometry result")

    actual_conditions = {row["condition_id"] for row in actual["cases"]}
    sequences = step_plan["sequences"]
    if len(sequences) != 12 or step_plan["source_parameter_id"] != "P418":
        raise ValueError("transient plan must contain 12 P418 endpoint sequences")
    for sequence in sequences:
        endpoints = {sequence["source_condition_id"], sequence["target_condition_id"]}
        if not endpoints.issubset(actual_conditions):
            raise ValueError(f"transient endpoint is outside P418: {sequence['sequence_id']}")
    if step_plan["new_physical_parameters"]:
        raise ValueError("transient plan contains new physical parameters")

    project_root = Path(__file__).resolve().parents[1]
    precedents = registry.get("domain_specific_pinn_precedents", [])
    expected_precedent_dois = set(PRECEDENT_CN)
    precedent_dois = {str(row.get("doi", "")).strip() for row in precedents}
    if precedent_dois != expected_precedent_dois:
        raise ValueError(
            "direct HCCB PINN precedent set differs; "
            f"expected={sorted(expected_precedent_dois)}, actual={sorted(precedent_dois)}"
        )
    missing_settings = {
        "network depth",
        "network width",
        "learning rate",
        "batch size",
        "random seed",
    }
    for row in precedents:
        required = (
            "paper",
            "venue",
            "doi",
            "paper_url",
            "physical_problem",
            "reported_method",
            "reported_training_data",
            "reported_loss_weights",
            "use_in_this_project",
        )
        missing = [key for key in required if not str(row.get(key, "")).strip()]
        if missing:
            raise ValueError(
                f"direct HCCB PINN precedent {row.get('doi', '<missing doi>')} "
                f"lacks: {', '.join(missing)}"
            )
        if not row["paper_url"].startswith((("http://", "https://"))):
            raise ValueError(f"direct HCCB PINN precedent lacks a public URL: {row['doi']}")
        if not missing_settings.issubset(set(row.get("unreported_settings", []))):
            raise ValueError(
                f"unreported settings are incomplete for direct HCCB PINN precedent: {row['doi']}"
            )
        for path_key, hash_key in (
            ("local_source_pdf", "local_source_pdf_sha256"),
            ("local_text", "local_text_sha256"),
        ):
            source = project_root / str(row.get(path_key, ""))
            expected_hash = str(row.get(hash_key, "")).strip()
            if not source.is_file() or not expected_hash:
                raise ValueError(
                    f"direct HCCB PINN precedent lacks local {path_key}: {row['doi']}"
                )
            if sha256(source) != expected_hash:
                raise ValueError(
                    f"direct HCCB PINN precedent {path_key} hash differs: {row['doi']}"
                )
        method_note = str(row.get("local_method_note", "")).strip()
        if method_note and not (project_root / method_note).is_file():
            raise ValueError(
                f"direct HCCB PINN precedent method note is missing: {row['doi']}"
            )
        if "a source of new pebble-bed physical parameters" not in row.get("not_used_as", []):
            raise ValueError(
                f"physical-parameter boundary is missing for direct HCCB PINN precedent: {row['doi']}"
            )

    architectures = registry["architectures"]
    architecture_names = [str(row.get("name", "")).strip() for row in architectures]
    if any(not name for name in architecture_names):
        raise ValueError("one or more model architectures lack a name")
    if len(set(architecture_names)) != len(architecture_names):
        raise ValueError("model architecture registry contains duplicate names")
    for row in architectures:
        missing = [
            key
            for key in ("role", "paper", "venue", "paper_url", "status")
            if not str(row.get(key, "")).strip()
        ]
        if missing:
            raise ValueError(
                f"model architecture {row['name']} lacks: {', '.join(missing)}"
            )
        for component in row.get("published_component_sources", []):
            if not str(component.get("paper", "")).strip() or not str(
                component.get("paper_url", "")
            ).startswith(("http://", "https://")):
                raise ValueError(
                    f"published component source is incomplete: {row['name']}"
                )
    if any(not row.get("paper_url", "").startswith(("http://", "https://")) for row in architectures):
        raise ValueError("one or more model architectures lack a paper URL")
    if any(row["is_physical_parameter"].strip().lower() != "no" for row in numerical):
        raise ValueError("a model numerical setting is incorrectly labelled as a physical input")
    for row in numerical:
        for field in ("source_path", "implementation_path"):
            paths = [
                project_root / item.strip()
                for item in row[field].split(";")
                if item.strip()
            ]
            if not paths or any(not path.is_file() for path in paths):
                raise ValueError(
                    f"model numerical setting points to a missing {field}: {row[field]}"
                )

    payload = {
        "status": "p418_physical_and_model_sources_complete",
        "physical_parameter_count": len(physical),
        "all_physical_parameter_statuses_extracted": True,
        "literature_manifest_records_match_physical_table": True,
        "equation_or_boundary_count": len(equations),
        "physical_parameters_used_by_equations": sorted(mapped_ids),
        "all_physical_parameters_are_used_by_equations": True,
        "actual_openfoam_case_count": actual["case_count"],
        "actual_case_values_match_sources": True,
        "actual_operating_points_match_P418": True,
        "actual_cases_share_one_fixed_mesh": True,
        "packing_and_meshing_inputs_match_sources": True,
        "fine_local_crop_is_a_computed_geometry_result": True,
        "fine_local_crop_lengths_dp": geometry["fine_local_crop_lengths_dp"],
        "fine_local_retained_particle_fragments": geometry[
            "fine_local_retained_particle_fragments"
        ],
        "fine_local_triangulated_porosity": geometry["fine_local_triangulated_porosity"],
        "transient_sequence_count": len(sequences),
        "transient_endpoints_are_published_p418_conditions": True,
        "domain_specific_pinn_precedent_count": len(precedents),
        "domain_specific_pinn_precedent_dois": sorted(precedent_dois),
        "all_domain_specific_pinn_precedent_files_match_hashes": True,
        "unreported_direct_precedent_settings_are_not_invented": True,
        "architecture_count": len(architectures),
        "model_numerical_setting_count": len(numerical),
        "model_numerical_settings_are_not_physical_parameters": True,
        "all_model_setting_source_and_implementation_files_exist": True,
        "new_physical_parameters": [],
        "source_files": {
            "physical_sources": str(args.physical_sources),
            "literature_manifest": str(args.literature_manifest),
            "equation_map": str(args.equation_map),
            "actual_cases": str(args.actual_cases),
            "step_plan": str(args.step_plan),
            "architecture_sources": str(args.architecture_sources),
            "numerical_settings": str(args.numerical_settings),
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# HCCB P418 参数、方程和模型来源总表",
        "",
        "这份总表把球床物理输入、有限体积方程、实际OpenFOAM算例、热阶跃和人工智能模型的来源放在一起。网络层数、学习率和时间步等数值设置与球床物理参数分开。",
        "",
        "## 结论先说",
        "",
        f"- 当前P418换热计算使用`{len(physical)}`项物理输入，全部在文献参数表中标为已摘录，数值、单位和来源标题逐项一致。",
        f"- 这些物理输入进入`{len(equations)}`处方程或边界条件；60个OpenFOAM稳态工况和12条热阶跃没有增加新的物理参数。",
        "- 颗粒直径、装填目标孔隙率、大球床尺寸、截取区域、近壁位置和网格用粒径也已与来源表逐项对照。",
        f"- 已保存并核对`{len(precedents)}`篇直接面向HCCB球床的PINN研究。它们证明这一方法确实已用于增殖球床，但没有公开网络层数、宽度、学习率、批量大小和随机种子，因此这些设置不能从原文照搬或自行补写。",
        "- PINN、Transformer、图神经算子和扩散模型的层数、学习率、批量等属于计算设置，不作为球床物性或运行参数。",
        "- 仓库其他旧研究分支中仍有公开资料未给出的几何或实验坐标，但这些内容没有进入当前P418换热计算。",
        "",
        "## 1. 球床换热物理输入",
        "",
        "| 编号 | 物理量 | 采用值或关系 | 单位 | 来源 |",
        "|---|---|---|---|---|",
    ]
    for row in physical:
        link = f"[{esc(row['文献'])}]({row['public_link']})"
        lines.append(
            f"| {row['parameter_id']} | {esc(row['物理量'])} | "
            f"{esc(row['采用值或关系式'])} | {esc(row['单位'])} | {link} |"
        )

    lines.extend([
        "",
        "## 2. 输入量进入的方程",
        "",
        "| 物理量或方程 | 文献参数 | 程序位置 | 在模型中的作用 |",
        "|---|---|---|---|",
    ])
    for row in equations:
        lines.append(
            f"| {esc(row['物理量或方程'])} | {esc(row['文献参数编号'])} | "
            f"{esc(row['Python实现'])} | {esc(row['在神经网络中的作用'])} |"
        )

    lines.extend([
        "",
        "## 3. 实际计算工况",
        "",
        f"- OpenFOAM稳态算例：`{actual['case_count']}`组，对应P418的`5 x 4 x 3`完整工况矩阵。",
        "- 60个算例的入口速度、入口温度、颗粒发热率、壁温、压力和材料物性已逐个与来源表对照。",
        f"- 当前精细局部域为`{' x '.join(f'{value:.4g}' for value in geometry['fine_local_crop_lengths_dp'])} dp`，包含`{geometry['fine_local_retained_particle_fragments']}`个颗粒片段，三角网格孔隙率为`{geometry['fine_local_triangulated_porosity']:.6f}`。这些是从文献规定的大球床和截取方式计算得到的局部几何，不作为新增物性。",
        f"- 热阶跃：`{len(sequences)}`条，每一个起点和终点都是上述60个P418稳态工况之一。",
        "- 实际网格孔隙率是固定颗粒堆积和表面三角化后的几何计算结果，不冒充文献常数。",
        "",
        "## 4. 本领域已有的直接PINN工作",
        "",
        "| 已有工作 | 解决的问题 | 原文给出的训练信息 | 原文未给出的设置 | 在本研究中的用法 |",
        "|---|---|---|---|---|",
    ])
    for row in precedents:
        label, problem = PRECEDENT_CN[row["doi"]]
        paper = f"[{label}]({row['paper_url']})"
        unreported = "、".join(row["unreported_settings"])
        lines.append(
            f"| {paper} | {esc(problem)} | {esc(row['reported_training_data'])} "
            f"{esc(row['reported_method'])} | {esc(unreported)} | "
            f"只支持坐标PINN对照和物理约束思路；不作为当前三维模型精度、网络规模或新物性的来源。 |"
        )
    lines.extend([
        "",
        "这两篇工作都处理二维宏观反演。当前研究增加三维流固共轭换热、完整热阶跃、热点位置和跨装填预测，因此不能把已有二维结果当作当前三维模型的验证结果。",
        "",
        "## 5. 人工智能和传统降阶模型",
        "",
        "| 模型 | 在本研究中的作用 | 主要来源 |",
        "|---|---|---|",
    ])
    for row in architectures:
        component_sources = row.get("published_component_sources", [])
        if component_sources:
            source = "; ".join(
                f"[{esc(item['paper'])}]({item['paper_url']})"
                for item in component_sources
            )
        else:
            source = f"[{esc(row['paper'])}]({row['paper_url']})"
        label, role = MODEL_CN.get(row["name"], (row["name"], row["role"]))
        lines.append(f"| {esc(label)} | {esc(role)} | {source} |")

    setting_counts = Counter(row["setting_type"] for row in numerical)
    lines.extend([
        "",
        "## 6. 数值设置与物理参数的区别",
        "",
        f"`parameters/hccb_p418_model_numerical_settings.csv`共记录`{len(numerical)}`项模型数值设置，全部明确标为不是球床物理参数。",
        "",
    ])
    for kind, count in sorted(setting_counts.items()):
        lines.append(f"- `{kind}`: {count}项。")
    lines.extend([
        "",
        "## 7. 可重复生成",
        "",
        "```bash",
        "python3 code/build_hccb_p418_source_summary.py",
        "```",
        "",
        "机器可读结果保存在`results/hccb_p418_source_summary.json`。",
    ])
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
