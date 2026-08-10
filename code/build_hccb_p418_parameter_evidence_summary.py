#!/usr/bin/env python3
"""Build a plain-Chinese parameter, equation and source cross-reference."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def build(sources: Path, evidence: Path, equations: Path) -> str:
    source_rows = read_rows(sources)
    evidence_rows = read_rows(evidence)
    evidence_by_id = {row["parameter_id"]: row for row in evidence_rows}
    project_root = evidence.resolve().parents[1]
    missing_local_sources: dict[str, list[str]] = {}
    for row in evidence_rows:
        missing = [
            item.strip()
            for item in row["local_evidence_paths"].split(";")
            if item.strip() and not (project_root / item.strip()).is_file()
        ]
        if missing:
            missing_local_sources[row["parameter_id"]] = missing
    available_parameter_count = len(evidence_rows) - len(missing_local_sources)
    equation_use: dict[str, list[str]] = defaultdict(list)
    code_use: dict[str, list[str]] = defaultdict(list)
    for row in read_rows(equations):
        for parameter_id in row["文献参数编号"].split(";"):
            parameter_id = parameter_id.strip()
            if parameter_id:
                equation_use[parameter_id].append(row["物理量或方程"])
                code_use[parameter_id].extend(
                    item.strip()
                    for item in row["Python实现"].split(";")
                    if item.strip()
                )

    lines = [
        "# P418物理参数、原文、方程和代码对应表",
        "",
        "这份表只回答四件事：每个物理量取什么值、原文在哪里、进入哪条方程、由哪段程序读取。22项物理参数均来自已登记文献，没有为了训练PINN、图--Transformer或扩散模型另填球床物性。",
        "",
        "## 资料覆盖情况",
        "",
        f"- 当前{available_parameter_count}/22项参数的来源文件可在项目中直接读取。",
        (
            "- 尚缺本地全文副本的参数："
            + "、".join(sorted(missing_local_sources))
            + "。这些参数的DOI、采用值和原始提取位置已保留，但在重新取得出版社或作者版全文前，不声称它们已完成本地复核。"
            if missing_local_sources
            else "- 22项参数的来源文件均可在项目中直接读取。"
        ),
        "- P430可在本地保存的CIAAW 2024官方网页中直接复核。",
        "- P428的全部焓系数来自Kleykamp 1996 ScienceDirect出版社公开摘要，P429为其解析导数；没有把元数据文件写成论文全文。本地另存Kleykamp 2000 J-STAGE公开论文；该文表1给出Li4SiO4在298 K和1100 K时的比热为182.1和304.2 J/(mol K)，与P429计算值四舍五入一致。",
        "- P431的精确938 K和996 K来自同一出版社摘要。Kleykamp 2000公开全文进一步给出648--683 °C和713--735 °C两段相变影响区及900和630 J/mol额外焓吸收；这些数据只用于判断计算温度是否进入平滑比热关系不充分的温区，不凭空假设热容峰形。FZKA5515官方报告又给出约940 K和998 K，Asou等独立研究组的出版社摘要还报告约885 K、930 K和985 K热容异常。",
        "- P406只用于满足稳态OpenFOAM材料字典格式，不进入12条正式热阶跃。正式瞬态储热使用P428--P431。",
        "",
        "## 逐项对应",
        "",
        "|编号|物理量|采用值或关系|原文位置和本地资料|进入的方程或边界|主要程序|",
        "|---|---|---|---|---|---|",
    ]
    for row in source_rows:
        parameter_id = row["parameter_id"]
        evidence_row = evidence_by_id[parameter_id]
        paths = "<br>".join(
            f"`{item.strip()}`"
            for item in evidence_row["local_evidence_paths"].split(";")
            if item.strip()
        )
        location = evidence_row["evidence_location"].replace("|", "/")
        uses = "；".join(dict.fromkeys(equation_use[parameter_id]))
        programs = "<br>".join(
            f"`{item}`" for item in dict.fromkeys(code_use[parameter_id])
        )
        value = row["采用值或关系式"].replace("|", "/")
        unit = row["单位"].strip()
        if unit:
            value = f"{value} [{unit}]"
        lines.append(
            "|{pid}|{quantity}|{value}|{location}<br>{paths}|{uses}|{programs}|".format(
                pid=parameter_id,
                quantity=row["物理量"],
                value=value,
                location=location,
                paths=paths,
                uses=uses,
                programs=programs,
            )
        )

    lines.extend(
        [
            "",
            "## 使用时需要注意的物理边界",
            "",
            "1. P048--P050和P390描述文献球床及其截取区域；当前精细局部网格来自该区域内部，不能写成完整12.5dp×12.5dp×10dp几何的原样复现。",
            "2. P404的1%缩径只用于网格消除点接触，因此当前模型解析氦气对流、气固导热和颗粒内发热，不包含真实受压颗粒接触导热。",
            "3. P428--P429是298--1300 K的平滑纯Li4SiO4量热关系。Kleykamp 2000公开论文说明热容由平滑焓曲线求导，并指出该方法适用于没有二级相变的温区。该文还给出648--683 °C和713--735 °C两段影响区及900和630 J/mol额外焓吸收，但没有给出唯一解析峰形，因此当前程序只统计温度场进入这些原文温区的程度，不修改OpenFOAM比热。P431的精确938 K和996 K取自Kleykamp 1996出版社摘要；FZKA5515给出约940 K和998 K，Asou等独立研究又在约885 K、930 K和985 K观察到热容异常。",
            "4. P430按天然同位素组成的简化标准原子量换算。若研究对象改为明确富集6Li的颗粒，应使用材料实际同位素组成重新计算摩尔质量。",
            "5. 神经网络层数、隐藏维数、学习率和扩散步数属于数值设置，不在本表中冒充材料参数。",
            "",
            "## 自动检查",
            "",
            "```bash",
            "python3 code/verify_hccb_p418_parameter_evidence_files.py \\",
            "  --output results/hccb_p418_parameter_evidence/summary.json",
            "python3 code/build_hccb_p418_parameter_evidence_summary.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        type=Path,
        default=ROOT / "parameters/hccb_p418_physical_parameter_sources.csv",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=ROOT / "parameters/hccb_p418_physical_parameter_evidence_files.csv",
    )
    parser.add_argument(
        "--equations",
        type=Path,
        default=ROOT / "parameters/hccb_p418_equation_input_map.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "parameters/HCCB_P418_PARAMETER_EVIDENCE_CN.md",
    )
    args = parser.parse_args()
    args.output.write_text(
        build(args.sources.resolve(), args.evidence.resolve(), args.equations.resolve()),
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
