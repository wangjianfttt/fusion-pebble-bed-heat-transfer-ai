#!/usr/bin/env python3
"""Write a concise, source-linked list of physical inputs used by the P418 study."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SELECTION = [
    ("P048", "颗粒直径", "定义源球床的1 mm颗粒"),
    ("P049", "源球床目标孔隙率", "定义堆积生成目标；实际三角网格孔隙率另行计算"),
    ("P050", "初始大球床尺寸", "先生成25dp x 25dp x 10dp大球床"),
    ("P390", "文献截取区域尺寸", "从大球床截取12.5dp x 12.5dp x 10dp区域"),
    ("P404", "网格颗粒直径修正", "网格生成前将颗粒直径缩小1%，去除点接触"),
    ("P423", "靠冷却壁的截取方式", "采用一侧靠壁、横向居中的文献截取方式"),
    ("P418", "60组入口速度、入口温度和颗粒发热率", "定义全部计算工况"),
    ("P425", "冷却壁温度", "固定为635 K"),
    ("P426", "工作压力", "出口绝对压力固定为0.12 MPa"),
    ("P427", "计算域和边界类型", "采用一面恒温冷却壁，其余横向面为对称边界；当前局部球床尺寸另行说明"),
    ("P070", "氦气动力黏度", "随局部温度更新"),
    ("P071", "氦气导热系数", "随局部温度和压力更新"),
    ("P388", "氦气定压比热", "流体能量方程"),
    ("P389", "氦气密度", "随局部温度和绝对压力更新"),
    ("P092", "Li4SiO4颗粒导热系数", "颗粒导热方程"),
    ("P403", "Li4SiO4颗粒密度", "稳态和瞬态固体能量方程"),
    ("P406", "EU参考球比热关系", "只用于60组稳态端点；不进入温度阶跃计算"),
    ("P428", "纯Li4SiO4焓增量关系", "定义瞬态固体储热关系"),
    ("P429", "纯Li4SiO4比热关系", "温度阶跃OpenFOAM和PINN瞬态储热项"),
    ("P430", "纯Li4SiO4摩尔质量换算", "把P429从J/mol/K换算为J/kg/K"),
    ("P431", "纯Li4SiO4二级相变温度", "标出平滑比热关系未解析的相变温区"),
    ("P424", "氦气物性表温度范围", "保证物性表覆盖300--1000 K"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = {row["parameter_id"]: row for row in csv.DictReader(stream)}
    output_rows: list[dict[str, str]] = []
    for parameter_id, chinese_name, use in SELECTION:
        row = rows[parameter_id]
        if row["status"] != "extracted":
            raise ValueError(f"{parameter_id} is not an extracted literature value")
        output_rows.append(
            {
                "parameter_id": parameter_id,
                "物理量": chinese_name,
                "采用值或关系式": row["value"],
                "单位": row["unit"],
                "在本研究中的用途": use,
                "文献": row["source_title"],
                "链接或DOI": row["source_url_or_doi"],
                "原文位置说明": row["notes"],
            }
        )

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    lines = [
        "# P418 三维球床换热计算采用的物理参数",
        "",
        "本表列出进入60组稳态端点、12组温度阶跃和瞬态能量方程的物理参数。神经网络层数、隐藏宽度和注意力头数不是球床物理参数，另列在模型结构文件中。",
        "",
        "| 编号 | 物理量 | 采用值或关系式 | 单位 | 在计算中的用途 | 来源 |",
        "|---|---|---|---|---|---|",
    ]
    for row in output_rows:
        value = row["采用值或关系式"].replace("|", "\\|")
        public_link = row["链接或DOI"]
        if not public_link.startswith(("http://", "https://")):
            alternatives = [
                item["source_url_or_doi"]
                for item in rows.values()
                if item["source_title"] == row["文献"]
                and item["source_url_or_doi"].startswith(("http://", "https://"))
            ]
            if alternatives:
                public_link = alternatives[0]
        source = f'[{row["文献"]}]({public_link})'
        lines.append(
            f'| {row["parameter_id"]} | {row["物理量"]} | {value} | '
            f'{row["单位"]} | {row["在本研究中的用途"]} | {source} |'
        )
    lines.extend(
        [
            "",
            "## 使用范围",
            "",
            "- 60 组工况严格采用 P418 中的 5 个入口速度、4 个入口温度和 3 个颗粒发热率的全部组合。",
            "- P048--P050、P390、P404和P423共同定义源球床生成过程：先生成大球床，再截取靠冷却壁的文献区域，并在网格前把颗粒直径缩小1%以消除点接触。",
            "- 当前精细局部域是在上述文献区域中进一步截取的数值计算域；其尺寸、保留颗粒数和三角网格孔隙率都是本次网格的计算结果，不回写成文献参数。",
            "- 当前计算域是局部致密球床，不等同于 P427 中的完整 12.5dp x 12.5dp x 10dp 区域；这里只沿用文献给出的边界类型。",
            "- P406 来自 EU reference Li4SiO4，仅用于生成60组稳态端点；稳态温度场不依赖热容。",
            "- 12组温度阶跃在建算例时会把P406替换为P428--P431给出的纯Li4SiO4高温量热关系，OpenFOAM和PINN使用同一比热。",
            "- P428--P429是298--1300 K的平滑关系。它不解析P431在938 K和996 K附近的尖锐热容异常，因此接近该温区的结果必须单独标明。",
            "- P430采用天然同位素组成的摩尔质量；若以后改为富集锂颗粒，需要换成实测或明确给定的同位素组成。",
        ]
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
