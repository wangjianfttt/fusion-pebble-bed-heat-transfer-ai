#!/usr/bin/env python3
"""Check the literature-backed P418 experimental observation mapping."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return {
            row["parameter_id"].strip(): row
            for row in csv.DictReader(stream)
            if row["parameter_id"].strip()
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("parameters/hccb_p418_experimental_observation_sources.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("parameters/literature_parameter_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/hccb_p418_experimental_observation_sources"),
    )
    args = parser.parse_args()

    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    manifest = read_manifest(args.manifest)
    if sources.get("new_physical_parameters") != []:
        raise ValueError("observation mapping must not add physical parameters")
    if sources.get("p418_assigned_numeric_values") != []:
        raise ValueError("literature instrument values must not be assigned to P418")

    families = sources.get("observation_families", [])
    if not families:
        raise ValueError("observation_families is empty")
    all_ids: list[str] = []
    family_rows: list[dict[str, object]] = []
    for family in families:
        ids = [str(value) for value in family.get("source_parameter_ids", [])]
        if not ids:
            raise ValueError(f"{family.get('family_id')}: no source_parameter_ids")
        missing = sorted(set(ids) - set(manifest))
        if missing:
            raise ValueError(f"{family.get('family_id')}: missing source ids {missing}")
        for field in (
            "research_quantity_cn",
            "measured_information_cn",
            "model_comparison_cn",
            "current_use_cn",
            "limitations_cn",
        ):
            if not str(family.get(field, "")).strip():
                raise ValueError(f"{family.get('family_id')}: missing {field}")
        all_ids.extend(ids)
        family_rows.append(
            {
                "family_id": family["family_id"],
                "research_quantity_cn": family["research_quantity_cn"],
                "source_parameter_ids": ids,
                "source_statuses": sorted({manifest[value]["status"] for value in ids}),
                "current_use_cn": family["current_use_cn"],
                "limitations_cn": family["limitations_cn"],
            }
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "experimental_observation_sources_verified",
        "family_count": len(families),
        "unique_literature_parameter_count": len(set(all_ids)),
        "literature_parameter_ids": sorted(set(all_ids)),
        "families": family_rows,
        "p418_assigned_numeric_values": [],
        "new_physical_parameters": [],
        "interpretation_cn": (
            "已有文献支持内部测温、流量压降、冷却侧热量和热机械辅助观测。"
            "这些数据保留原装置含义，不直接变成P418测点、仪器精度或训练噪声。"
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# P418实验观测量与模型输出对应",
        "",
        "这份说明只回答三个问题：已有实验测过什么、模型拿什么量来比较、当前还缺什么。",
        "",
    ]
    for family in families:
        source_text = "、".join(family["source_parameter_ids"])
        lines.extend(
            [
                f"## {family['research_quantity_cn']}",
                "",
                f"- 文献记录：`{source_text}`。",
                f"- 实际测量：{family['measured_information_cn']}",
                f"- 与模型比较：{family['model_comparison_cn']}",
                f"- 当前用途：{family['current_use_cn']}",
                f"- 仍有限制：{family['limitations_cn']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 当前必须遵守的处理方式",
            "",
            *[f"- {rule}" for rule in sources["usage_rules_cn"]],
            "",
        ]
    )
    (output / "P418_实验观测量与模型对应_CN.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
