#!/usr/bin/env python3
"""Report which heat-flow regimes occur in every P418 data split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from summarize_hccb_p418_thermal_regime_model_errors import (
    load_physical_conditions,
)


ROLES = ("train", "validation", "test", "unused")


def count_values(
    identifiers: list[str],
    physical: dict[str, dict[str, object]],
    key: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for identifier in identifiers:
        if identifier not in physical:
            raise ValueError(f"physical classification is missing for {identifier}")
        value = str(physical[identifier][key])
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def summarize_split_file(
    split_file: Path,
    physical: dict[str, dict[str, object]],
    *,
    allow_partial_physical_coverage: bool = False,
) -> list[dict[str, object]]:
    payload = json.loads(split_file.read_text(encoding="utf-8"))
    splits = payload.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise ValueError(f"no splits in {split_file}")
    rows: list[dict[str, object]] = []
    for split_name, split in splits.items():
        if not isinstance(split, dict):
            raise ValueError(f"invalid split {split_name}")
        for role in ROLES:
            identifiers = [str(value) for value in split.get(role, [])]
            if not identifiers:
                continue
            unknown_identifiers = [
                identifier for identifier in identifiers if identifier not in physical
            ]
            if unknown_identifiers and not allow_partial_physical_coverage:
                raise ValueError(
                    "physical classification is missing for "
                    + ", ".join(unknown_identifiers)
                )
            known_identifiers = [
                identifier for identifier in identifiers if identifier in physical
            ]
            wall_counts = count_values(
                known_identifiers, physical, "cooling_wall_heat_direction"
            )
            solid_counts = count_values(
                known_identifiers, physical, "solid_temperature_relation"
            )
            rows.append(
                {
                    "split_file": split_file.name,
                    "split": str(split_name),
                    "role": role,
                    "case_count": len(identifiers),
                    "known_case_count": len(known_identifiers),
                    "unknown_case_count": len(unknown_identifiers),
                    "coverage_complete": not unknown_identifiers,
                    "wall_to_fluid_count": wall_counts.get("wall_to_fluid", 0),
                    "fluid_to_wall_count": wall_counts.get("fluid_to_wall", 0),
                    "zero_wall_heat_count": wall_counts.get("zero", 0),
                    "solid_maximum_above_wall_count": solid_counts.get(
                        "solid_maximum_above_wall", 0
                    ),
                    "solid_maximum_at_or_below_wall_count": solid_counts.get(
                        "solid_maximum_at_or_below_wall", 0
                    ),
                    "contains_both_nonzero_wall_heat_directions": (
                        wall_counts.get("wall_to_fluid", 0) > 0
                        and wall_counts.get("fluid_to_wall", 0) > 0
                    ),
                    "condition_ids": ";".join(identifiers),
                    "unknown_condition_ids": ";".join(unknown_identifiers),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no split coverage rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_chinese_summary(
    path: Path,
    rows: list[dict[str, object]],
    *,
    coverage_complete: bool,
) -> None:
    role_names = {
        "train": "训练",
        "validation": "验证",
        "test": "测试",
        "unused": "未使用",
    }
    lines = [
        "# P418稳态数据划分中的换热状态",
        "",
        "这份结果只检查每套数据划分包含哪些换热状态，不比较模型精度。",
        "`壁面向流体`表示635 K壁面向球床和氦气供热；"
        "`流体向壁面`表示较热的球床和氦气向壁面放热。",
        "",
    ]
    if coverage_complete:
        lines.append("60个稳态工况的换热方向均已读取，下面是完整统计。")
    else:
        lines.append(
            "OpenFOAM矩阵仍在计算，下面只统计已经生成正式结果的工况；"
            "尚未完成的工况不判断换热方向。"
        )
    lines.extend(
        [
            "",
            "| 工况划分 | 数据组 | 已完成/总数 | 壁面向流体 | 流体向壁面 | 当前已含两种方向 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {split} | {role} | {known}/{total} | {wall} | {fluid} | {both} |".format(
                split=row["split"],
                role=role_names.get(str(row["role"]), str(row["role"])),
                known=row["known_case_count"],
                total=row["case_count"],
                wall=row["wall_to_fluid_count"],
                fluid=row["fluid_to_wall_count"],
                both="是" if row["contains_both_nonzero_wall_heat_directions"] else "否",
            )
        )
    lines.extend(
        [
            "",
            "温度外推划分有意用较低入口温度训练、较高入口温度验证和测试，"
            "因此不同数据组可能分别落在相反的换热方向；这正是该外推检验要考察的物理变化。",
            "其余数据组若当前只出现一种方向，必须结合未完成工况数判断，不能提前下结论。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-csv", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-partial-physical-coverage",
        action="store_true",
        help=(
            "Report completed and unfinished cases separately while the matrix is "
            "still running. The default remains strict for final paper results."
        ),
    )
    args = parser.parse_args()

    physical = load_physical_conditions(args.physical_csv)
    rows = summarize_split_file(
        args.split_file,
        physical,
        allow_partial_physical_coverage=args.allow_partial_physical_coverage,
    )
    unknown_case_count = sum(int(row["unknown_case_count"]) for row in rows)
    coverage_complete = unknown_case_count == 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "thermal_regime_split_coverage.csv"
    json_path = args.output_dir / "thermal_regime_split_coverage.json"
    write_csv(csv_path, rows)
    write_chinese_summary(
        args.output_dir / "P418_换热状态与数据划分_CN.md",
        rows,
        coverage_complete=coverage_complete,
    )
    json_path.write_text(
        json.dumps(
            {
                "status": (
                    "thermal_regime_split_coverage_complete"
                    if coverage_complete
                    else "thermal_regime_split_coverage_in_progress"
                ),
                "physical_classification_source": str(args.physical_csv.resolve()),
                "split_source": str(args.split_file.resolve()),
                "row_count": len(rows),
                "coverage_complete": coverage_complete,
                "unknown_case_references_across_rows": unknown_case_count,
                "rows": rows,
                "interpretation": (
                    "The table reports physical heat-flow coverage only. Unknown "
                    "condition identifiers are unfinished simulations when partial "
                    "coverage is explicitly enabled. A one-sided extrapolation split "
                    "is retained as designed and is not treated as an error."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
