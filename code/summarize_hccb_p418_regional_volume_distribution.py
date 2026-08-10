#!/usr/bin/env python3
"""Summarize regional finite-volume sizes used by the P418 physics loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def volume_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("cell volumes must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("cell volumes must be finite and positive")
    return {
        "region_count": int(len(values)),
        "minimum_m3": float(values.min()),
        "p05_m3": float(np.quantile(values, 0.05)),
        "median_m3": float(np.median(values)),
        "p95_m3": float(np.quantile(values, 0.95)),
        "maximum_m3": float(values.max()),
        "maximum_over_minimum": float(values.max() / values.min()),
        "coefficient_of_variation": float(values.std() / values.mean()),
        "total_volume_m3": float(values.sum()),
    }


def build_summary(geometry: Path) -> dict[str, object]:
    with np.load(geometry, allow_pickle=False) as loaded:
        fluid = volume_summary(loaded["fluid_cell_volume_m3"])
        solid = volume_summary(loaded["solid_cell_volume_m3"])
    return {
        "status": "p418_regional_volume_distribution_summarized",
        "geometry": str(geometry),
        "fluid": fluid,
        "solid": solid,
        "scientific_conclusion_cn": (
            "区域节点代表的物理体积相差多个数量级，能量方程误差必须按区域体积积分；"
            "不能把每个区域节点简单视为相同体积。"
        ),
        "physical_parameters_added": [],
    }


def write_chinese(path: Path, payload: dict[str, object]) -> None:
    fluid = payload["fluid"]
    solid = payload["solid"]
    text = f"""# P418区域网格体积分布

- 流体区域数：{fluid['region_count']}
- 流体最大/最小体积：{fluid['maximum_over_minimum']:.6g}
- 颗粒区域数：{solid['region_count']}
- 颗粒最大/最小体积：{solid['maximum_over_minimum']:.6g}
- 流体体积变异系数：{fluid['coefficient_of_variation']:.6g}
- 颗粒体积变异系数：{solid['coefficient_of_variation']:.6g}

{payload['scientific_conclusion_cn']}

该统计只描述区域有限体积网格，没有增加材料参数或运行参数。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chinese-summary", type=Path, required=True)
    args = parser.parse_args()
    payload = build_summary(args.geometry.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_chinese(args.chinese_summary, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
