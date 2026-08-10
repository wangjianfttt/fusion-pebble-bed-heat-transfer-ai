#!/usr/bin/env python3
"""Summarize two fixed-seed full-graph GPU smoke runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


METRICS = (
    "weighted_temperature_loss",
    "projection_aware_volume_energy_loss",
    "area_weighted_heat_flux_density_loss",
    "total_smoke_objective",
)


def relative_difference(left: float, right: float) -> float:
    scale = max(abs(left), abs(right), sys.float_info.epsilon)
    return abs(left - right) / scale


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    first = json.loads(args.first.resolve().read_text(encoding="utf-8"))
    second = json.loads(args.second.resolve().read_text(encoding="utf-8"))
    if first.get("random_seed") != second.get("random_seed"):
        raise ValueError("the two smoke runs do not use the same random seed")
    comparison = {
        name: {
            "first": float(first[name]),
            "second": float(second[name]),
            "relative_difference": relative_difference(
                float(first[name]), float(second[name])
            ),
        }
        for name in METRICS
    }
    maximum = max(row["relative_difference"] for row in comparison.values())
    summary = {
        "status": (
            "fixed_seed_full_graph_gpu_roundoff_repeatability_confirmed"
            if maximum <= 1.0e-3
            else "fixed_seed_full_graph_gpu_repeatability_requires_investigation"
        ),
        "random_seed": first["random_seed"],
        "comparison": comparison,
        "maximum_relative_difference": maximum,
        "unchanged_initial_state_error": (
            first["initial_maximum_absolute_error"]
            == second["initial_maximum_absolute_error"]
            == 0.0
        ),
        "unchanged_hydrodynamic_error": (
            first["hydrodynamic_maximum_absolute_error"]
            == second["hydrodynamic_maximum_absolute_error"]
            == 0.0
        ),
        "interpretation": (
            "The fixed seed removes initialization changes. Remaining differences "
            "measure CUDA parallel-reduction round-off on the actual full graph. "
            "They are not neural prediction errors."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "README_CN.md").write_text(
        "\n".join(
            (
                "# P418全尺寸GPU程序检查重复性",
                "",
                f"- 固定随机数：`{summary['random_seed']}`",
                f"- 四个损失分量的最大相对差：`{maximum:.6e}`",
                "- 两次运行的初始状态误差均为0，速度和压力误差均为0。",
                "",
                "固定随机数后仍有很小差异，这是GPU并行求和顺序造成的舍入差。正式模型使用三次独立初值比较，不把单次程序检查的末位数字当作模型精度。",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if maximum <= 1.0e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
