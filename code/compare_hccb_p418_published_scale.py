#!/usr/bin/env python3
"""Compare a local P418 crop with the published full-domain reference scale."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


def read_parameters(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["parameter_id"]: row for row in csv.DictReader(handle)}


def parse_published_reference(rows: dict[str, dict[str, str]]) -> dict[str, float]:
    values = {
        key: float(value)
        for key, value in re.findall(r"([A-Za-z_]+)=([0-9.]+)", rows["P391"]["value"])
    }
    geometry = [float(value) for value in re.findall(r"([0-9.]+)dp", rows["P390"]["value"])]
    if len(geometry) < 3:
        raise ValueError("P390 does not contain three packed-region dimensions")
    return {
        "inlet_velocity_m_s": values["u_in"],
        "inlet_temperature_K": values["T_in"],
        "pressure_drop_Pa": values["deltaP"],
        "solid_maximum_K": values["Tmax"],
        "solid_heat_source_W_m3": float(rows["P053"]["value"]) * 1.0e6,
        "particle_diameter_m": float(rows["P048"]["value"]) * 1.0e-3,
        "packed_width_x_dp": geometry[0],
        "packed_width_y_dp": geometry[1],
        "packed_flow_length_dp": geometry[2],
    }


def local_geometry(sample_path: Path) -> dict[str, float]:
    with np.load(sample_path, allow_pickle=False) as sample:
        centres = np.vstack(
            [sample["fluid_cell_centroid_m"], sample["solid_cell_centroid_m"]]
        )
        boundary_centres = sample["fluid_boundary_face_centroid_m"]
        boundary_patch = sample["fluid_boundary_face_patch"]
    inlet_z = float(np.mean(boundary_centres[boundary_patch == 0, 2]))
    outlet_z = float(np.mean(boundary_centres[boundary_patch == 1, 2]))
    return {
        "width_x_m": float(np.ptp(centres[:, 0])),
        "width_y_m": float(np.ptp(centres[:, 1])),
        "flow_length_m": outlet_z - inlet_z,
        "inlet_z_m": inlet_z,
        "outlet_z_m": outlet_z,
    }


def compare(
    parameters_path: Path,
    case_summary_path: Path,
    sample_path: Path,
) -> dict[str, object]:
    rows = read_parameters(parameters_path)
    published = parse_published_reference(rows)
    case = json.loads(case_summary_path.read_text(encoding="utf-8"))
    geometry = local_geometry(sample_path)
    diameter = published["particle_diameter_m"]

    local = {
        "inlet_velocity_m_s": float(case["physical_conditions"]["inlet_velocity_m_s"]),
        "inlet_temperature_K": float(case["physical_conditions"]["inlet_temperature_K"]),
        "solid_heat_source_W_m3": float(
            case["physical_conditions"]["solid_heat_source_W_m3"]
        ),
        "pressure_drop_Pa": float(case["flow"]["pressure_drop_Pa"]),
        "solid_maximum_K": float(case["temperature"]["solid_maximum_K"]),
        "width_x_dp": geometry["width_x_m"] / diameter,
        "width_y_dp": geometry["width_y_m"] / diameter,
        "flow_length_dp": geometry["flow_length_m"] / diameter,
    }

    operating_matches = {
        name: bool(np.isclose(local[name], published[name], rtol=0.0, atol=tolerance))
        for name, tolerance in {
            "inlet_velocity_m_s": 1.0e-12,
            "inlet_temperature_K": 1.0e-9,
            "solid_heat_source_W_m3": 1.0e-6,
        }.items()
    }
    dimension_ratios = {
        "width_x_local_to_published": local["width_x_dp"]
        / published["packed_width_x_dp"],
        "width_y_local_to_published": local["width_y_dp"]
        / published["packed_width_y_dp"],
        "flow_length_local_to_published_bed": local["flow_length_dp"]
        / published["packed_flow_length_dp"],
    }
    same_geometry = all(np.isclose(value, 1.0, rtol=0.01) for value in dimension_ratios.values())
    local_gradient = local["pressure_drop_Pa"] / geometry["flow_length_m"]
    published_gradient = published["pressure_drop_Pa"] / (
        published["packed_flow_length_dp"] * diameter
    )

    return {
        "status": "direct_numeric_validation" if same_geometry else "same_operating_point_different_geometry",
        "parameter_ids": ["P048", "P053", "P390", "P391"],
        "published_source": rows["P391"]["source_title"],
        "published_doi": rows["P391"]["source_url_or_doi"],
        "operating_conditions_match": operating_matches,
        "published_reference": published,
        "local_crop": local,
        "local_geometry_m": geometry,
        "dimension_ratios": dimension_ratios,
        "pressure_gradient_Pa_m": {
            "published_over_10dp_bed": published_gradient,
            "local_over_crop": local_gradient,
            "local_to_published_ratio": local_gradient / published_gradient,
        },
        "temperature_rise_above_635K_wall_K": {
            "published": published["solid_maximum_K"] - 635.0,
            "local": local["solid_maximum_K"] - 635.0,
        },
        "direct_pressure_or_temperature_error_is_valid": same_geometry,
        "interpretation": (
            "The operating point matches the published u=0.20 m/s, Tin=700 K and "
            "q'''=6.85 MW/m3 case, but the present pore-resolved crop is much smaller "
            "than the published 12.5dp x 12.5dp x 10dp packed region. Pressure drop "
            "and maximum temperature therefore cannot be treated as direct replication errors."
        ),
    }


def write_outputs(payload: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pub = payload["published_reference"]
    local = payload["local_crop"]
    ratios = payload["dimension_ratios"]
    gradients = payload["pressure_gradient_Pa_m"]
    rows = [
        ("inlet velocity", pub["inlet_velocity_m_s"], local["inlet_velocity_m_s"], "m/s"),
        ("inlet temperature", pub["inlet_temperature_K"], local["inlet_temperature_K"], "K"),
        ("solid heat source", pub["solid_heat_source_W_m3"], local["solid_heat_source_W_m3"], "W/m3"),
        ("pressure drop", pub["pressure_drop_Pa"], local["pressure_drop_Pa"], "Pa"),
        ("solid maximum", pub["solid_maximum_K"], local["solid_maximum_K"], "K"),
        ("width x", pub["packed_width_x_dp"], local["width_x_dp"], "dp"),
        ("width y", pub["packed_width_y_dp"], local["width_y_dp"], "dp"),
        ("packed/crop flow length", pub["packed_flow_length_dp"], local["flow_length_dp"], "dp"),
    ]
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", "published_full_region", "current_local_crop", "unit"])
        writer.writerows(rows)

    text = f"""# P418中心工况与原文数值的正确比较方式

当前局部模型与原文中心工况使用相同的入口速度、入口温度和颗粒发热率，但几何尺寸不同，因此不能把压降和最高温度直接写成复现误差。

| 数量 | 原文完整球床 | 当前精细局部域 |
|---|---:|---:|
| 横向尺寸 x | {pub['packed_width_x_dp']:.2f} dp | {local['width_x_dp']:.3f} dp |
| 横向尺寸 y | {pub['packed_width_y_dp']:.2f} dp | {local['width_y_dp']:.3f} dp |
| 轴向球床/局部域长度 | {pub['packed_flow_length_dp']:.2f} dp | {local['flow_length_dp']:.3f} dp |
| 压降 | {pub['pressure_drop_Pa']:.1f} Pa | {local['pressure_drop_Pa']:.3f} Pa |
| 最高颗粒温度 | {pub['solid_maximum_K']:.1f} K | {local['solid_maximum_K']:.3f} K |

局部域三个方向分别只有原文尺寸的 {ratios['width_x_local_to_published']:.1%}、{ratios['width_y_local_to_published']:.1%} 和 {ratios['flow_length_local_to_published_bed']:.1%}。当前压降梯度为 {gradients['local_over_crop']:.1f} Pa/m，原文按 10dp 球床长度换算为 {gradients['published_over_10dp_bed']:.1f} Pa/m；这个差异仍包含截面尺寸、壁面效应和颗粒装填不同，不能当作单一模型误差。

论文中的正确说法是：当前60工况采用原文运行参数和边界类型，在局部致密球床上研究人工智能模型的三维场预测；它不是原文完整几何的直接重算。若要验证87 Pa和897 K，必须另建原文12.5dp × 12.5dp × 10dp球床及其入口、出口延长段。
"""
    (output_dir / "README_CN.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--case-summary", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = compare(args.parameters, args.case_summary, args.sample)
    write_outputs(payload, args.output_dir)


if __name__ == "__main__":
    main()
