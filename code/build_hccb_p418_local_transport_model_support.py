#!/usr/bin/env python3
"""Build the current P418 local-flow and interface-heat model support table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHYSICS = (
    ROOT / "results/hccb_p418_sourceflow_partial_physics/completed_case_physics.csv"
)
DEFAULT_PRESSURE = (
    ROOT
    / "results/hccb_p418_sourceflow_partial_pressure_correlation/pressure_correlation.csv"
)
DEFAULT_HEAT = (
    ROOT
    / "results/hccb_p418_sourceflow_partial_dimensionless_heat_transfer_with_flux"
    / "dimensionless_heat_transfer.csv"
)
DEFAULT_BOUNDARY = (
    ROOT / "results/hccb_p418_sourceflow_partial_boundary_heat/summary.json"
)
DEFAULT_CONTRACT = ROOT / "parameters/hccb_p418_local_transport_model_contract.json"
DEFAULT_PARAMETERS = ROOT / "parameters/literature_parameter_manifest.csv"
DEFAULT_OUTPUT = ROOT / "results/hccb_p418_local_transport_model_support"


OUTPUT_COLUMNS = (
    "condition_id",
    "inlet_velocity_m_s",
    "inlet_temperature_K",
    "solid_heat_source_W_m3",
    "resolved_pressure_drop_Pa",
    "P420_P421_pressure_drop_Pa",
    "pressure_relation_absolute_difference_percent",
    "particle_Reynolds_axial_throughflow",
    "particle_Reynolds_local_velocity_magnitude",
    "local_to_throughflow_Re_ratio",
    "local_Prandtl_number",
    "local_Peclet_number",
    "phase_temperature_difference_K",
    "OpenFOAM_interface_heat_into_fluid_W",
    "OpenFOAM_interface_flux_Nusselt_number",
    "P419_aggregate_Nusselt_number",
    "solid_wall_heat_into_solid_W",
    "generated_power_W",
    "interface_heat_over_generated_power",
    "solid_wall_heat_over_generated_power",
    "interface_pair_difference_W",
    "solid_balance_relative",
    "cooling_wall_heat_direction",
)

OPTIONAL_NUMERIC_COLUMNS = {
    "P419_aggregate_Nusselt_number",
}


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    records = {str(row["condition_id"]): row for row in rows}
    if len(records) != len(rows):
        raise ValueError(f"duplicate condition_id in {path}")
    return records


def parameter_ids(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return {str(row["parameter_id"]) for row in csv.DictReader(stream)}


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def numeric_range(rows: list[dict[str, object]], key: str) -> list[float]:
    values = np.asarray(
        [
            float(row[key])
            for row in rows
            if row[key] is not None and np.isfinite(float(row[key]))
        ],
        dtype=np.float64,
    )
    if values.size == 0:
        raise ValueError(f"no finite values in {key}")
    return [float(values.min()), float(values.max())]


def render_cn(summary: dict[str, object]) -> str:
    r = summary["ranges"]
    return "\n".join(
        [
            "# P418局部三维流动与换热模型数据说明",
            "",
            f"当前已纳入 `{summary['case_count']}` 组正常完成的三维稳态工况，"
            f"完整计划为 `{summary['planned_case_count']}` 组。当前数据用于确定模型必须保存哪些物理量，"
            "不用于宣布正式模型精度。",
            "",
            "## 当前三维结果给出的范围",
            "",
            f"- 入口速度：`{r['inlet_velocity_m_s'][0]:.3g}--{r['inlet_velocity_m_s'][1]:.3g} m/s`；",
            f"- 入口温度：`{r['inlet_temperature_K'][0]:.0f}--{r['inlet_temperature_K'][1]:.0f} K`；",
            f"- 颗粒发热率：`{r['solid_heat_source_W_m3'][0] / 1e6:.2f}--{r['solid_heat_source_W_m3'][1] / 1e6:.2f} MW/m3`；",
            f"- 轴向净流量颗粒雷诺数：`{r['particle_Reynolds_axial_throughflow'][0]:.3f}--{r['particle_Reynolds_axial_throughflow'][1]:.3f}`；",
            f"- 局部速度模量颗粒雷诺数：`{r['particle_Reynolds_local_velocity_magnitude'][0]:.3f}--{r['particle_Reynolds_local_velocity_magnitude'][1]:.3f}`；",
            f"- 局部/轴向雷诺数比：`{r['local_to_throughflow_Re_ratio'][0]:.3f}--{r['local_to_throughflow_Re_ratio'][1]:.3f}`；",
            f"- 流固界面热流努塞尔数：`{r['OpenFOAM_interface_flux_Nusselt_number'][0]:.3f}--{r['OpenFOAM_interface_flux_Nusselt_number'][1]:.3f}`；",
            f"- P419整床平均定义的努塞尔数：`{r['P419_aggregate_Nusselt_number'][0]:.3f}--{r['P419_aggregate_Nusselt_number'][1]:.3f}`。",
            "",
            "## 对融合模型的直接要求",
            "",
            "1. 节点输入保留三维坐标、流固区域、边界邻接比例、局部速度矢量、压力和温度；",
            "2. 连接面保留面积矢量、面质量流、面能量流以及流固界面类型；",
            "3. 冷却壁热量、流固界面热量和颗粒体积发热量分别保存；",
            "4. PINN部分直接检查质量、动量、流体焓、颗粒导热和界面热量关系；",
            "5. Transformer负责完整时间过程，扩散模型只修正受物理方程限制的温度或低维残差；",
            "6. P417/P419整床平均关系仅作量级比较，不强迫局部靠壁三维场服从该关系。",
            "",
            "## 本文件的适用范围",
            "",
            "本文件只汇总早期14组稳态结果，用来确定局部流动与换热模型需要保留的物理量，"
            "不代表项目当前完成进度，也不用于给出最终模型排名。完整矩阵、瞬态计算和模型训练的"
            "最新状态以 `CURRENT_STATUS_CN.md` 和正式结果文件为准。",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physics-csv", type=Path, default=DEFAULT_PHYSICS)
    parser.add_argument("--pressure-csv", type=Path, default=DEFAULT_PRESSURE)
    parser.add_argument("--heat-csv", type=Path, default=DEFAULT_HEAT)
    parser.add_argument("--boundary-summary", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--parameter-manifest", type=Path, default=DEFAULT_PARAMETERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    physics = read_rows(args.physics_csv.resolve())
    pressure = read_rows(args.pressure_csv.resolve())
    heat = read_rows(args.heat_csv.resolve())
    boundary_payload = json.loads(args.boundary_summary.resolve().read_text(encoding="utf-8"))
    boundary = {str(row["condition_id"]): row for row in boundary_payload["cases"]}
    condition_ids = set(physics)
    same_cases = condition_ids == set(pressure) == set(heat) == set(boundary)
    if not same_cases:
        raise ValueError("physics, pressure, heat and boundary summaries use different cases")

    contract = json.loads(args.contract.resolve().read_text(encoding="utf-8"))
    available_ids = parameter_ids(args.parameter_manifest.resolve())
    required_ids = set(map(str, contract["physical_source_ids"]))
    source_ids_present = required_ids <= available_ids
    if not source_ids_present:
        raise ValueError(f"missing literature parameter ids: {sorted(required_ids-available_ids)}")
    if contract.get("new_physical_parameters") != []:
        raise ValueError("local transport model contract introduces physical parameters")

    rows: list[dict[str, object]] = []
    for condition_id in sorted(condition_ids):
        p = physics[condition_id]
        d = pressure[condition_id]
        h = heat[condition_id]
        b = boundary[condition_id]
        re_axial = f(h, "reynolds_particle_axial_throughflow")
        re_local = f(h, "reynolds_particle_local_magnitude_volume_average")
        p419_nusselt = f(h, "nusselt_from_resolved_field_P419")
        if not np.isfinite(p419_nusselt):
            p419_nusselt = None
        row = {
            "condition_id": condition_id,
            "inlet_velocity_m_s": f(p, "inlet_velocity_m_s"),
            "inlet_temperature_K": f(p, "inlet_temperature_K"),
            "solid_heat_source_W_m3": f(h, "solid_heat_source_W_m3"),
            "resolved_pressure_drop_Pa": f(d, "resolved_pressure_drop_Pa"),
            "P420_P421_pressure_drop_Pa": f(d, "P420_P421_pressure_drop_Pa"),
            "pressure_relation_absolute_difference_percent": f(
                d, "absolute_difference_percent"
            ),
            "particle_Reynolds_axial_throughflow": re_axial,
            "particle_Reynolds_local_velocity_magnitude": re_local,
            "local_to_throughflow_Re_ratio": re_local / re_axial,
            "local_Prandtl_number": f(h, "prandtl_local_volume_average"),
            "local_Peclet_number": f(
                h, "peclet_local_magnitude_volume_average"
            ),
            "phase_temperature_difference_K": f(
                h, "phase_temperature_difference_K"
            ),
            "OpenFOAM_interface_heat_into_fluid_W": f(
                h, "openfoam_interphase_heat_into_fluid_W"
            ),
            "OpenFOAM_interface_flux_Nusselt_number": f(
                h, "nusselt_from_openfoam_interphase_flux"
            ),
            "P419_aggregate_Nusselt_number": p419_nusselt,
            "solid_wall_heat_into_solid_W": f(
                h, "openfoam_solid_wall_heat_into_solid_W"
            ),
            "generated_power_W": float(b["generated_power_W"]),
            "interface_heat_over_generated_power": f(
                h, "openfoam_interphase_heat_over_generated_power"
            ),
            "solid_wall_heat_over_generated_power": f(
                h, "openfoam_solid_wall_heat_over_generated_power"
            ),
            "interface_pair_difference_W": float(b["interface_pair_difference_W"]),
            "solid_balance_relative": float(b["solid_balance_relative"]),
            "cooling_wall_heat_direction": p["cooling_wall_heat_direction"],
        }
        rows.append(row)

    numeric_columns = OUTPUT_COLUMNS[1:-1]
    ranges = {key: numeric_range(rows, key) for key in numeric_columns}
    finite_counts = {
        key: int(
            sum(
                bool(
                    row[key] is not None
                    and np.isfinite(float(row[key]))
                )
                for row in rows
            )
        )
        for key in numeric_columns
    }
    required_numeric_columns = [
        key for key in numeric_columns if key not in OPTIONAL_NUMERIC_COLUMNS
    ]
    checks = {
        "all_four_sources_use_the_same_cases": same_cases,
        "all_literature_parameter_ids_are_registered": source_ids_present,
        "contract_introduces_no_physical_parameters": (
            contract.get("new_physical_parameters") == []
        ),
        "all_numeric_values_are_finite": all(
            np.all(np.isfinite([float(row[key]) for row in rows]))
            for key in required_numeric_columns
        ),
        "optional_P419_values_are_used_only_when_phase_temperature_is_positive": all(
            (row["P419_aggregate_Nusselt_number"] is not None)
            == (float(row["phase_temperature_difference_K"]) > 0.0)
            for row in rows
        ),
        "interface_and_solid_balance_are_consistent": (
            max(abs(float(row["interface_pair_difference_W"])) for row in rows)
            <= 1.0e-4
            and max(abs(float(row["solid_balance_relative"])) for row in rows)
            <= 1.0e-6
        ),
        "P419_is_comparison_only": any(
            "P417/P419" in rule and "不作为" in rule for rule in contract["rules"]
        ),
    }
    summary = {
        "status": (
            "p418_local_transport_model_support_ready"
            if all(checks.values())
            else "failed"
        ),
        "case_count": len(rows),
        "planned_case_count": int(contract["current_evidence"]["planned_case_count"]),
        "support_is_partial": len(rows)
        < int(contract["current_evidence"]["planned_case_count"]),
        "ranges": ranges,
        "finite_value_counts": finite_counts,
        "checks": checks,
        "physical_source_ids": sorted(required_ids),
        "model_contract": str(args.contract.resolve()),
        "new_physical_parameters": [],
    }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "local_transport_model_support.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "P418_局部三维模型数据说明_CN.md").write_text(
        render_cn(summary), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
