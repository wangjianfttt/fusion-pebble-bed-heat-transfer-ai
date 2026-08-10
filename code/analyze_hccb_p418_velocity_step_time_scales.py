#!/usr/bin/env python3
"""Quantify flow and thermal time scales for the P418 velocity-step cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


VALUE_COLUMN = "采用值或关系式"


def physical_source_rows(source: Path) -> dict[str, dict[str, str]]:
    with source.open(newline="", encoding="utf-8") as handle:
        return {row["parameter_id"]: row for row in csv.DictReader(handle)}


def p418_inlet_velocities(source: Path) -> list[float]:
    rows = physical_source_rows(source)
    if "P418" not in rows:
        raise ValueError("P418 is missing from the physical-parameter table")
    value = rows["P418"][VALUE_COLUMN]
    match = re.search(r"u_in=([0-9.,]+)\s*m/s", value)
    if not match:
        raise ValueError(f"cannot parse the P418 velocity set: {value}")
    velocities = [float(item) for item in match.group(1).split(",")]
    if not velocities or any(value <= 0.0 for value in velocities):
        raise ValueError(f"invalid P418 velocity set: {velocities}")
    return velocities


def p418_source_geometry(source: Path) -> dict[str, float]:
    """Return literature-reported particle and streamwise geometry dimensions."""
    rows = physical_source_rows(source)
    required = {"P048", "P427"}
    missing = required.difference(rows)
    if missing:
        raise ValueError(f"physical-parameter table is missing {sorted(missing)}")
    diameter_mm = float(rows["P048"][VALUE_COLUMN])
    geometry = rows["P427"][VALUE_COLUMN]
    bed_match = re.search(r"bed=([^;]+)", geometry)
    inlet_match = re.search(r"inlet_extension=([0-9.]+)dp", geometry)
    outlet_match = re.search(r"outlet_extension=([0-9.]+)dp", geometry)
    if not (bed_match and inlet_match and outlet_match):
        raise ValueError(f"cannot parse the P427 geometry: {geometry}")
    bed_dimensions_dp = [
        float(value) for value in re.findall(r"([0-9.]+)dp", bed_match.group(1))
    ]
    if len(bed_dimensions_dp) != 3:
        raise ValueError(f"cannot parse three P427 bed dimensions: {geometry}")
    # P427 describes the source flow direction as the final, 10 dp bed dimension.
    bed_flow_dp = bed_dimensions_dp[-1]
    inlet_dp = float(inlet_match.group(1))
    outlet_dp = float(outlet_match.group(1))
    diameter_m = diameter_mm * 1.0e-3
    return {
        "particle_diameter_m": diameter_m,
        "packed_bed_flow_length_dp": bed_flow_dp,
        "packed_bed_flow_length_m": bed_flow_dp * diameter_m,
        "full_source_domain_flow_length_dp": inlet_dp + bed_flow_dp + outlet_dp,
        "full_source_domain_flow_length_m": (inlet_dp + bed_flow_dp + outlet_dp)
        * diameter_m,
    }


def sourceflow_velocity_mapping(
    input_summary_path: Path,
    source_velocities_m_s: list[float],
) -> tuple[list[float], float]:
    """Read the area-preserving pore-boundary velocities used by the formal cases."""
    payload = json.loads(input_summary_path.read_text(encoding="utf-8"))
    if payload.get("status") != "hccb_p418_60_actual_case_inputs_verified":
        raise ValueError(f"unexpected source-flow input summary: {input_summary_path}")
    rows = payload.get("cases") or payload.get("conditions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source-flow input summary contains no cases")

    mapping: dict[float, tuple[float, float]] = {}
    for row in rows:
        source_velocity = float(row["inlet_velocity_m_s"])
        pore_velocity = float(row["pore_opening_boundary_velocity_m_s"])
        open_fraction = float(row["inlet_open_area_fraction"])
        if not bool(row.get("source_channel_volume_flow_preserved", False)):
            raise ValueError("formal case does not preserve the source-channel volume flow")
        if not (0.0 < open_fraction <= 1.0 and pore_velocity > 0.0):
            raise ValueError("invalid pore-boundary velocity mapping")
        if not math.isclose(
            pore_velocity * open_fraction,
            source_velocity,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ):
            raise ValueError("pore-boundary velocity does not preserve source-channel flow")
        previous = mapping.setdefault(source_velocity, (pore_velocity, open_fraction))
        if not (
            math.isclose(previous[0], pore_velocity, rel_tol=0.0, abs_tol=1.0e-12)
            and math.isclose(previous[1], open_fraction, rel_tol=0.0, abs_tol=1.0e-12)
        ):
            raise ValueError("one source velocity maps to inconsistent pore-boundary values")

    missing = [value for value in source_velocities_m_s if value not in mapping]
    if missing:
        raise ValueError(f"source-flow input summary misses velocities: {missing}")
    open_fractions = [mapping[value][1] for value in source_velocities_m_s]
    if max(open_fractions) - min(open_fractions) > 1.0e-12:
        raise ValueError("formal matrix does not use one fixed inlet open-area fraction")
    return [mapping[value][0] for value in source_velocities_m_s], open_fractions[0]


def cell_outflow_turnover_rate(
    volume_m3: np.ndarray,
    density_kg_m3: np.ndarray,
    internal_owner: np.ndarray,
    internal_neighbour: np.ndarray,
    internal_mass_flow_kg_s: np.ndarray,
    boundary_owner: np.ndarray,
    boundary_mass_flow_kg_s: np.ndarray,
) -> np.ndarray:
    """Return positive mass outflow divided by cell mass for every fluid cell."""
    volume = np.asarray(volume_m3, dtype=np.float64)
    density = np.asarray(density_kg_m3, dtype=np.float64)
    if volume.shape != density.shape or np.any(volume <= 0.0) or np.any(density <= 0.0):
        raise ValueError("fluid volume and density must be positive arrays of equal shape")
    outflow = np.zeros(volume.size, dtype=np.float64)
    flux = np.asarray(internal_mass_flow_kg_s, dtype=np.float64)
    np.add.at(outflow, np.asarray(internal_owner, dtype=np.int64), np.maximum(flux, 0.0))
    np.add.at(
        outflow,
        np.asarray(internal_neighbour, dtype=np.int64),
        np.maximum(-flux, 0.0),
    )
    boundary_flux = np.asarray(boundary_mass_flow_kg_s, dtype=np.float64)
    np.add.at(
        outflow,
        np.asarray(boundary_owner, dtype=np.int64),
        np.maximum(boundary_flux, 0.0),
    )
    return outflow / (density * volume)


def analyze(
    topology_path: Path,
    field_path: Path,
    parameter_source: Path,
    particle_scale_path: Path,
    flow_axis: int,
    thermal_time_step_s: float,
    sourceflow_input_summary_path: Path | None = None,
) -> dict[str, object]:
    topology = np.load(topology_path, allow_pickle=False)
    field = np.load(field_path, allow_pickle=False)
    centroids = topology["fluid_cell_centroid_m"]
    if flow_axis not in (0, 1, 2):
        raise ValueError("flow axis must be 0, 1 or 2")
    flow_span = float(np.ptp(centroids[:, flow_axis]))
    source_velocities = p418_inlet_velocities(parameter_source)
    if sourceflow_input_summary_path is None:
        pore_boundary_velocities = source_velocities
        inlet_open_area_fraction = 1.0
        velocity_basis = "source_velocity_assumed_equal_to_pore_boundary_velocity"
    else:
        pore_boundary_velocities, inlet_open_area_fraction = sourceflow_velocity_mapping(
            sourceflow_input_summary_path,
            source_velocities,
        )
        velocity_basis = "source_channel_area_preserving_pore_boundary_velocity"
    geometry = p418_source_geometry(parameter_source)
    local_crossing = [flow_span / velocity for velocity in pore_boundary_velocities]
    bed_crossing = [
        geometry["packed_bed_flow_length_m"] / velocity for velocity in source_velocities
    ]
    full_domain_crossing = [
        geometry["full_source_domain_flow_length_m"] / velocity
        for velocity in source_velocities
    ]

    turnover = cell_outflow_turnover_rate(
        topology["fluid_cell_volume_m3"],
        field["fluid_density_kg_m3"],
        topology["fluid_internal_face_owner"],
        topology["fluid_internal_face_neighbour"],
        field["fluid_internal_face_mass_flow_kg_s"],
        topology["fluid_boundary_face_owner"],
        field["fluid_boundary_face_mass_flow_kg_s"],
    )
    positive = turnover[turnover > 0.0]
    if positive.size == 0:
        raise ValueError("the endpoint field contains no positive cell outflow")
    percentile_levels = (50.0, 90.0, 95.0, 99.0, 99.9, 100.0)
    percentiles = {
        f"p{level:g}": float(np.percentile(positive, level))
        for level in percentile_levels
    }
    dt_co05_p95 = 0.5 / percentiles["p95"]
    dt_co10_p95 = 1.0 / percentiles["p95"]

    particle = json.loads(particle_scale_path.read_text(encoding="utf-8"))
    particle_scales = [
        float(row["particle_radial_diffusion_scale_s"]) for row in particle["values"]
    ]
    maximum_local_crossing = max(local_crossing)
    minimum_local_crossing = min(local_crossing)
    minimum_particle_scale = min(particle_scales)
    maximum_particle_scale = max(particle_scales)
    comparison_rows = []
    for source_velocity, pore_velocity, local_time, bed_time, full_time in zip(
        source_velocities,
        pore_boundary_velocities,
        local_crossing,
        bed_crossing,
        full_domain_crossing,
    ):
        comparison_rows.append(
            {
                "source_inlet_channel_velocity_m_s": source_velocity,
                "pore_opening_boundary_velocity_m_s": pore_velocity,
                "resolved_local_crop_crossing_time_s": local_time,
                "published_10dp_bed_crossing_time_s": bed_time,
                "published_30dp_full_domain_crossing_time_s": full_time,
                "particle_conduction_to_10dp_bed_crossing_ratio_min": minimum_particle_scale
                / bed_time,
                "particle_conduction_to_10dp_bed_crossing_ratio_max": maximum_particle_scale
                / bed_time,
            }
        )
    result = {
        "status": "p418_velocity_step_flow_thermal_time_scale_comparison",
        "parameter_ids": sorted(
            set(["P048", "P427", "P418", *particle.get("parameter_ids", [])])
        ),
        "topology_file": str(topology_path),
        "endpoint_field_file": str(field_path),
        "flow_axis_index": flow_axis,
        "resolved_flow_span_m": flow_span,
        "published_source_geometry": geometry,
        "published_inlet_velocities_m_s": source_velocities,
        "pore_opening_boundary_velocities_m_s": pore_boundary_velocities,
        "inlet_open_area_fraction": inlet_open_area_fraction,
        "velocity_basis": velocity_basis,
        "sourceflow_input_summary": (
            str(sourceflow_input_summary_path) if sourceflow_input_summary_path else None
        ),
        "resolved_local_crop_crossing_times_s": local_crossing,
        "published_10dp_bed_crossing_times_s": bed_crossing,
        "published_30dp_full_domain_crossing_times_s": full_domain_crossing,
        # Retain the original names for readers of the earlier result file. They refer
        # only to the resolved local crop, not to the source paper's full geometry.
        "nominal_domain_crossing_times_s": local_crossing,
        "minimum_domain_crossing_time_s": minimum_local_crossing,
        "maximum_domain_crossing_time_s": maximum_local_crossing,
        "time_scale_comparison_by_velocity": comparison_rows,
        "particle_radial_conduction_scale_s": {
            "minimum": min(particle_scales),
            "maximum": max(particle_scales),
        },
        "cell_outflow_turnover_rate_per_s": percentiles,
        "numerical_flow_time_step_indicators_s": {
            "Co_0p5_at_p95_turnover": dt_co05_p95,
            "Co_1p0_at_p95_turnover": dt_co10_p95,
        },
        "thermal_step_time_step_s": thermal_time_step_s,
        "implied_p95_Courant_if_flow_were_active": thermal_time_step_s
        * percentiles["p95"],
        "steps_for_slowest_domain_crossing_at_Co0p5_p95": math.ceil(
            maximum_local_crossing / dt_co05_p95
        ),
        "time_scale_ratios": {
            "minimum_particle_conduction_to_fastest_local_crossing": minimum_particle_scale
            / minimum_local_crossing,
            "maximum_particle_conduction_to_slowest_local_crossing": maximum_particle_scale
            / maximum_local_crossing,
            "particle_conduction_to_published_10dp_bed_crossing_global_min": min(
                minimum_particle_scale / value for value in bed_crossing
            ),
            "particle_conduction_to_published_10dp_bed_crossing_global_max": max(
                maximum_particle_scale / value for value in bed_crossing
            ),
            "first_full_field_output_to_slowest_local_crossing": 1.0
            / maximum_local_crossing,
            "first_full_field_output_to_slowest_10dp_bed_crossing": 1.0
            / max(bed_crossing),
            "first_full_field_output_to_slowest_30dp_full_domain_crossing": 1.0
            / max(full_domain_crossing),
        },
        "evidence_limit": (
            "L/u is a nominal flow-through time derived from the published geometry and inlet "
            "velocity, not a measured hydrodynamic relaxation time. The resolved-crop value uses "
            "the area-preserving pore-opening boundary velocity, whereas the published 10 dp and "
            "30 dp values use the source inlet-channel velocity. At the lowest velocity, the "
            "published full-domain crossing time and the early particle-conduction scale are of "
            "the same order. The calculation therefore must not be described as resolving a "
            "simultaneous inlet-velocity and thermal transient."
        ),
        "fixed_flow_scope": {
            "represented": (
                "Coupled fluid-solid temperature evolution after importing the converged target "
                "U, p, p_rgh and conservative face mass flux phi."
            ),
            "not_represented": [
                "the initial pore-scale momentum transient",
                "pressure-wave propagation",
                "flow instability or buoyancy-driven recirculation",
                "temperature-to-flow feedback during the transition",
                "pebble motion or thermal-expansion-induced rearrangement",
            ],
        },
        "interpretation": (
            "The resolved local crop has a flow-through scale below 0.1 s. The published 10 dp "
            "packed region gives a 0.04-0.20 s nominal crossing time, whereas the full source "
            "domain including inlet and outlet extensions gives 0.12-0.60 s. These scales support "
            "separating the post-adjustment thermal response from the first momentum transient, "
            "but they do not justify treating the latter as resolved. The 1 s thermal solver step "
            "is valid only because converged target U, p, p_rgh and conservative phi are imported "
            "and fixed. The 12 planned curves therefore describe coupled fluid-solid thermal "
            "response after flow adjustment."
        ),
        "new_physical_parameters": [],
        "numerical_note": (
            "Courant limits 0.5 and 1.0 are numerical resolution indicators, not breeder-bed "
            "material or operating parameters."
        ),
    }
    return result


def write_plain_chinese_summary(result: dict[str, object], path: Path) -> None:
    ratios = result["time_scale_ratios"]
    particle = result["particle_radial_conduction_scale_s"]
    local = result["resolved_local_crop_crossing_times_s"]
    bed = result["published_10dp_bed_crossing_times_s"]
    full = result["published_30dp_full_domain_crossing_times_s"]
    lines = [
        "# P418流动与换热时间尺度比较",
        "",
        "## 得到的数值",
        "",
        f"- 当前三维局部球床内，氦气穿过计算域约需 `{min(local):.4f}--{max(local):.4f} s`。",
        f"- 原论文的 `10dp` 颗粒床段约需 `{min(bed):.3f}--{max(bed):.3f} s`。",
        f"- 把原论文入口段、颗粒床和出口段合在一起，约需 `{min(full):.3f}--{max(full):.3f} s`。",
        f"- 由文献导热系数、密度和比热计算的单颗粒径向导热时间约为 "
        f"`{particle['minimum']:.3f}--{particle['maximum']:.3f} s`。",
        f"- 单颗粒导热时间与 `10dp` 床层穿过时间之比覆盖 "
        f"`{ratios['particle_conduction_to_published_10dp_bed_crossing_global_min']:.2f}--"
        f"{ratios['particle_conduction_to_published_10dp_bed_crossing_global_max']:.2f}`。",
        "",
        "## 对瞬态研究的含义",
        "",
        "速度变化后的最初流动调整与早期颗粒导热，在低流速时可能处于相近时间范围。"
        "因此本研究不能把固定流场计算称为完整的速度阶跃流热双向瞬态。",
        "",
        "正式12条曲线从目标工况已经收敛的速度、压力和质量流开始，只计算随后流体与颗粒温度如何变化。"
        "它回答的是“流量已经调到新工况以后，球床温度还需要多久变化”，不回答“阀门刚改变后的前几百毫秒内流场怎样建立”。",
        "",
        "## 没有计算的过程",
        "",
        "- 最初的孔隙尺度动量变化和压力传播；",
        "- 过渡过程中温度反过来改变速度场的双向作用；",
        "- 浮升流、流动不稳定、颗粒移动和热膨胀引起的堆积重排。",
        "",
        "这里没有新增物性或运行参数。颗粒直径、床层尺寸、流速、导热系数、密度和比热均来自参数表中登记的文献。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument(
        "--parameter-source",
        type=Path,
        default=root / "parameters/hccb_p418_physical_parameter_sources.csv",
    )
    parser.add_argument(
        "--particle-scale-summary",
        type=Path,
        default=root / "results/hccb_p418_transient_time_resolution/summary.json",
    )
    parser.add_argument("--flow-axis", type=int, default=2)
    parser.add_argument("--thermal-time-step-s", type=float, default=1.0)
    parser.add_argument(
        "--sourceflow-input-summary",
        type=Path,
        help="Verified formal-case summary containing source-to-pore velocity mapping.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results/hccb_p418_velocity_step_time_scales",
    )
    args = parser.parse_args()
    result = analyze(
        args.topology.resolve(),
        args.field.resolve(),
        args.parameter_source.resolve(),
        args.particle_scale_summary.resolve(),
        args.flow_axis,
        args.thermal_time_step_s,
        args.sourceflow_input_summary.resolve()
        if args.sourceflow_input_summary
        else None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "time_scale_by_velocity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        rows = result["time_scale_comparison_by_velocity"]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_plain_chinese_summary(result, args.output_dir / "README_CN.md")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
