#!/usr/bin/env python3
"""Build the no-solver plan for a matched-initial fully coupled comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "hccb_p418_matched_initial_direct_transport_proposal_20260809"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build() -> dict[str, object]:
    preflight_path = (
        ROOT
        / "results"
        / "hccb_p418_matched_initial_direct_transport_preflight_20260802"
        / "preflight.json"
    )
    validation_path = (
        ROOT
        / "results"
        / "hccb_p418_helium_transport_lookup_20260802"
        / "openfoam_pointwise_validation.json"
    )
    transport_path = (
        ROOT
        / "solver_extensions"
        / "hccbHeliumTransport"
        / "hccbHeliumTransportI.H"
    )
    thermos_path = (
        ROOT
        / "solver_extensions"
        / "hccbHeliumTransport"
        / "hccbHeliumThermos.C"
    )
    parameter_path = ROOT / "parameters" / "literature_parameter_manifest.csv"
    candidate_path = ROOT / "parameters" / "hccb_p418_matched_initial_coupling_candidate.json"
    package_plan_path = (
        ROOT
        / "cloud_build"
        / "p418_matched_initial_direct_transport_smoke_20260802"
        / "smoke_plan.json"
    )
    fixed_reference_path = (
        ROOT
        / "results"
        / "hccb_p418_matched_initial_fixed_flow_reference_0p01s_20260809"
        / "summary.json"
    )
    short_exporter_path = (
        ROOT / "code" / "export_hccb_p418_matched_initial_short_observables.py"
    )

    inputs = (
        preflight_path,
        validation_path,
        transport_path,
        thermos_path,
        parameter_path,
        candidate_path,
        package_plan_path,
        fixed_reference_path,
        short_exporter_path,
    )
    missing = [str(path) for path in inputs if not path.is_file()]
    require(not missing, f"missing plan inputs: {missing}")

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    package_plan = json.loads(package_plan_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    fixed_reference = json.loads(fixed_reference_path.read_text(encoding="utf-8"))
    transport_text = transport_path.read_text(encoding="utf-8")
    thermo_text = thermos_path.read_text(encoding="utf-8")
    parameter_text = parameter_path.read_text(encoding="utf-8")

    require(preflight["openfoam_solver_started"] is False, "preflight started a solver")
    require(preflight["physical_correlations_changed"] is False, "correlations changed")
    require(preflight["operating_conditions_changed"] is False, "conditions changed")
    require(preflight["finite_pressure_table_used"] is False, "finite table selected")
    require(preflight["registered_parameter_ids"] == ["P070", "P071"], "wrong parameters")
    require(
        all(row["byte_identical"] for row in preflight["time_zero_field_identity"].values()),
        "time-zero fields are not byte-identical",
    )
    require(preflight["processor_time_dictionary_count"] == 32, "not a 32-rank case")
    require(preflight["pointwise_maximum_mu_relative_error"] < 1.0e-14, "mu mismatch")
    require(preflight["pointwise_maximum_kappa_relative_error"] < 1.0e-14, "k mismatch")
    require(
        validation["status"] == "hccb_helium_transport_pointwise_check_passed"
        and validation["all_values_positive_and_finite"] is True
        and validation["solver_started"] is False,
        "pointwise validation failed",
    )
    require("viscosityCoefficient_" in transport_text, "direct viscosity missing")
    require("conductivityPressureExponent_" in transport_text, "direct conductivity missing")
    require("perfectGas" in thermo_text, "registered perfect-gas EOS missing")
    require("P070,helium_dynamic_viscosity" in parameter_text, "P070 source missing")
    require("P071,helium_thermal_conductivity" in parameter_text, "P071 source missing")
    require(candidate["new_physical_parameters"] == [], "candidate adds a parameter")
    require(
        fixed_reference["status"]
        == "p418_fixed_flow_matched_initial_0p01s_reference_ready",
        "fixed-flow short reference is missing",
    )
    require(fixed_reference["time_point_count"] == 1001, "wrong fixed reference length")
    require(fixed_reference["end_time_s"] == 0.01, "wrong fixed reference duration")
    require(fixed_reference["interpolation_used"] is False, "fixed reference was interpolated")
    require(fixed_reference["signal_count"] == 15, "wrong fixed reference signals")

    proposal = {
        "status": "matched_initial_direct_transport_representative_plan_ready_no_solver_started",
        "scientific_question": (
            "For the same pore-resolved initial state and boundary conditions, how much "
            "does transient hydrodynamic feedback change the early conjugate-heat-transfer "
            "response relative to the fixed-hydrodynamics reference?"
        ),
        "sequence_id": "source_up_u0p15_T700",
        "why_this_sequence": (
            "The volumetric heat source changes while inlet velocity and inlet temperature "
            "remain unchanged, so hydrodynamic feedback is not mixed with an imposed flow step."
        ),
        "same_initial_state": (
            "Target-endpoint U, p, p_rgh and phi plus source-endpoint fluid and solid T; "
            "all six fields are byte-identical to the fixed-flow case."
        ),
        "same_physics": {
            "helium_viscosity": "P070: mu=0.4646*T_K^0.66*1e-6 Pa s",
            "helium_conductivity": (
                "P071: k=0.1448*(T_K/273)^0.68*"
                "(1+2.5e-3*p_MPa^1.17*(T_K/273)^-1.85) W m^-1 K^-1"
            ),
            "equation_of_state": "existing perfectGas model",
            "new_fitted_parameters": [],
            "geometry_mesh_boundaries_operating_conditions_changed": False,
        },
        "stage_1_representative_smoke": {
            "purpose": (
                "Numerical feasibility and early hydrodynamic-feedback check only; this stage "
                "is not sufficient for the final heat-transfer comparison."
            ),
            "target_end_time_s": package_plan["target_end_time_s"],
            "initial_delta_t_s": package_plan["initial_delta_t_s"],
            "adaptive_max_co": package_plan["adaptive_max_co"],
            "adaptive_max_delta_t_s": package_plan["adaptive_max_delta_t_s"],
            "mpi_ranks": package_plan["mpi_ranks"],
            "memory_gib": package_plan["memory_gib"],
            "time_limit_h": package_plan["time_limit_h"],
            "required_outputs": [
                "32-rank common complete fields at 1e-7 s and 0.01 s",
                "pressure range and Courant-number history",
                "inlet/outlet mass flow and signed mass residual",
                "outlet temperature, maximum solid temperature and volume-average temperatures",
                "cooling-wall heat rate and net outward enthalpy flow",
                "fixed-flow versus fully coupled differences on common output times",
            ],
            "fixed_flow_reference": (
                "results/hccb_p418_matched_initial_fixed_flow_reference_0p01s_20260809/"
                "fixed_flow_reference_0p01s.csv"
            ),
            "fixed_flow_reference_time_points": fixed_reference["time_point_count"],
            "fixed_flow_reference_interpolation_used": False,
        },
        "stage_2_manuscript_comparison": {
            "approved": False,
            "automatic_submission": False,
            "decision_rule": (
                "Only after stage 1 finishes without a solver error, non-positive pressure or "
                "temperature, non-finite field, incomplete 32-rank output or uncontrolled "
                "Courant growth. The extension duration must be selected from the measured "
                "time-step cost and the observed thermal-response time; 0.01 s alone cannot "
                "support a manuscript claim about the full thermal transient."
            ),
            "minimum_comparison_quantities": [
                "pressure drop",
                "outlet temperature",
                "maximum solid temperature",
                "volume-average fluid and solid temperatures",
                "cooling-wall heat rate",
                "signed mass residual",
                "net outward enthalpy flow",
            ],
            "reporting_rule": (
                "Report dimensional RMSE, largest absolute difference and endpoint difference; "
                "do not introduce an arbitrary agreement percentage."
            ),
        },
        "evidence": {
            "preflight": str(preflight_path.relative_to(ROOT)),
            "preflight_sha256": sha256(preflight_path),
            "pointwise_validation": str(validation_path.relative_to(ROOT)),
            "pointwise_validation_sha256": sha256(validation_path),
            "direct_transport_source_sha256": sha256(transport_path),
            "thermo_registration_source_sha256": sha256(thermos_path),
            "parameter_manifest_sha256": sha256(parameter_path),
            "fixed_flow_reference_summary": str(fixed_reference_path.relative_to(ROOT)),
            "fixed_flow_reference_summary_sha256": sha256(fixed_reference_path),
            "short_observable_exporter_sha256": sha256(short_exporter_path),
        },
        "openfoam_solver_started_by_this_plan": False,
        "formal_solver_submission_approved": False,
        "solver_submission_requires_exact_phrase": "批准短算",
    }
    return proposal


def markdown(plan: dict[str, object]) -> str:
    stage1 = plan["stage_1_representative_smoke"]
    stage2 = plan["stage_2_manuscript_comparison"]
    return f"""# 固定流场与完全耦合的代表性对照方案

## 要回答的问题

在同一个三维初始场和相同边界条件下，允许氦气速度和压力随时间变化，会在多大程度上改变早期流固共轭换热响应？代表序列为`source_up_u0p15_T700`，因为它只改变颗粒体热源，入口速度和入口温度不变，不会把热源阶跃与外加流量阶跃混在一起。

## 哪些条件保持不变

- `U、p、p_rgh、phi、流体T、颗粒T`六个初始场与固定流场算例逐字节一致。
- 网格、颗粒装填、边界条件、操作工况、状态方程、比热和固体物性不变。
- 氦气黏度和导热系数仍使用文献登记的P070/P071关联式，只是由OpenFOAM直接计算，不再查有限压力范围的二维表。
- 没有新拟合参数。P070/P071的12个OpenFOAM逐点检查相对误差均小于`1e-14`。

## 第一步：0.01 s代表性短算

- 32 MPI，64 GiB，最长2 h。
- 首步`{stage1['initial_delta_t_s']:.0e} s`，随后按`maxCo={stage1['adaptive_max_co']}`自适应，最大时间步`{stage1['adaptive_max_delta_t_s']:.0e} s`。
- 必须保存1e-7 s和0.01 s的32分区完整场，并记录压力、Courant数、质量收支、能量收支、出口温度、颗粒最高温度和冷却壁热量。
- 这一步只证明直接物性实现能否支持完全耦合启动，**不能单独支撑完整热响应结论**。

## 第二步：论文定量对照

只有第一步正常完成，才根据实测时间步成本和温度响应速度决定延长时间。对照量至少包含：{'、'.join(stage2['minimum_comparison_quantities'])}。报告量为有量纲RMSE、最大绝对差和末时刻绝对差，不设人为“一致百分比”。

## 当前状态

- 直接物性模块已编译并通过逐点检查。
- 32分区初始场和提交包已做无求解检查。
- OpenFOAM求解器未由本方案启动。
- 第一步仍需要明确的“批准短算”后才能提交；第二步不会自动提交。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = build()
    plan_path = output / "plan.json"
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path = output / "PLAN_CN.md"
    md_path.write_text(markdown(plan), encoding="utf-8")
    record = {
        "status": plan["status"],
        "plan_sha256": sha256(plan_path),
        "plan_cn_sha256": sha256(md_path),
        "openfoam_solver_started_by_this_plan": False,
    }
    (output / "record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
