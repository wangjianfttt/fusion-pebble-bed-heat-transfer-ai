#!/usr/bin/env python3
"""Verify that OpenFOAM and the neural physics term use the same solid thermo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from build_hccb_li4sio4_transient_solid_thermo import (
    PARAMETER_IDS,
    physical_properties_text,
    table_values,
)
from hccb_source_backed_thermophysical import (
    OPENFOAM_TSTD_K,
    li4sio4_sensible_internal_energy,
    li4sio4_specific_heat_capacity,
    load_hccb_thermophysical_parameters,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def piecewise_linear_integral(
    nodes: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
    *,
    reference: float = OPENFOAM_TSTD_K,
) -> np.ndarray:
    """Reproduce OpenFOAM-13 integratedNonUniformTable::intfdT."""
    nodes = np.asarray(nodes, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if nodes.ndim != 1 or values.shape != nodes.shape or len(nodes) < 2:
        raise ValueError("nodes and values must be one-dimensional tables")
    if np.any(np.diff(nodes) <= 0.0):
        raise ValueError("table nodes must be strictly increasing")
    if not nodes[0] <= reference <= nodes[-1]:
        raise ValueError("reference temperature is outside the table")
    if np.any(query < nodes[0]) or np.any(query > nodes[-1]):
        raise ValueError("query temperature is outside the table")

    interval = np.diff(nodes)
    cumulative = np.concatenate(
        ([0.0], np.cumsum(0.5 * (values[:-1] + values[1:]) * interval))
    )

    def primitive(temperature: np.ndarray) -> np.ndarray:
        index = np.searchsorted(nodes, temperature, side="right") - 1
        index = np.clip(index, 0, len(nodes) - 2)
        delta = temperature - nodes[index]
        gradient = (values[index + 1] - values[index]) / interval[index]
        return cumulative[index] + values[index] * delta + 0.5 * gradient * delta**2

    return primitive(query) - primitive(np.asarray(reference, dtype=np.float64))


def verify_actual_case(case: Path | None) -> dict[str, object]:
    if case is None:
        return {"checked": False, "reason": "no actual step case was supplied"}
    properties = case / "constant/solid/physicalProperties"
    metadata = case / "transient_solid_thermo.json"
    if not properties.is_file() or not metadata.is_file():
        raise FileNotFoundError(f"incomplete step case: {case}")
    actual_text = properties.read_text(encoding="ascii")
    registered = json.loads(metadata.read_text(encoding="utf-8"))
    expected_text = physical_properties_text()
    checks = {
        "dictionary_exactly_matches_current_source_builder": actual_text == expected_text,
        "P406_is_absent": "P406" not in actual_text,
        "P428_to_P431_are_declared": "P428-P431" in actual_text,
        "uses_eIcoTabulated": "thermo          eIcoTabulated;" in actual_text,
        "uses_sensible_internal_energy": "energy          sensibleInternalEnergy;" in actual_text,
        "metadata_hash_matches_file": registered.get("physical_properties_sha256")
        == sha256(properties),
        "metadata_parameter_ids_match": registered.get("parameter_ids") == list(PARAMETER_IDS),
    }
    return {
        "checked": True,
        "case_path": str(case),
        "physical_properties_sha256": sha256(properties),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def build_summary(actual_case: Path | None = None) -> dict[str, object]:
    params = load_hccb_thermophysical_parameters()
    nodes, table_cp = table_values()
    dense = np.linspace(nodes[0], nodes[-1], 20001, dtype=np.float64)
    dense_tensor = torch.as_tensor(dense, dtype=torch.float64)
    analytical_cp = li4sio4_specific_heat_capacity(dense_tensor).detach().cpu().numpy()
    table_interpolated_cp = np.interp(dense, nodes, table_cp)
    analytical_energy = (
        li4sio4_sensible_internal_energy(dense_tensor).detach().cpu().numpy()
    )
    openfoam_table_energy = piecewise_linear_integral(nodes, table_cp, dense)

    cp_relative_error = np.abs(table_interpolated_cp - analytical_cp) / analytical_cp
    energy_absolute_error = np.abs(openfoam_table_energy - analytical_energy)
    energy_span = float(np.ptp(analytical_energy))
    energetic = np.abs(analytical_energy) >= 0.01 * energy_span
    energy_relative_error = energy_absolute_error[energetic] / np.abs(
        analytical_energy[energetic]
    )

    derivative_temperature = torch.linspace(
        nodes[0], nodes[-1], 1001, dtype=torch.float64, requires_grad=True
    )
    derivative_energy = li4sio4_sensible_internal_energy(derivative_temperature)
    derivative = torch.autograd.grad(derivative_energy.sum(), derivative_temperature)[0]
    derivative_cp = li4sio4_specific_heat_capacity(derivative_temperature.detach())
    derivative_relative_error = torch.max(
        torch.abs(derivative - derivative_cp) / derivative_cp
    )

    actual = verify_actual_case(actual_case)
    checks = {
        "analytical_internal_energy_derivative_matches_P429": float(
            derivative_relative_error
        )
        < 1.0e-12,
        "OpenFOAM_table_Cp_interpolation_relative_error_below_1e_4": float(
            cp_relative_error.max()
        )
        < 1.0e-4,
        "OpenFOAM_table_energy_error_below_1e_4_of_full_span": float(
            energy_absolute_error.max() / energy_span
        )
        < 1.0e-4,
        "heat_capacity_is_positive": bool(np.all(analytical_cp > 0.0)),
        "internal_energy_is_monotonic": bool(np.all(np.diff(analytical_energy) > 0.0)),
        "actual_case_matches_when_supplied": bool(
            not actual["checked"] or actual["all_checks_pass"]
        ),
    }
    return {
        "status": "passed_same_transient_solid_thermo_in_openfoam_and_neural_physics"
        if all(checks.values())
        else "failed",
        "scientific_meaning_cn": (
            "OpenFOAM温度阶跃与神经网络物理约束使用同一套纯Li4SiO4平滑量热关系。"
            "稳态字典中的P406没有进入瞬态储热计算。"
        ),
        "parameter_ids": list(PARAMETER_IDS),
        "temperature_range_K": [float(nodes[0]), float(nodes[-1])],
        "openfoam_reference_temperature_K": OPENFOAM_TSTD_K,
        "table_point_count": int(len(nodes)),
        "table_step_K": 5.0,
        "numerical_comparison": {
            "maximum_Cp_interpolation_relative_error": float(cp_relative_error.max()),
            "maximum_internal_energy_absolute_error_J_kg": float(
                energy_absolute_error.max()
            ),
            "maximum_internal_energy_error_over_full_span": float(
                energy_absolute_error.max() / energy_span
            ),
            "maximum_internal_energy_relative_error_above_1pct_span": float(
                energy_relative_error.max()
            ),
            "maximum_analytical_derivative_relative_error": float(
                derivative_relative_error
            ),
            "heat_capacity_range_J_kg_K": [
                float(analytical_cp.min()),
                float(analytical_cp.max()),
            ],
            "internal_energy_span_J_kg": energy_span,
        },
        "checks": checks,
        "actual_step_case": actual,
        "model_scope_cn": (
            "P428-P429是298--1300 K的平滑关系，不解析938 K和996 K附近的尖锐热容异常，"
            "也不代替具体制造批次颗粒的实测热容。"
        ),
        "new_fitted_physical_parameters": [],
    }


def write_chinese_summary(path: Path, payload: dict[str, object]) -> None:
    numbers = payload["numerical_comparison"]
    actual = payload["actual_step_case"]
    lines = [
        "# P418瞬态固体储热关系核对",
        "",
        f"- 结论：`{payload['status']}`",
        f"- 温度范围：{payload['temperature_range_K'][0]:g}--{payload['temperature_range_K'][1]:g} K",
        f"- OpenFOAM参考温度：{payload['openfoam_reference_temperature_K']:.2f} K",
        f"- 热容表：{payload['table_point_count']}个点，间隔5 K",
        f"- 热容插值最大相对误差：{numbers['maximum_Cp_interpolation_relative_error']:.6g}",
        f"- 内能最大绝对差：{numbers['maximum_internal_energy_absolute_error_J_kg']:.6g} J/kg",
        f"- 内能最大差占全温区内能跨度：{numbers['maximum_internal_energy_error_over_full_span']:.6g}",
        "",
        "OpenFOAM 13对该热容表做分段线性积分；Python物理约束直接使用P428解析内能。",
        "两者来自同一P429热容关系，数值差只来自5 K表格插值，不含新拟合参数。",
        "P406只用于不依赖热容的稳态端点，没有进入温度阶跃。",
    ]
    if actual["checked"]:
        lines.extend(
            [
                "",
                "## 实际阶跃算例",
                "",
                f"- 路径：`{actual['case_path']}`",
                f"- 当前生成器与算例字典完全一致：{actual['checks']['dictionary_exactly_matches_current_source_builder']}",
                f"- 文件哈希与元数据一致：{actual['checks']['metadata_hash_matches_file']}",
            ]
        )
    lines.extend(
        [
            "",
            "## 物理范围",
            "",
            payload["model_scope_cn"],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual-step-case", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chinese-summary", type=Path, required=True)
    args = parser.parse_args()
    payload = build_summary(
        args.actual_step_case.resolve() if args.actual_step_case else None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_chinese_summary(args.chinese_summary, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"].startswith("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
