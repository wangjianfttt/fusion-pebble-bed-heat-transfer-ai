#!/usr/bin/env python3
"""Check which physical fields and engineering quantities are available for P418 learning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any

from build_hccb_p418_shared_mesh_dataset import FIELD_KEYS, TOPOLOGY_KEYS
from summarize_hccb_p418_formal_steady_tail import verified_recorded_summary


OPERATING_INPUTS = (
    "inlet_velocity_m_s",
    "inlet_temperature_K",
    "solid_heat_source_W_m3",
    "outlet_pressure_Pa",
    "cooling_wall_temperature_K",
)

STATE_GROUPS = {
    "fluid_state": (
        "fluid_velocity_m_s",
        "fluid_pressure_Pa",
        "fluid_temperature_K",
        "fluid_density_kg_m3",
    ),
    "solid_state": ("solid_temperature_K",),
    "mass_transport": (
        "fluid_internal_face_mass_flow_kg_s",
        "fluid_boundary_face_mass_flow_kg_s",
    ),
    "boundary_state": (
        "fluid_boundary_velocity_m_s",
        "fluid_boundary_pressure_Pa",
        "fluid_boundary_temperature_K",
        "fluid_boundary_density_kg_m3",
        "solid_boundary_temperature_K",
    ),
}

RESULT_VALUES = {
    "pressure_drop_Pa": ("flow", "pressure_drop_Pa"),
    "outlet_temperature_K": ("temperature", "outlet_average_K"),
    "solid_maximum_temperature_K": ("temperature", "solid_maximum_K"),
    "generated_solid_heat_W": ("heat_balance", "solid_generated_power_W"),
    "cooling_wall_heat_W": ("heat_balance", "cooling_wall_heat_flow_W"),
    "inlet_enthalpy_flow_W": ("heat_balance", "inlet_enthalpy_flow_W"),
    "outlet_enthalpy_flow_W": ("heat_balance", "outlet_enthalpy_flow_W"),
    "relative_mass_difference": ("flow", "relative_mass_difference"),
    "relative_energy_difference": ("heat_balance", "relative_energy_difference"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nested(document: dict[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = document
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sample_path_from_marker(case: Path, marker: dict[str, Any]) -> Path:
    sample = Path(str(marker["training_sample"]))
    return sample if sample.is_absolute() else (case / sample).resolve()


def npz_keys(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return {
            Path(member).stem
            for member in archive.namelist()
            if member.endswith(".npy")
        }


def result_summary_path(case: Path, time_name: str) -> Path:
    exact = case / f"cht_result_summary_{time_name}.json"
    if exact.is_file():
        return exact
    candidates = sorted(case.glob("cht_result_summary_*.json"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(exact)


def check_case(
    marker_path: Path,
    *,
    verify_checksum: bool,
    reference_shapes: dict[str, list[int]] | None,
    reference_dtypes: dict[str, str] | None,
    reference_patches: tuple[tuple[str, ...], tuple[str, ...]] | None,
) -> tuple[dict[str, Any], list[str], dict[str, list[int]], dict[str, str], tuple[tuple[str, ...], tuple[str, ...]]]:
    case = marker_path.parent
    problems: list[str] = []
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    condition_id = str(marker.get("condition_id", case.name))
    if condition_id != case.name:
        problems.append(f"completion marker condition_id={condition_id}, directory={case.name}")
    if marker.get("solver_finished") is not True:
        problems.append("solver_finished is not true")
    for key in ("relative_mass_difference", "relative_energy_difference"):
        if not finite_number(marker.get(key)):
            problems.append(f"completion marker {key} is missing or non-finite")
    steady_summary_path = None
    steady_full_field_available = None
    try:
        steady_summary_path, steady_document = verified_recorded_summary(case, marker)
        steady_full_field_available = bool(steady_document["full_field_available"])
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        problems.append(str(error))

    sample = sample_path_from_marker(case, marker)
    metadata_path = sample.parent / "metadata.json"
    if not sample.is_file():
        raise FileNotFoundError(sample)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 3:
        problems.append(f"sample schema_version={metadata.get('schema_version')}, expected 3")
    if marker.get("training_sample_schema_version") != 3:
        problems.append("completion marker does not identify a schema-3 sample")

    expected_keys = set(TOPOLOGY_KEYS + FIELD_KEYS)
    archive_keys = npz_keys(sample)
    if archive_keys != expected_keys:
        problems.append(
            f"NPZ keys differ: missing={sorted(expected_keys - archive_keys)}, "
            f"unexpected={sorted(archive_keys - expected_keys)}"
        )
    shapes = {str(key): list(value) for key, value in metadata.get("array_shapes", {}).items()}
    dtypes = {str(key): str(value) for key, value in metadata.get("array_dtypes", {}).items()}
    if set(shapes) != expected_keys:
        problems.append("metadata array_shapes do not cover the schema-3 arrays exactly")
    if set(dtypes) != expected_keys:
        problems.append("metadata array_dtypes do not cover the schema-3 arrays exactly")
    for key in expected_keys.intersection(shapes):
        if not shapes[key] or any(int(length) < 0 for length in shapes[key]):
            problems.append(f"invalid shape for {key}: {shapes[key]}")
    if reference_shapes is not None and shapes != reference_shapes:
        problems.append("array shapes differ from the first completed fixed-mesh case")
    if reference_dtypes is not None and dtypes != reference_dtypes:
        problems.append("array dtypes differ from the first completed fixed-mesh case")

    patches = (
        tuple(str(value) for value in metadata.get("fluid_patch_names", [])),
        tuple(str(value) for value in metadata.get("solid_patch_names", [])),
    )
    if reference_patches is not None and patches != reference_patches:
        problems.append("boundary patch ordering differs from the first completed case")

    physical = metadata.get("physical_conditions", {})
    if physical.get("operating_condition_id") != case.name:
        problems.append("sample physical_conditions has the wrong operating_condition_id")
    for key in OPERATING_INPUTS:
        if not finite_number(physical.get(key)):
            problems.append(f"physical input {key} is missing or non-finite")

    marker_digest = marker.get("training_sample_sha256")
    metadata_digest = metadata.get("sample_sha256")
    if not marker_digest or marker_digest != metadata_digest:
        problems.append("sample checksum differs between completion marker and metadata")
    if verify_checksum and marker_digest != sha256(sample):
        problems.append("sample file checksum differs from the recorded checksum")

    steady_iteration = str(
        marker.get("steady_iteration_end", marker.get("time"))
    )
    solver_time_semantics = str(
        marker.get("solver_time_semantics", "steady_iteration_index")
    )
    physical_time_s = marker.get("physical_time_s")
    if solver_time_semantics != "steady_iteration_index":
        problems.append(
            "completion marker does not identify the result index as a steady iteration"
        )
    if physical_time_s is not None:
        problems.append("steady completion marker unexpectedly reports physical_time_s")
    summary_path = result_summary_path(case, steady_iteration)
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    if result.get("solver_finished") is not True:
        problems.append("result summary does not report solver completion")
    if result.get("all_reported_values_are_finite") is not True:
        problems.append("result summary contains a non-finite reported value")
    result_values: dict[str, float | None] = {}
    for name, key_path in RESULT_VALUES.items():
        try:
            value = nested(result, key_path)
        except KeyError:
            value = None
        if not finite_number(value):
            problems.append(f"result value {name} is missing or non-finite")
            result_values[name] = None
        else:
            result_values[name] = float(value)

    for key in (
        "inlet_velocity_m_s",
        "inlet_temperature_K",
        "solid_heat_source_W_m3",
        "cooling_wall_temperature_K",
    ):
        result_value = result.get("physical_conditions", {}).get(key)
        sample_value = physical.get(key)
        if finite_number(result_value) and finite_number(sample_value):
            if not math.isclose(float(result_value), float(sample_value), rel_tol=0.0, abs_tol=1.0e-12):
                problems.append(f"result and sample disagree for {key}")
        else:
            problems.append(f"cannot compare result and sample for {key}")

    row = {
        "condition_id": case.name,
        "steady_iteration": steady_iteration,
        "solver_time_semantics": solver_time_semantics,
        "physical_time_s": physical_time_s,
        "sample_path": str(sample),
        "sample_schema_version": metadata.get("schema_version"),
        "fluid_cells": metadata.get("fluid_cells"),
        "solid_cells": metadata.get("solid_cells"),
        "interface_faces": metadata.get("interface_faces"),
        "inlet_velocity_m_s": physical.get("inlet_velocity_m_s"),
        "inlet_temperature_K": physical.get("inlet_temperature_K"),
        "solid_heat_source_W_m3": physical.get("solid_heat_source_W_m3"),
        "outlet_pressure_Pa": physical.get("outlet_pressure_Pa"),
        "cooling_wall_temperature_K": physical.get("cooling_wall_temperature_K"),
        **result_values,
        "sample_checksum_recomputed": verify_checksum,
        "steady_final_window_summary": (
            str(steady_summary_path) if steady_summary_path is not None else None
        ),
        "steady_final_window_full_field_available": steady_full_field_available,
        "status": "complete" if not problems else "problem",
        "problems": "; ".join(problems),
    }
    return row, problems, shapes, dtypes, patches


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["condition_id", "status", "problems"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_chinese(path: Path, payload: dict[str, Any]) -> None:
    count = int(payload["completed_case_count"])
    expected = int(payload["expected_case_count"])
    partial = count != expected
    conclusion = (
        f"当前{count}个已完成工况的三维训练字段与工程结果都完整，尚有{expected - count}个稳态工况未完成。"
        if partial
        else f"{expected}个正式稳态工况的三维训练字段与工程结果均完整。"
    )
    lines = [
        "# P418三维训练数据内容检查",
        "",
        "## 结论",
        "",
        conclusion,
        "这里只检查OpenFOAM实际输出、三维样本和工程结果是否对应，不增加任何材料物性或运行参数。",
        "",
        "## 神经网络直接读取的内容",
        "",
        "| 类别 | 实际字段 | 用途 |",
        "|---|---|---|",
        "| 运行工况 | 入口速度、入口温度、颗粒体积发热率、出口压力、冷却壁温度 | 条件输入 |",
        "| 流体三维场 | 速度、压力、温度、密度 | PINN/图--Transformer的场输出与方程计算 |",
        "| 颗粒三维场 | 颗粒温度 | 颗粒温度场、热点和热储量 |",
        "| 质量输运 | 流体内部面与边界面质量流量 | 质量守恒和出口平均量 |",
        "| 网格与流固界面 | 单元中心、体积、面连接、面积向量、流固对应面 | 三维图结构、动量和能量方程 |",
        "",
        "## 可直接对照的工程量",
        "",
        "| 工程量 | 来源或计算方式 |",
        "|---|---|",
        "| 压降 | OpenFOAM入口和出口平均压力；也可由预测边界压力重新计算 |",
        "| 出口温度 | OpenFOAM出口平均值；也可由出口温度和质量流量重新计算 |",
        "| 颗粒最高温度及位置 | 颗粒温度最大值，并与对应单元坐标结合 |",
        "| 冷却壁热量 | OpenFOAM工程结果直接保存；正式后处理还会导出逐面热流 |",
        "| 颗粒发热功率、入/出口焓流 | OpenFOAM能量结果直接保存 |",
        "| 质量与总能量差 | 由全部边界流量、壁面热量、颗粒发热和焓流联合计算 |",
        "",
        "## 物性处理",
        "",
        "氦气密度已存入三维场。动力黏度和导热系数不作为人工标签保存，而是在计算物理方程时，由文献登记的P070和P071温度/压力关系重新计算。颗粒密度、导热系数和比热也使用参数表中的文献值。",
        "",
        "## 当前进度",
        "",
        f"- 正式稳态工况：`{count}/{expected}`。",
        "- CSV中的`steady_iteration`是稳态非线性迭代编号，不是物理秒；`physical_time_s`为空。",
        f"- 已完成工况编号：`{', '.join(payload['completed_condition_ids'])}`。",
    ]
    if payload["missing_condition_ids"]:
        lines.append(f"- 尚未完成：`{', '.join(payload['missing_condition_ids'])}`。")
    if payload["verify_file_checksums"]:
        lines.append("- 本次重新计算了每个三维样本的SHA256。")
    else:
        lines.append("- 本次核对了样本内容目录和已记录的SHA256；正式60工况数据集生成时会再逐文件计算SHA256并读取全部数组。")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-case-count", type=int, default=60)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="report completed formal cases before all expected cases have finished",
    )
    parser.add_argument(
        "--verify-file-checksums",
        action="store_true",
        help="re-read every large NPZ file and compare its SHA256",
    )
    args = parser.parse_args()

    matrix_root = args.matrix_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_condition_ids = sorted(
        case.name
        for case in matrix_root.iterdir()
        if (
            case.is_dir()
            and ".precomputed_input_" not in case.name
            and (case / "cht_smoke_metadata.json").is_file()
        )
    )
    markers = sorted(matrix_root.glob("*/formal_sample_complete.json"))
    if not markers:
        raise FileNotFoundError(f"no formal_sample_complete.json under {matrix_root}")
    if len(all_condition_ids) != args.expected_case_count:
        raise ValueError(
            f"matrix contains {len(all_condition_ids)} registered cases, "
            f"expected {args.expected_case_count}"
        )
    if not args.allow_partial and len(markers) != args.expected_case_count:
        raise ValueError(
            f"completed formal cases {len(markers)}/{args.expected_case_count}; "
            "use --allow-partial only for an explicit progress report"
        )

    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    reference_shapes = None
    reference_dtypes = None
    reference_patches = None
    for marker in markers:
        try:
            row, case_problems, shapes, dtypes, patches = check_case(
                marker,
                verify_checksum=args.verify_file_checksums,
                reference_shapes=reference_shapes,
                reference_dtypes=reference_dtypes,
                reference_patches=reference_patches,
            )
        except Exception as error:  # retain the condition identifier in the result table
            row = {
                "condition_id": marker.parent.name,
                "status": "problem",
                "problems": str(error),
            }
            case_problems = [str(error)]
            shapes, dtypes, patches = {}, {}, ((), ())
        if reference_shapes is None and not case_problems:
            reference_shapes = shapes
            reference_dtypes = dtypes
            reference_patches = patches
        rows.append(row)
        problems.extend(f"{marker.parent.name}: {problem}" for problem in case_problems)

    completed_ids = [str(row["condition_id"]) for row in rows if row.get("status") == "complete"]
    missing_ids = sorted(set(all_condition_ids) - set(completed_ids))
    all_completed_rows_valid = len(completed_ids) == len(markers) and not problems
    full_matrix_ready = all_completed_rows_valid and len(completed_ids) == args.expected_case_count
    payload = {
        "status": (
            "p418_training_data_coverage_ready"
            if full_matrix_ready
            else "p418_partial_training_data_coverage_ready"
            if args.allow_partial and all_completed_rows_valid
            else "failed"
        ),
        "matrix_root": str(matrix_root),
        "expected_case_count": args.expected_case_count,
        "completed_case_count": len(completed_ids),
        "completed_fraction": len(completed_ids) / args.expected_case_count,
        "completed_condition_ids": completed_ids,
        "missing_condition_ids": missing_ids,
        "solver_time_semantics": "steady_iteration_index",
        "physical_time_s": None,
        "steady_iteration_column": "steady_iteration",
        "verify_file_checksums": args.verify_file_checksums,
        "schema_version": 3,
        "operating_inputs": list(OPERATING_INPUTS),
        "topology_keys": list(TOPOLOGY_KEYS),
        "state_groups": {key: list(values) for key, values in STATE_GROUPS.items()},
        "field_keys": list(FIELD_KEYS),
        "engineering_result_values": list(RESULT_VALUES),
        "derived_from_predicted_fields": {
            "pressure_drop_Pa": "fluid boundary pressure",
            "outlet_temperature_K": "outlet temperature and mass flow",
            "solid_maximum_temperature_K": "solid temperature",
            "solid_hotspot_location_m": "argmax(solid temperature) and solid cell centroid",
        },
        "separate_postprocess_targets": {
            "cooling_wall_face_heat_W": "OpenFOAM boundary heat-flux export",
            "fluid_solid_interphase_heat_W": "paired fluid-solid interface heat-flux export",
            "regional_mass_and_energy_flow_W": "regional face aggregation",
        },
        "field_dependent_properties": {
            "helium_dynamic_viscosity_Pa_s": "P070 evaluated from predicted temperature",
            "helium_thermal_conductivity_W_m_K": "P071 evaluated from predicted temperature and pressure",
            "helium_density_kg_m3": "stored field and P389 consistency calculation",
        },
        "exporter_checks": (
            "The schema-3 exporter rejects non-finite arrays, non-positive cell volumes "
            "or boundary areas, and out-of-range interface cell indices before writing "
            "formal_sample_complete.json."
        ),
        "problems": problems,
        "new_physical_parameters": [],
        "case_csv": str(output / "case_training_data_coverage.csv"),
    }
    write_csv(output / "case_training_data_coverage.csv", rows)
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_chinese(output / "P418_训练数据字段完整性_CN.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] == "failed":
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
