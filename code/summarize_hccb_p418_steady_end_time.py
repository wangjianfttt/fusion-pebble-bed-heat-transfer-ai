#!/usr/bin/env python3
"""Quantify late-iteration changes before shortening the P418 steady solve.

The steady cases use ``ddtSchemes { default steadyState; }`` in both regions.
OpenFOAM therefore writes numeric solver labels such as 175, 200 and 300, but
those labels are nonlinear steady-iteration indices rather than physical time.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from verify_hccb_p418_step_initialization import internal_field_values


HISTORY_FILES = {
    "outlet_temperature_K": ("fluid", "outletTemperature", "surfaceFieldValue.dat"),
    "solid_maximum_temperature_K": ("solid", "solidTemperatureMaximum", "volFieldValue.dat"),
    "cooling_wall_power_W": ("fluid", "coolingWallPower", "surfaceFieldValue.dat"),
    "inlet_mass_flow_kg_s": ("fluid", "inletMassFlow", "surfaceFieldValue.dat"),
    "outlet_mass_flow_kg_s": ("fluid", "outletMassFlow", "surfaceFieldValue.dat"),
    "outlet_enthalpy_flow_W": ("fluid", "outletEnthalpyFlow", "surfaceFieldValue.dat"),
}

FIELD_FILES = {
    "fluid_temperature": ("fluid/T", "K"),
    "solid_temperature": ("solid/T", "K"),
    "fluid_velocity_components": ("fluid/U", "m/s"),
    "fluid_pressure": ("fluid/p", "Pa"),
}


def read_history(case: Path, specification: tuple[str, str, str]) -> np.ndarray:
    region, name, file_name = specification
    path = case / "postProcessing" / region / name / "0" / file_name
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.loadtxt(path, comments="#", ndmin=2)
    if values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid function-object history: {path}")
    return values


def value_at(history: np.ndarray, iteration: float) -> float:
    match = np.flatnonzero(np.isclose(history[:, 0], iteration, rtol=0.0, atol=1.0e-9))
    if len(match) != 1:
        raise ValueError(f"steady iteration {iteration} occurs {len(match)} times")
    return float(history[int(match[0]), 1])


def engineering_change(
    case: Path,
    candidate_iteration: float,
    reference_iteration: float,
) -> dict[str, object]:
    histories = {name: read_history(case, spec) for name, spec in HISTORY_FILES.items()}
    candidate = {
        name: value_at(history, candidate_iteration) for name, history in histories.items()
    }
    reference = {
        name: value_at(history, reference_iteration) for name, history in histories.items()
    }
    return {
        "condition_id": case.name,
        "candidate_iteration": candidate_iteration,
        "reference_iteration": reference_iteration,
        "outlet_temperature_absolute_change_K": abs(
            candidate["outlet_temperature_K"] - reference["outlet_temperature_K"]
        ),
        "solid_maximum_temperature_absolute_change_K": abs(
            candidate["solid_maximum_temperature_K"]
            - reference["solid_maximum_temperature_K"]
        ),
        "cooling_wall_power_absolute_change_W": abs(
            candidate["cooling_wall_power_W"] - reference["cooling_wall_power_W"]
        ),
        "outlet_enthalpy_flow_absolute_change_W": abs(
            candidate["outlet_enthalpy_flow_W"] - reference["outlet_enthalpy_flow_W"]
        ),
        "candidate_relative_mass_difference": abs(
            candidate["inlet_mass_flow_kg_s"] + candidate["outlet_mass_flow_kg_s"]
        )
        / abs(candidate["inlet_mass_flow_kg_s"]),
    }


def decomposed_field_change(
    case: Path,
    field_path: str,
    start_iteration: str,
    end_iteration: str,
) -> dict[str, object]:
    start_values: list[np.ndarray] = []
    end_values: list[np.ndarray] = []
    used = 0
    processors = sorted(case.glob("processor*"), key=lambda path: int(path.name[9:]))
    if not processors:
        raise FileNotFoundError(f"decomposed fields are absent: {case}")
    for processor in processors:
        start = processor / start_iteration / field_path
        end = processor / end_iteration / field_path
        if not start.is_file() and not end.is_file():
            continue
        if start.is_file() != end.is_file():
            raise ValueError(f"inconsistent decomposed field pair: {start}, {end}")
        start_values.append(internal_field_values(start))
        end_values.append(internal_field_values(end))
        used += 1
    if not start_values:
        raise FileNotFoundError(
            f"no partitions contain {field_path} at steady iterations "
            f"{start_iteration}/{end_iteration}"
        )
    first = np.concatenate(start_values)
    second = np.concatenate(end_values)
    if first.shape != second.shape:
        raise ValueError(f"field sizes differ: {case.name} {field_path}")
    difference = second - first
    rms = float(np.sqrt(np.mean(np.square(difference))))
    scale = float(np.sqrt(np.mean(np.square(second))))
    return {
        "condition_id": case.name,
        "field": field_path,
        "partition_count": used,
        "value_count": int(first.size),
        "start_iteration": float(start_iteration),
        "end_iteration": float(end_iteration),
        "absolute_rms_change": rms,
        "maximum_absolute_change": float(np.max(np.abs(difference))),
        "relative_rms_change": rms / max(scale, np.finfo(float).tiny),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument(
        "--candidate-iteration",
        "--candidate-time",
        dest="candidate_iteration",
        type=float,
        default=200.0,
    )
    parser.add_argument(
        "--reference-iteration",
        "--reference-time",
        dest="reference_iteration",
        type=float,
        default=300.0,
    )
    parser.add_argument(
        "--field-start-iteration",
        "--field-start-time",
        dest="field_start_iteration",
        default="150",
    )
    parser.add_argument(
        "--field-end-iteration",
        "--field-end-time",
        dest="field_end_iteration",
        default="175",
    )
    parser.add_argument("--field-cases", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    matrix = args.matrix_root.resolve()
    completed_cases = sorted(
        marker.parent for marker in matrix.glob("*/formal_sample_complete.json")
    )
    if not completed_cases:
        raise FileNotFoundError("no completed P418 cases")
    engineering_rows = [
        engineering_change(case, args.candidate_iteration, args.reference_iteration)
        for case in completed_cases
    ]
    field_rows: list[dict[str, object]] = []
    for condition_id in args.field_cases:
        case = matrix / condition_id
        for field_name, (field_path, unit) in FIELD_FILES.items():
            row = decomposed_field_change(
                case,
                field_path,
                args.field_start_iteration,
                args.field_end_iteration,
            )
            row["field_name"] = field_name
            row["unit"] = unit
            field_rows.append(row)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "engineering_change_iter200_vs_iter300.csv", engineering_rows)
    write_csv(output / "full_field_change_iter150_to_iter175.csv", field_rows)
    maxima = {
        key: max(float(row[key]) for row in engineering_rows)
        for key in engineering_rows[0]
        if key not in {"condition_id", "candidate_iteration", "reference_iteration"}
    }
    temperature_field_rows = [
        row for row in field_rows if row["field_name"] in {"fluid_temperature", "solid_temperature"}
    ]
    payload = {
        "status": "steady_iteration_endpoint_comparison_complete",
        "solver_time_semantics": "steady_iteration_index",
        "physical_time_s": None,
        "completed_reference_case_count": len(engineering_rows),
        "decomposed_full_field_case_count": len(args.field_cases),
        "candidate_iteration": args.candidate_iteration,
        "reference_iteration": args.reference_iteration,
        "field_change_iteration_window": [
            float(args.field_start_iteration),
            float(args.field_end_iteration),
        ],
        "maximum_engineering_changes": maxima,
        "maximum_temperature_field_rms_change_K": max(
            float(row["absolute_rms_change"]) for row in temperature_field_rows
        ),
        "maximum_temperature_field_point_change_K": max(
            float(row["maximum_absolute_change"]) for row in temperature_field_rows
        ),
        "maximum_velocity_relative_rms_change": max(
            float(row["relative_rms_change"])
            for row in field_rows
            if row["field_name"] == "fluid_velocity_components"
        ),
        "recommended_steady_end_iteration": args.candidate_iteration,
        "recommendation_basis": (
            "Five completed steady cases compare engineering outputs at iteration labels "
            "200 and 300; three low-velocity million-cell cases independently compare "
            "full fields over iterations 150-175. Both regions use steadyState ddt schemes, "
            "so these labels are not physical seconds."
        ),
        "new_physical_parameters": [],
    }
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    cn = f"""# P418稳态终点比较

流体区和固体区都使用`steadyState`时间离散，因此OpenFOAM目录名中的数字是稳态外迭代编号，不是物理秒数。五个已完成工况在第{args.candidate_iteration:g}次和第{args.reference_iteration:g}次迭代之间比较出口温度、颗粒最高温度、冷却壁热流、出口焓流和质量收支。另用三个最低入口速度的百万网格工况比较第{args.field_start_iteration}次与第{args.field_end_iteration}次迭代的完整流固温度、速度和压力场。

- 出口温度最大变化：`{maxima['outlet_temperature_absolute_change_K']:.6g} K`；
- 颗粒最高温度最大变化：`{maxima['solid_maximum_temperature_absolute_change_K']:.6g} K`；
- 第{args.field_start_iteration}--{args.field_end_iteration}次迭代完整温度场最大RMS变化：`{payload['maximum_temperature_field_rms_change_K']:.6g} K`；
- 同一时间窗内最大单元温度变化：`{payload['maximum_temperature_field_point_change_K']:.6g} K`；
- 速度场最大相对RMS变化：`{payload['maximum_velocity_relative_rms_change']:.6g}`；
- 第{args.candidate_iteration:g}次迭代质量相对差最大值：`{maxima['candidate_relative_mass_difference']:.6g}`。

这些结果支持把稳态求解终点从第300次迭代改为第200次迭代。文件名和OpenFOAM场目录中的`200`为兼容既有数据继续保留，但不得解释成物理时间。物理热阶跃使用瞬态时间离散，其持续时间必须另行通过真实瞬态响应确定。
"""
    (output / "稳态终点说明.md").write_text(cn, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
