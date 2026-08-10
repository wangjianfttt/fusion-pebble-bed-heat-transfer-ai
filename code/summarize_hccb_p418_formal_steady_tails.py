#!/usr/bin/env python3
"""Combine measured final-window changes for the formal P418 steady matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(change: dict[str, object], name: str, kind: str) -> float:
    return float(change[name][kind])  # type: ignore[index]


def summarize(
    matrix: Path,
    expected: int,
    minimum_full_fields: int,
    allow_partial: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    marker_paths = sorted(matrix.glob("*/formal_sample_complete.json"))
    if allow_partial and not 0 < len(marker_paths) <= expected:
        raise ValueError(f"formal steady markers: {len(marker_paths)}, expected 1--{expected}")
    if not allow_partial and len(marker_paths) != expected:
        raise ValueError(f"formal steady markers: {len(marker_paths)}, expected {expected}")
    rows: list[dict[str, object]] = []
    full_field_count = 0
    for marker_path in marker_paths:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("steady_final_window_status") != "formal_steady_final_window_measured":
            raise ValueError(f"final-window measurement missing from {marker_path}")
        summary_path = Path(str(marker["steady_final_window_summary"]))
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        if sha256(summary_path) != marker.get("steady_final_window_summary_sha256"):
            raise ValueError(f"final-window checksum differs: {summary_path}")
        document = json.loads(summary_path.read_text(encoding="utf-8"))
        if document.get("status") != "formal_steady_final_window_measured":
            raise ValueError(f"invalid final-window summary: {summary_path}")
        changes = document["engineering_changes"]
        field_rows = {row["field_name"]: row for row in document["full_field_changes"]}
        full = bool(document["full_field_available"])
        full_field_count += int(full)
        window = document.get("window_iterations", document.get("window_s"))
        if not isinstance(window, list) or len(window) != 2:
            raise ValueError(f"invalid steady-iteration window: {summary_path}")
        row: dict[str, object] = {
            "condition_id": marker_path.parent.name,
            "window_start_iteration": float(window[0]),
            "window_end_iteration": float(window[1]),
            "full_field_available": full,
            "outlet_temperature_change_K": metric(changes, "outlet_temperature_K", "absolute_change"),
            "solid_maximum_temperature_change_K": metric(changes, "solid_maximum_temperature_K", "absolute_change"),
            "cooling_wall_power_relative_change": metric(changes, "cooling_wall_power_W", "relative_change"),
            "outlet_enthalpy_flow_relative_change": metric(changes, "outlet_enthalpy_flow_W", "relative_change"),
            "pressure_drop_relative_change": metric(changes, "pressure_drop_Pa", "relative_change"),
            "final_relative_mass_difference": float(changes["final_relative_mass_difference"]),
            "temperature_field_rms_change_K": document["maximum_temperature_field_rms_change_K"],
            "temperature_field_point_change_K": document["maximum_temperature_field_point_change_K"],
            "velocity_field_relative_rms_change": (
                float(field_rows["fluid_velocity_components"]["relative_rms_change"])
                if "fluid_velocity_components" in field_rows
                else None
            ),
            "pressure_field_relative_rms_change": (
                float(field_rows["fluid_pressure"]["relative_rms_change"])
                if "fluid_pressure" in field_rows
                else None
            ),
        }
        numeric = [float(value) for value in row.values() if isinstance(value, (int, float))]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"non-finite final-window row: {marker_path.parent.name}")
        rows.append(row)
    if full_field_count < minimum_full_fields:
        raise ValueError(
            f"full-field final-window cases: {full_field_count}, required {minimum_full_fields}"
        )

    def maximum(key: str, required: bool = True) -> float | None:
        values = [float(row[key]) for row in rows if row[key] is not None]
        if not values:
            if required:
                raise ValueError(f"no values for {key}")
            return None
        return max(values)

    complete = len(rows) == expected
    summary = {
        "status": (
            "formal_steady_final_windows_ready"
            if complete
            else "partial_steady_final_windows_ready"
        ),
        "expected_case_count": expected,
        "case_count": len(rows),
        "full_field_case_count": full_field_count,
        "minimum_full_field_case_count": minimum_full_fields,
        "scalar_history_case_count": len(rows),
        "solver_time_semantics": "steady_iteration_index",
        "physical_time_s": None,
        "window_iterations": sorted(
            {
                (row["window_start_iteration"], row["window_end_iteration"])
                for row in rows
            }
        ),
        "maximum_changes": {
            "outlet_temperature_K": maximum("outlet_temperature_change_K"),
            "solid_maximum_temperature_K": maximum("solid_maximum_temperature_change_K"),
            "cooling_wall_power_relative": maximum("cooling_wall_power_relative_change"),
            "outlet_enthalpy_flow_relative": maximum("outlet_enthalpy_flow_relative_change"),
            "pressure_drop_relative": maximum("pressure_drop_relative_change"),
            "relative_mass_difference": maximum("final_relative_mass_difference"),
            "temperature_field_rms_K": maximum(
                "temperature_field_rms_change_K", required=not allow_partial
            ),
            "temperature_field_point_K": maximum(
                "temperature_field_point_change_K", required=not allow_partial
            ),
            "velocity_field_relative_rms": maximum(
                "velocity_field_relative_rms_change", required=not allow_partial
            ),
            "pressure_field_relative_rms": maximum(
                "pressure_field_relative_rms_change", required=not allow_partial
            ),
        },
        "interpretation": (
            "Measured changes over each case's final 25 saved steady iterations. The "
            "iteration labels are not physical seconds. Values are reported directly; no "
            "fitted physical parameter or arbitrary convergence threshold is added."
        ),
        "new_physical_parameters": [],
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--expected-case-count", type=int, default=60)
    parser.add_argument("--minimum-full-field-count", type=int, default=56)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latex-output", type=Path)
    args = parser.parse_args()
    rows, summary = summarize(
        args.matrix_root.resolve(),
        args.expected_case_count,
        args.minimum_full_field_count,
        args.allow_partial,
    )
    if args.allow_partial and args.latex_output:
        raise ValueError("partial final-window results cannot be written into the manuscript")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "formal_steady_final_windows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    maxima = summary["maximum_changes"]
    field_lines = (
        f"- 完整温度场最大RMS变化：`{maxima['temperature_field_rms_K']:.6g} K`；\n"
        f"- 完整温度场最大单元变化：`{maxima['temperature_field_point_K']:.6g} K`；"
        if maxima["temperature_field_rms_K"] is not None
        else "- 当前完成工况尚无可保留的完整末段场；从第5个工况起将逐组加入。"
    )
    scope_text = (
        "60个正式三维共轭换热工况"
        if summary["case_count"] == summary["expected_case_count"]
        else f"当前已完成的{summary['case_count']}/{summary['expected_case_count']}个正式三维共轭换热工况"
    )
    full_field_missing_count = summary["case_count"] - summary["full_field_case_count"]
    text = f"""# P418正式稳态工况末段变化

{scope_text}都比较最后25次保存的稳态迭代内，出口温度、颗粒最高温度、压降、壁面热量、出口焓流和质量收支的变化。这里的迭代编号不是物理秒数。其中{summary['full_field_case_count']}个工况还保存了成对的175次和200次完整三维场，可直接比较流体温度、颗粒温度、速度和压力场。其余{full_field_missing_count}个工况保留了第200次完整终态和工程量末段历史，但没有可配对的175次完整场，因此不对这些工况报告三维场末段变化，也不根据已有工况外推。

- 出口温度最大变化：`{maxima['outlet_temperature_K']:.6g} K`；
- 颗粒最高温度最大变化：`{maxima['solid_maximum_temperature_K']:.6g} K`；
{field_lines}
- 压降最大相对变化：`{maxima['pressure_drop_relative']:.6g}`；
- 质量收支最大相对差：`{maxima['relative_mass_difference']:.6g}`。

这些数值直接描述正式算例末段仍在发生的变化，不引入新的材料参数，也不人为拟合一个收敛比例。它们将与时间步和网格比较一起说明稳态场是否适合进入网络训练。
"""
    (output / "P418_正式稳态末段变化_CN.md").write_text(text, encoding="utf-8")
    if args.latex_output:
        latex_output = args.latex_output.resolve()
        latex_output.parent.mkdir(parents=True, exist_ok=True)
        latex_output.write_text(
            "Across all 60 corrected-source-flow steady cases, the final "
            f"25 saved steady iterations changed the outlet temperature by at most "
            f"\\SI{{{maxima['outlet_temperature_K']:.3g}}}{{K}} and the maximum solid "
            f"temperature by at most \\SI{{{maxima['solid_maximum_temperature_K']:.3g}}}{{K}}; "
            f"among the {summary['full_field_case_count']} cases with paired full fields, the "
            f"largest pointwise temperature change was "
            f"\\SI{{{maxima['temperature_field_point_K']:.3g}}}{{K}}.\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
