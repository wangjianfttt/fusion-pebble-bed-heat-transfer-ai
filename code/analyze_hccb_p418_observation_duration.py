#!/usr/bin/env python3
"""Estimate the observation duration from a completed P418 step response."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


SIGNAL_ENDPOINTS = {
    "outlet_temperature_K": (
        ("temperature", "outlet_average_K"),
        "出口温度",
    ),
    "maximum_solid_temperature_K": (
        ("temperature", "solid_maximum_K"),
        "颗粒最高温度",
    ),
    "cooling_wall_power_W": (
        ("heat_balance", "cooling_wall_heat_flow_W"),
        "冷却壁换热量",
    ),
    "net_outward_enthalpy_flow_W": (
        ("heat_balance", "net_outward_enthalpy_flow_W"),
        "出口净焓流",
    ),
}


def nested_number(record: dict, path: tuple[str, str]) -> float:
    return float(record[path[0]][path[1]])


def log_tail_fit(
    time_s: np.ndarray,
    remaining_fraction: np.ndarray,
    start_s: float,
) -> dict:
    mask = (
        (time_s >= start_s)
        & np.isfinite(remaining_fraction)
        & (remaining_fraction > 1.0e-12)
        & (remaining_fraction < 1.0)
    )
    x = time_s[mask]
    y = np.log(remaining_fraction[mask])
    if x.size < 3:
        raise ValueError(f"not enough tail points after {start_s} s")
    slope, intercept = np.polyfit(x, y, 1)
    if slope >= 0.0:
        raise ValueError(f"tail does not decay after {start_s} s")
    fitted = intercept + slope * x
    residual = y - fitted
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / denominator

    def crossing_time(threshold: float) -> float:
        return float((math.log(threshold) - intercept) / slope)

    return {
        "fit_start_s": float(start_s),
        "fit_end_s": float(x[-1]),
        "point_count": int(x.size),
        "effective_time_constant_s": float(-1.0 / slope),
        "r_squared": r_squared,
        "estimated_time_to_1_percent_s": crossing_time(0.01),
        "estimated_time_to_0p1_percent_s": crossing_time(0.001),
    }


def build(args: argparse.Namespace) -> dict:
    history = np.load(args.history, allow_pickle=True)
    source = json.loads(args.source_summary.read_text(encoding="utf-8"))
    target = json.loads(args.target_summary.read_text(encoding="utf-8"))
    formal_plan = json.loads(args.formal_plan.read_text(encoding="utf-8"))
    plan_duration = float(formal_plan["numerical_time_design"]["duration_s"])
    if not math.isclose(plan_duration, args.candidate_duration_s):
        raise ValueError(
            f"candidate duration {args.candidate_duration_s} s does not match "
            f"formal plan duration {plan_duration} s"
        )

    time_s = np.asarray(history["time_s"][0], dtype=float)
    values = np.asarray(history["values"][0], dtype=float)
    signal_names = [str(value) for value in history["signal_names"]]
    if not np.all(np.diff(time_s) > 0.0):
        raise ValueError("history time must be strictly increasing")

    rows: list[dict] = []
    signal_results: list[dict] = []
    for signal, (endpoint_path, label_cn) in SIGNAL_ENDPOINTS.items():
        index = signal_names.index(signal)
        source_value = nested_number(source, endpoint_path)
        target_value = nested_number(target, endpoint_path)
        steady_step = target_value - source_value
        if steady_step == 0.0:
            raise ValueError(f"zero steady step for {signal}")
        remaining = np.abs((target_value - values[:, index]) / steady_step)
        fits = [log_tail_fit(time_s, remaining, start) for start in args.fit_starts_s]
        for fit in fits:
            rows.append({"signal": signal, "label_cn": label_cn, **fit})

        tau_values = np.array(
            [fit["effective_time_constant_s"] for fit in fits], dtype=float
        )
        one_percent = np.array(
            [fit["estimated_time_to_1_percent_s"] for fit in fits], dtype=float
        )
        point_one_percent = np.array(
            [fit["estimated_time_to_0p1_percent_s"] for fit in fits], dtype=float
        )
        signal_results.append(
            {
                "signal": signal,
                "label_cn": label_cn,
                "source_steady": source_value,
                "target_steady": target_value,
                "final_transient_value": float(values[-1, index]),
                "remaining_fraction_at_end": float(remaining[-1]),
                "fit_results": fits,
                "effective_time_constant_range_s": [
                    float(np.min(tau_values)),
                    float(np.max(tau_values)),
                ],
                "estimated_time_to_1_percent_range_s": [
                    float(np.min(one_percent)),
                    float(np.max(one_percent)),
                ],
                "estimated_time_to_0p1_percent_range_s": [
                    float(np.min(point_one_percent)),
                    float(np.max(point_one_percent)),
                ],
            }
        )

    representative_max_tau = max(
        row["effective_time_constant_range_s"][1] for row in signal_results
    )
    representative_max_t_0p1 = max(
        row["estimated_time_to_0p1_percent_range_s"][1]
        for row in signal_results
    )
    velocity_scale = args.reference_velocity_m_s / args.minimum_velocity_m_s
    conservative_tau = representative_max_tau * velocity_scale
    conservative_t_0p1 = representative_max_t_0p1 * velocity_scale
    remaining_at_candidate = math.exp(-args.candidate_duration_s / conservative_tau)
    schedule = formal_plan["numerical_time_design"]["time_step_schedule"]
    write_schedule = formal_plan["numerical_time_design"]["field_write_schedule"]
    formal_step_count = int(
        round(
            sum(
                (float(row["end_s"]) - float(row["start_s"]))
                / float(row["delta_t_s"])
                for row in schedule
            )
        )
    )
    formal_full_field_count = 1 + int(
        round(
            sum(
                (float(row["end_s"]) - float(row["start_s"]))
                / float(row["interval_s"])
                for row in write_schedule
            )
        )
    )
    reference_step_count = int(time_s.size - 1)
    estimated_wall_time_s = (
        args.fine_reference_wall_time_s * formal_step_count / reference_step_count
    )
    estimated_final_bytes_per_case = (
        args.fine_reference_step_dir_bytes
        * formal_full_field_count
        / args.fine_reference_full_field_count
    )

    result = {
        "status": "completed_p418_observation_duration_analysis",
        "analysis_scope": (
            "Tail fits use one completed fixed-hydrodynamics heat-source step. "
            "The minimum-velocity estimate scales the slowest fitted time constant "
            "by u_ref/u_min; it is a conservative design estimate, not a new "
            "material parameter or a replacement for the formal low-velocity runs."
        ),
        "history": str(args.history),
        "source_summary": str(args.source_summary),
        "target_summary": str(args.target_summary),
        "history_end_s": float(time_s[-1]),
        "fit_starts_s": [float(value) for value in args.fit_starts_s],
        "reference_velocity_m_s": args.reference_velocity_m_s,
        "minimum_matrix_velocity_m_s": args.minimum_velocity_m_s,
        "velocity_time_scale_factor": velocity_scale,
        "candidate_duration_s": args.candidate_duration_s,
        "signals": signal_results,
        "representative_slowest_time_constant_s": representative_max_tau,
        "representative_latest_estimated_time_to_0p1_percent_s": (
            representative_max_t_0p1
        ),
        "conservative_minimum_velocity_time_constant_s": conservative_tau,
        "conservative_minimum_velocity_time_to_0p1_percent_s": conservative_t_0p1,
        "single_exponential_remaining_fraction_at_candidate_duration": (
            remaining_at_candidate
        ),
        "resource_estimate": {
            "basis": (
                "The completed fine 25 s job used 32 MPI ranks for 07:10:00 and "
                "left 8,480,487,842 bytes after processor cleanup. Wall time is "
                "scaled by the declared time-step count; final result size is "
                "scaled by the declared number of full-field writes."
            ),
            "mpi_ranks_per_case": args.mpi_ranks_per_case,
            "reference_step_count": reference_step_count,
            "formal_step_count": formal_step_count,
            "reference_full_field_count": args.fine_reference_full_field_count,
            "formal_full_field_count": formal_full_field_count,
            "estimated_wall_time_s_per_case": estimated_wall_time_s,
            "estimated_wall_time_h_per_case": estimated_wall_time_s / 3600.0,
            "formal_sequence_count": args.formal_sequence_count,
            "estimated_total_core_hours": (
                estimated_wall_time_s
                / 3600.0
                * args.mpi_ranks_per_case
                * args.formal_sequence_count
            ),
            "estimated_final_bytes_per_case": estimated_final_bytes_per_case,
            "estimated_final_bytes_all_cases": (
                estimated_final_bytes_per_case * args.formal_sequence_count
            ),
            "estimate_is_not_a_measured_formal_runtime": True,
        },
        "interpretation": (
            "The 25 s representative run captures the main response but not the "
            "full tail. A 300 s observation window exceeds the conservatively "
            "velocity-scaled 0.1% time by a substantial margin. The formal "
            "low-velocity histories must still be checked directly."
        ),
        "new_physical_parameters": [],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "tail_fit_windows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = f"""# P418正式热阶跃观察时长分析

代表工况为`u={args.reference_velocity_m_s:g} m/s`、`T_in=700 K`、颗粒发热率由`4.85`升至`8.85 MW/m3`的固定流场热阶跃。

- 25 s曲线尾段的最慢有效时间常数为`{representative_max_tau:.2f} s`。
- 不同尾段拟合窗口给出的最晚`0.1%`剩余量时间为`{representative_max_t_0p1:.1f} s`。
- 按最低流速`{args.minimum_velocity_m_s:g} m/s`相对代表流速作保守的`1/u`缩放，最慢时间常数为`{conservative_tau:.1f} s`，`0.1%`剩余量时间为`{conservative_t_0p1:.1f} s`。
- 候选观察时长`{args.candidate_duration_s:g} s`约为该保守时间常数的`{args.candidate_duration_s / conservative_tau:.1f}`倍；单指数估计剩余量为`{100.0 * remaining_at_candidate:.4f}%`。
- 按已完成细档的时间步数和场输出量缩放，300 s单条约需`{estimated_wall_time_s / 3600.0:.1f} h`、清理并行分区后约`{estimated_final_bytes_per_case / 1.0e9:.1f} GB`；12条约为`{estimated_wall_time_s / 3600.0 * args.mpi_ranks_per_case * args.formal_sequence_count:.0f}`核时和`{estimated_final_bytes_per_case * args.formal_sequence_count / 1.0e9:.0f} GB`最终结果。这里是资源估算，不是正式运行结果。

因此，25 s适合展示主要响应和时间步收敛，但不适合作为全部低流速正式曲线的统一终点。`300 s`是更稳妥的统一观察时长。这个结论来自已算曲线和最低流速缩放，没有增加材料参数；正式低流速曲线完成后仍需直接检查尾段。
"""
    (args.output_dir / "观察时长说明_CN.md").write_text(report, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    base = ROOT / "results/hccb_p418_fixed_timestep_source_up_u0p15_T700_v7"
    parser.add_argument(
        "--history",
        type=Path,
        default=base / "dt_1em05/hccb_p418_transient_observables.npz",
    )
    parser.add_argument(
        "--source-summary",
        type=Path,
        default=base / "source_summary_steady_iteration_200.json",
    )
    parser.add_argument(
        "--target-summary",
        type=Path,
        default=base / "target_summary_steady_iteration_200.json",
    )
    parser.add_argument(
        "--fit-starts-s",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 15.0, 20.0],
    )
    parser.add_argument("--reference-velocity-m-s", type=float, default=0.15)
    parser.add_argument("--minimum-velocity-m-s", type=float, default=0.05)
    parser.add_argument("--candidate-duration-s", type=float, default=300.0)
    parser.add_argument(
        "--formal-plan",
        type=Path,
        default=ROOT
        / "results/hccb_p418_thermal_timestep_sensitivity/formal_step_plan_candidate_300s.json",
    )
    parser.add_argument("--fine-reference-wall-time-s", type=float, default=25800.0)
    parser.add_argument(
        "--fine-reference-step-dir-bytes", type=float, default=8_480_487_842
    )
    parser.add_argument("--fine-reference-full-field-count", type=int, default=45)
    parser.add_argument("--formal-sequence-count", type=int, default=12)
    parser.add_argument("--mpi-ranks-per-case", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/hccb_p418_observation_duration",
    )
    args = parser.parse_args()
    result = build(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
