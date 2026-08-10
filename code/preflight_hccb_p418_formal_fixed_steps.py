#!/usr/bin/env python3
"""Check the P418 formal fixed-hydrodynamics transient plan without solving it."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expand_times(stages: list[dict], interval_key: str) -> list[float]:
    times: set[float] = set()
    previous_end = 0.0
    for stage in stages:
        start = float(stage["start_s"])
        end = float(stage["end_s"])
        interval = float(stage[interval_key])
        if abs(start - previous_end) > 1.0e-12:
            raise ValueError(f"non-contiguous stage begins at {start}, expected {previous_end}")
        if end <= start or interval <= 0.0:
            raise ValueError(f"invalid stage: {stage}")
        count = int(round((end - start) / interval))
        if abs(start + count * interval - end) > 1.0e-10:
            raise ValueError(f"stage is not divisible by {interval_key}: {stage}")
        times.update(round(start + index * interval, 12) for index in range(count + 1))
        previous_end = end
    return sorted(times)


def build_preflight(
    plan_path: Path,
    endpoint_path: Path,
    observation_path: Path,
    timestep_path: Path,
) -> dict:
    plan = load_json(plan_path)
    endpoints = load_json(endpoint_path)
    observation = load_json(observation_path)
    timestep = load_json(timestep_path)

    sequences = plan["sequences"]
    family_counts = Counter(row["family"] for row in sequences)
    unique_endpoints = {
        row[key]
        for row in sequences
        for key in ("source_condition_id", "target_condition_id")
    }
    endpoint_sequences = {
        row["sequence_id"] for row in endpoints.get("sequences", [])
    }
    planned_sequences = {row["sequence_id"] for row in sequences}

    numerical = plan["numerical_time_design"]
    duration = float(numerical["duration_s"])
    time_stages = numerical["time_step_schedule"]
    field_stages = numerical["field_write_schedule"]
    numerical_times = expand_times(time_stages, "delta_t_s")
    snapshot_times = expand_times(field_stages, "interval_s")

    checks = {
        "sequence_count_is_12": len(sequences) == 12,
        "three_families_have_four_sequences_each": dict(family_counts)
        == {
            "inlet_temperature_step": 4,
            "inlet_velocity_step": 4,
            "solid_heat_source_step": 4,
        },
        "unique_endpoint_count_is_11": len(unique_endpoints) == 11,
        "endpoint_sequence_set_matches_plan": endpoint_sequences == planned_sequences,
        "endpoint_check_reports_12_sequences": endpoints.get("sequence_count") == 12,
        "endpoint_check_adds_no_physical_parameters": not endpoints.get(
            "new_physical_parameters"
        ),
        "plan_adds_no_physical_parameters": not plan.get("new_physical_parameters"),
        "duration_is_300_s": duration == 300.0,
        "time_step_schedule_ends_at_duration": numerical_times[-1] == duration,
        "field_write_schedule_ends_at_duration": snapshot_times[-1] == duration,
        "full_field_snapshot_count_is_56": len(snapshot_times) == 56,
        "observation_analysis_supports_300_s": float(
            observation["candidate_duration_s"]
        )
        == duration,
        "representative_timestep_result_is_complete": timestep.get("status")
        == "completed_p418_thermal_timestep_sensitivity",
        "selected_early_schedule_matches_time_step_result": time_stages[:3]
        == timestep["selected_time_step_schedule"],
        "observation_analysis_adds_no_physical_parameters": not observation.get(
            "new_physical_parameters"
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("formal fixed-step preflight failed: " + ", ".join(failed))

    resources = observation["resource_estimate"]
    return {
        "status": "ready_for_300_s_duration_approval_not_submitted",
        "scientific_state": (
            "The 25 s representative response and three-level time-step comparison are "
            "complete. The 300 s duration is supported as a conservative observation "
            "window but remains a sample-design decision until explicitly approved."
        ),
        "sequence_count": len(sequences),
        "family_counts": dict(family_counts),
        "unique_endpoint_count": len(unique_endpoints),
        "unique_endpoint_ids": sorted(unique_endpoints),
        "duration_s": duration,
        "time_step_schedule": time_stages,
        "field_write_schedule": field_stages,
        "numerical_step_count": len(numerical_times) - 1,
        "full_field_snapshot_count": len(snapshot_times),
        "first_full_field_time_s": snapshot_times[0],
        "last_full_field_time_s": snapshot_times[-1],
        "observation_basis": {
            "representative_slowest_time_constant_s": observation[
                "representative_slowest_time_constant_s"
            ],
            "conservative_minimum_velocity_time_constant_s": observation[
                "conservative_minimum_velocity_time_constant_s"
            ],
            "conservative_time_to_0p1_percent_s": observation[
                "conservative_minimum_velocity_time_to_0p1_percent_s"
            ],
            "estimated_remaining_fraction_at_300_s": observation[
                "single_exponential_remaining_fraction_at_candidate_duration"
            ],
        },
        "resource_estimate": resources,
        "checks": checks,
        "input_sha256": {
            "plan": sha256(plan_path),
            "endpoint_readiness": sha256(endpoint_path),
            "observation_duration": sha256(observation_path),
            "timestep_sensitivity": sha256(timestep_path),
        },
        "new_physical_parameters": [],
        "formal_solver_submitted": False,
    }


def chinese_summary(payload: dict) -> str:
    resource = payload["resource_estimate"]
    observation = payload["observation_basis"]
    return f"""# P418正式固定流场热阶跃提交前检查

- 12条曲线和11个稳态端点全部对应一致。
- 三类过程各4条：入口温度阶跃、入口速度阶跃和颗粒发热率阶跃。
- 计划观察时长为300物理秒，共{payload['numerical_step_count']}个数值时间步、{payload['full_field_snapshot_count']}个完整三维场时刻。
- 代表工况最慢时间常数为{observation['representative_slowest_time_constant_s']:.2f} s。
- 按最低流速保守估计，达到剩余0.1%约需{observation['conservative_time_to_0p1_percent_s']:.1f} s；300 s时估计剩余量为{100.0 * observation['estimated_remaining_fraction_at_300_s']:.6f}%。
- 按已完成细时间步工况估算，单条约{resource['estimated_wall_time_h_per_case']:.1f}小时、32 MPI，12条约{resource['estimated_total_core_hours']:.0f}核时。
- 清理并行分区目录后，预计单条约{resource['estimated_final_bytes_per_case'] / 1.0e9:.1f} GB，12条约{resource['estimated_final_bytes_all_cases'] / 1.0e9:.0f} GB。
- 上述运行时间和存储量是估算，不是正式12条工况的实测结果。
- 本检查没有启动OpenFOAM。正式计算仍等待300 s观察时长获得明确批准。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "parameters/hccb_p418_transient_step_plan.json",
    )
    parser.add_argument(
        "--endpoint-readiness",
        type=Path,
        default=ROOT
        / "results/hccb_p418_step_endpoint_readiness_60/endpoint_field_check.json",
    )
    parser.add_argument(
        "--observation-duration",
        type=Path,
        default=ROOT / "results/hccb_p418_observation_duration/summary.json",
    )
    parser.add_argument(
        "--timestep-sensitivity",
        type=Path,
        default=ROOT
        / "results/hccb_p418_thermal_timestep_sensitivity/thermal_timestep_sensitivity.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/hccb_p418_formal_fixed_steps_preflight",
    )
    args = parser.parse_args()

    payload = build_preflight(
        args.plan.resolve(),
        args.endpoint_readiness.resolve(),
        args.observation_duration.resolve(),
        args.timestep_sensitivity.resolve(),
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preflight.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "提交前检查_CN.md").write_text(
        chinese_summary(payload),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
