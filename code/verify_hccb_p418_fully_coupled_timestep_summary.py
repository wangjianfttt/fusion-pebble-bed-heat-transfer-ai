#!/usr/bin/env python3
"""Verify that the representative fully coupled time-step study matches the formal plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def normalized_schedule(rows: list[dict[str, object]]) -> list[dict[str, float]]:
    return [
        {
            "start_s": float(row["start_s"]),
            "end_s": float(row["end_s"]),
            "delta_t_s": float(row["delta_t_s"]),
        }
        for row in rows
    ]


def verify_summary(
    summary_path: Path,
    config_path: Path,
    plan_path: Path,
    root: Path = ROOT,
) -> dict[str, object]:
    summary = json.loads(summary_path.resolve().read_text(encoding="utf-8"))
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    plan = json.loads(plan_path.resolve().read_text(encoding="utf-8"))
    base_plan_path = root.resolve() / plan["base_thermal_step_plan"]
    base_plan = json.loads(base_plan_path.read_text(encoding="utf-8"))

    sequence = next(
        (
            row
            for row in plan["sequences"]
            if row["sequence_id"] == config["sequence_id"]
        ),
        None,
    )
    if sequence is None:
        raise ValueError("representative time-step sequence is absent from the formal plan")
    if sequence["family"] != "inlet_velocity_step":
        raise ValueError("representative time-step sequence is not a velocity step")
    for key in ("source_condition_id", "target_condition_id"):
        if config[key] != sequence[key]:
            raise ValueError(f"representative time-step {key} changed")

    declared_steps = [float(value) for value in config["delta_t_s"]]
    if len(declared_steps) != 3 or any(value <= 0.0 for value in declared_steps):
        raise ValueError("fully coupled time-step study must contain three positive steps")
    refinement_ratio = float(
        config["discretization_uncertainty_method"]["refinement_ratio"]
    )
    if not all(
        abs(coarse / fine - refinement_ratio) <= 1.0e-12
        for coarse, fine in zip(declared_steps[:-1], declared_steps[1:])
    ):
        raise ValueError("fully coupled time-step refinement ratio changed")

    formal_schedule = normalized_schedule(config["formal_time_step_schedule"])
    base_schedule = normalized_schedule(
        base_plan["numerical_time_design"]["time_step_schedule"][: len(formal_schedule)]
    )
    if formal_schedule != base_schedule:
        raise ValueError(
            "representative time-step schedule no longer matches the formal 300 s plan"
        )
    if formal_schedule[-1]["end_s"] != float(config["duration_s"]):
        raise ValueError("representative time-step duration and schedule end differ")

    if summary.get("status") != "completed_p418_fully_coupled_timestep_sensitivity":
        raise ValueError("fully coupled time-step study has the wrong status")
    if summary.get("analysis_kind") != "fully_coupled_flow_heat":
        raise ValueError("fully coupled time-step study has the wrong analysis kind")
    if summary.get("sequence_id") != config["sequence_id"]:
        raise ValueError("fully coupled time-step summary used another sequence")
    if [float(value) for value in summary.get("delta_t_s", [])] != sorted(
        declared_steps, reverse=True
    ):
        raise ValueError("fully coupled time-step summary used another resolution set")
    if summary.get("formal_selection_rule") != config["formal_selection_rule"]:
        raise ValueError("fully coupled time-step selection rule changed")
    if float(summary.get("selected_delta_t_s", -1.0)) != min(declared_steps):
        raise ValueError("formal fully coupled step is not the finest declared step")
    if normalized_schedule(summary.get("selected_time_step_schedule", [])) != (
        formal_schedule
    ):
        raise ValueError("selected fully coupled schedule differs from the formal schedule")
    if summary.get("new_physical_parameters") != []:
        raise ValueError("time-step comparison introduced a physical parameter")

    metadata_path = root.resolve() / config["discretization_uncertainty_method"][
        "source_metadata"
    ]
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    return {
        "status": "verified_p418_fully_coupled_timestep_summary",
        "sequence_id": config["sequence_id"],
        "source_condition_id": config["source_condition_id"],
        "target_condition_id": config["target_condition_id"],
        "selected_delta_t_s": min(declared_steps),
        "selected_time_step_schedule": formal_schedule,
        "formal_duration_s": float(base_plan["numerical_time_design"]["duration_s"]),
        "new_physical_parameters": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "parameters/hccb_p418_fully_coupled_timestep_sensitivity.json",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "parameters/hccb_p418_fully_coupled_step_plan.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_summary(args.summary, args.config, args.plan)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
