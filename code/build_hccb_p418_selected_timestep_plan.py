#!/usr/bin/env python3
"""Write the 12-case P418 step plan with the finest declared numerical step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--sensitivity-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.base_plan.resolve().read_text(encoding="utf-8"))
    summary = json.loads(
        args.sensitivity_summary.resolve().read_text(encoding="utf-8")
    )
    if summary.get("status") != "completed_p418_thermal_timestep_sensitivity":
        raise ValueError("thermal time-step comparison is incomplete")
    selected = summary.get("selected_delta_t_s")
    if selected is None:
        raise ValueError("time-step comparison did not select a numerical step")
    base_root = args.base_plan.resolve().parents[1]
    declared_config = Path(plan["time_step_sensitivity_file"])
    if not declared_config.is_absolute():
        declared_config = base_root / declared_config
    config_path = Path(summary.get("config", declared_config))
    if not config_path.is_absolute():
        config_path = base_root / config_path
    if not config_path.is_file():
        if not declared_config.is_file():
            raise FileNotFoundError(
                f"neither recorded nor declared time-step config exists: "
                f"{config_path}, {declared_config}"
            )
        config_path = declared_config
    available = {
        float(value)
        for value in json.loads(
            config_path.resolve().read_text(encoding="utf-8")
        )["delta_t_s"]
    }
    if float(selected) not in available:
        raise ValueError("selected time step was not one of the predeclared calculations")
    if summary.get("formal_selection_rule") != "finest_completed_predeclared_step":
        raise ValueError("time-step summary does not use the declared finest-step rule")
    if float(selected) != min(available):
        raise ValueError("formal histories must use the finest predeclared time step")

    base_schedule = plan["numerical_time_design"].get("time_step_schedule")
    if not base_schedule:
        raise ValueError("base plan does not define the staged time-step schedule")
    base_initial = float(base_schedule[0]["delta_t_s"])
    scale = float(selected) / base_initial
    selected_schedule = [
        {**row, "delta_t_s": float(row["delta_t_s"]) * scale}
        for row in base_schedule
    ]
    sensitivity_schedule = summary.get("selected_time_step_schedule")
    if not sensitivity_schedule:
        raise ValueError("time-step comparison did not record its staged schedule")
    for expected, observed in zip(selected_schedule, sensitivity_schedule):
        if float(expected["start_s"]) >= float(sensitivity_schedule[-1]["end_s"]):
            break
        if any(
            abs(float(expected[key]) - float(observed[key])) > 1.0e-12
            for key in ("start_s", "end_s", "delta_t_s")
        ):
            raise ValueError("selected full-history schedule differs from the tested schedule")

    plan["numerical_time_design"]["delta_t_s"] = float(selected)
    plan["numerical_time_design"]["time_step_schedule"] = selected_schedule
    plan["numerical_time_design"]["selected_from_time_step_sensitivity"] = str(
        args.sensitivity_summary.resolve()
    )
    plan["numerical_time_design"]["selection_scope"] = summary["selection_scope"]
    plan["new_physical_parameters"] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "p418_formal_step_plan_uses_time_resolved_delta_t",
                "selected_delta_t_s": float(selected),
                "selected_time_step_schedule": selected_schedule,
                "output": str(args.output.resolve()),
                "new_physical_parameters": [],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
