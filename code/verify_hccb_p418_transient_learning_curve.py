#!/usr/bin/env python3
"""Check the P418 transient small-sample comparison without loading field data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_COUNTS = {
    "transient_learning_n03_up": (3, 2, 4, 3),
    "transient_learning_n03_down": (3, 2, 4, 3),
    "transient_learning_n06_both": (6, 2, 4, 0),
}
ROLES = ("train", "validation", "test", "unused")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_learning_curve_plan(
    step_plan_path: Path, split_path: Path, runner_path: Path
) -> dict[str, object]:
    plan = load_json(step_plan_path)
    comparison = load_json(split_path)
    records = {
        str(row["sequence_id"]): row for row in plan["sequences"]  # type: ignore[index]
    }
    expected_ids = set(records)
    if len(expected_ids) != 12:
        raise ValueError(f"expected 12 unique physical curves, found {len(expected_ids)}")
    if plan.get("source_parameter_id") != "P418":
        raise ValueError("step plan is not tied to P418")
    if comparison.get("source_parameter_id") != "P418":
        raise ValueError("learning-curve plan is not tied to P418")
    if comparison.get("new_physical_parameters") != []:
        raise ValueError("learning-curve plan introduces new physical parameters")

    fixed_validation = list(comparison["fixed_validation"])  # type: ignore[index]
    fixed_test = list(comparison["fixed_test"])  # type: ignore[index]
    summaries: dict[str, object] = {}
    for name, expected_counts in EXPECTED_COUNTS.items():
        split = comparison["splits"][name]  # type: ignore[index]
        groups = {role: [str(value) for value in split.get(role, [])] for role in ROLES}
        actual_counts = tuple(len(groups[role]) for role in ROLES)
        if actual_counts != expected_counts:
            raise ValueError(
                f"{name} counts {actual_counts} differ from {expected_counts}"
            )
        for role, values in groups.items():
            if len(values) != len(set(values)):
                raise ValueError(f"{name} repeats a curve inside {role}")
        for left, left_role in enumerate(ROLES):
            for right_role in ROLES[left + 1 :]:
                overlap = set(groups[left_role]) & set(groups[right_role])
                if overlap:
                    raise ValueError(
                        f"{name} reuses complete curves across roles: {sorted(overlap)}"
                    )
        covered = set().union(*(set(values) for values in groups.values()))
        if covered != expected_ids:
            raise ValueError(
                f"{name} does not cover the same 12 physical curves: "
                f"missing={sorted(expected_ids-covered)}, extra={sorted(covered-expected_ids)}"
            )
        if groups["validation"] != fixed_validation or groups["test"] != fixed_test:
            raise ValueError(f"{name} changes the fixed validation or test curves")

        train_families = {str(records[item]["family"]) for item in groups["train"]}
        if train_families != {
            "inlet_temperature_step",
            "inlet_velocity_step",
            "solid_heat_source_step",
        }:
            raise ValueError(f"{name} does not cover all three physical step families")
        if name.endswith("_up") and any("_up_" not in item for item in groups["train"]):
            raise ValueError(f"{name} contains a non-upward training curve")
        if name.endswith("_down") and any(
            "_down_" not in item for item in groups["train"]
        ):
            raise ValueError(f"{name} contains a non-downward training curve")

        summaries[name] = {
            "train_curve_count": len(groups["train"]),
            "validation_curve_count": len(groups["validation"]),
            "test_curve_count": len(groups["test"]),
            "unused_curve_count": len(groups["unused"]),
            "train_families": sorted(train_families),
        }

    runner = runner_path.read_text(encoding="utf-8")
    required_runner_text = (
        "EXECUTE=${EXECUTE:-0}",
        "PAUSE_NEW_P418_CASES_FOR_CLOUD_MIGRATION",
        "--run-role formal_factorized",
        "--physics-mode energy_and_flux",
        "--spatial-temporal-mode factorized_static_spatial",
    )
    for text in required_runner_text:
        if text not in runner:
            raise ValueError(f"learning-curve runner is missing: {text}")
    forbidden_runner_text = (
        "train_hccb_p418_low_rank_temperature_residual.py",
        "train_hccb_p418_temporal_temperature_diffusion.py",
    )
    for text in forbidden_runner_text:
        if text in runner:
            raise ValueError(f"learning-curve runner must not start correction model: {text}")

    return {
        "status": "passed",
        "physical_curve_count": len(expected_ids),
        "independent_training_curve_counts": [3, 6],
        "fixed_validation_curve_count": len(fixed_validation),
        "fixed_test_curve_count": len(fixed_test),
        "field_observations_are_not_independent_conditions": True,
        "new_physical_parameters": [],
        "splits": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step-plan",
        type=Path,
        default=Path("parameters/hccb_p418_transient_step_plan.json"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("parameters/hccb_p418_transient_learning_curve_splits.json"),
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("code/run_hccb_p418_transient_learning_curve.sh"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check_learning_curve_plan(args.step_plan, args.splits, args.runner)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
