#!/usr/bin/env python3
"""Build nested P418 training subsets for OpenFOAM data-efficiency tests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


AXIS_KEYS = ("velocity_index", "temperature_index", "source_index")
TRAINING_SIZES = (9, 18, 27, 36)


def normalized_coordinate(
    condition: dict[str, object], maxima: dict[str, int]
) -> tuple[float, ...]:
    return tuple(float(condition[key]) / float(maxima[key]) for key in AXIS_KEYS)


def select_nested_training_cases(
    conditions: dict[str, dict[str, object]], candidates: list[str]
) -> list[str]:
    """Select a deterministic, space-filling order while covering every level early."""
    maxima = {
        key: max(int(conditions[item][key]) for item in candidates) for key in AXIS_KEYS
    }
    coordinates = {
        item: normalized_coordinate(conditions[item], maxima) for item in candidates
    }
    selected: list[str] = []
    while len(selected) < len(candidates):
        covered = {
            key: {int(conditions[item][key]) for item in selected}
            for key in AXIS_KEYS
        }
        scored: list[tuple[tuple[object, ...], str]] = []
        for item in candidates:
            if item in selected:
                continue
            new_level_count = sum(
                int(conditions[item][key]) not in covered[key] for key in AXIS_KEYS
            )
            coordinate = coordinates[item]
            if selected:
                minimum_distance = min(
                    math.dist(coordinate, coordinates[chosen]) for chosen in selected
                )
            else:
                minimum_distance = -math.dist(coordinate, (0.5, 0.5, 0.5))
            centrality = tuple(-abs(value - 0.5) for value in coordinate)
            score = (new_level_count, minimum_distance, centrality)
            scored.append((score, item))
        best_score = max(score for score, _ in scored)
        selected.append(min(item for score, item in scored if score == best_score))
    return selected


def validate_level_coverage(
    *,
    conditions: dict[str, dict[str, object]],
    selected: list[str],
    reference: list[str],
) -> None:
    for key in AXIS_KEYS:
        expected = {int(conditions[item][key]) for item in reference}
        observed = {int(conditions[item][key]) for item in selected}
        if observed != expected:
            raise ValueError(
                f"training subset does not cover every {key} level; "
                f"expected={sorted(expected)}, observed={sorted(observed)}"
            )


def build_payload(base: dict[str, object]) -> dict[str, object]:
    conditions_list = list(base["conditions"])
    conditions = {
        str(item["condition_id"]): item for item in conditions_list
    }
    base_split = base["splits"]["interleaved_all_ranges"]
    base_train = [str(item) for item in base_split["train"]]
    validation = [str(item) for item in base_split["validation"]]
    test = [str(item) for item in base_split["test"]]
    all_ids = set(conditions)
    if set(base_train).union(validation, test) != all_ids:
        raise ValueError("interleaved_all_ranges must partition the full P418 matrix")
    if set(base_train).intersection(validation) or set(base_train).intersection(test):
        raise ValueError("base P418 train, validation and test conditions overlap")
    if set(validation).intersection(test):
        raise ValueError("base P418 validation and test conditions overlap")

    ordered = select_nested_training_cases(conditions, base_train)
    splits: dict[str, object] = {}
    previous: set[str] = set()
    for size in TRAINING_SIZES:
        if size > len(ordered):
            raise ValueError(f"requested training size {size} exceeds {len(ordered)}")
        train = ordered[:size]
        validate_level_coverage(
            conditions=conditions, selected=train, reference=base_train
        )
        if not previous.issubset(train):
            raise ValueError("learning-curve training subsets are not nested")
        previous = set(train)
        split_name = f"learning_curve_n{size:02d}"
        splits[split_name] = {
            "train": train,
            "validation": validation,
            "test": test,
            "unused": [item for item in base_train if item not in set(train)],
            "question": (
                f"With the same 12 validation and 12 test conditions, how much "
                f"accuracy is obtained from {size} literature-defined OpenFOAM "
                "training conditions?"
            ),
        }

    return {
        "source_parameter_id": base["source_parameter_id"],
        "source_title": base["source_title"],
        "source_doi": base["source_doi"],
        "condition_count": base["condition_count"],
        "conditions": conditions_list,
        "base_split": "interleaved_all_ranges",
        "training_sizes": list(TRAINING_SIZES),
        "selection_rule": (
            "nested deterministic space-filling subsets; first cover all velocity, "
            "temperature and heat-source levels, then maximize distance in normalized "
            "condition space"
        ),
        "physical_parameter_rule": (
            "No physical parameter is added or changed. All condition values are the "
            "P418 literature-defined matrix values."
        ),
        "splits": splits,
        "comparison_rule": (
            "Validation and test conditions are fixed across all training sizes; only "
            "the number of OpenFOAM training conditions changes."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base_splits.resolve().read_text(encoding="utf-8"))
    payload = build_payload(base)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "p418_learning_curve_splits_ready",
                "output": str(output),
                "training_sizes": list(TRAINING_SIZES),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
