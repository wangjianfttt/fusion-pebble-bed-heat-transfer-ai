#!/usr/bin/env python3
"""Create common model-comparison splits from the exact P418 condition matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from build_hccb_dense_cht_p418_matrix import MANIFEST, all_conditions


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "parameters/hccb_p418_model_splits.json"


def rows() -> dict[str, dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return {row["parameter_id"]: row for row in csv.DictReader(handle)}


def records() -> list[dict[str, object]]:
    p418 = rows()["P418"]
    if p418["status"] != "extracted":
        raise RuntimeError("P418 is not an extracted literature condition matrix")
    conditions = all_conditions(p418["value"])
    velocities = sorted({values[0] for values in conditions.values()})
    temperatures = sorted({values[1] for values in conditions.values()})
    sources = sorted({values[2] for values in conditions.values()})
    output = []
    for condition_id, (velocity, temperature, source) in sorted(conditions.items()):
        output.append(
            {
                "condition_id": condition_id,
                "inlet_velocity_m_s": velocity,
                "inlet_temperature_K": temperature,
                "solid_heat_source_MW_m3": source,
                "velocity_index": velocities.index(velocity),
                "temperature_index": temperatures.index(temperature),
                "source_index": sources.index(source),
            }
        )
    if len(output) != 60:
        raise RuntimeError(f"P418 should contain 60 conditions, found {len(output)}")
    return output


def ids(items: list[dict[str, object]]) -> list[str]:
    return [str(item["condition_id"]) for item in items]


def make_splits(items: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    interleaved = {"train": [], "validation": [], "test": []}
    for item in items:
        score = (
            int(item["velocity_index"])
            + 2 * int(item["temperature_index"])
            + 3 * int(item["source_index"])
        ) % 5
        group = "test" if score == 0 else "validation" if score == 1 else "train"
        interleaved[group].append(str(item["condition_id"]))

    def value_split(field: str, train_values: set[float], validation: float, test: float):
        return {
            "train": ids([item for item in items if float(item[field]) in train_values]),
            "validation": ids([item for item in items if float(item[field]) == validation]),
            "test": ids([item for item in items if float(item[field]) == test]),
        }

    heat_interpolation_validation = []
    heat_interpolation_train = []
    heat_interpolation_test = []
    for item in items:
        source_index = int(item["source_index"])
        if source_index == 1:
            heat_interpolation_test.append(str(item["condition_id"]))
        else:
            score = (
                int(item["velocity_index"])
                + 2 * int(item["temperature_index"])
                + source_index
            ) % 5
            target = heat_interpolation_validation if score == 0 else heat_interpolation_train
            target.append(str(item["condition_id"]))

    return {
        "interleaved_all_ranges": {
            **interleaved,
            "question": "Prediction at unseen combinations while every published value appears in training.",
        },
        "temperature_extrapolation": {
            **value_split("inlet_temperature_K", {300.0, 500.0}, 700.0, 900.0),
            "question": "Prediction at the highest published inlet temperature.",
        },
        "velocity_extrapolation": {
            **value_split("inlet_velocity_m_s", {0.05, 0.10, 0.15}, 0.20, 0.25),
            "question": "Prediction at the highest published inlet velocity.",
        },
        "heat_source_interpolation": {
            "train": heat_interpolation_train,
            "validation": heat_interpolation_validation,
            "test": heat_interpolation_test,
            "question": "Interpolation to the middle published volumetric heat source.",
        },
        "heat_source_extrapolation": {
            **value_split("solid_heat_source_MW_m3", {4.85}, 6.85, 8.85),
            "question": "Prediction at the highest published volumetric heat source.",
        },
    }


def validate(splits: dict[str, dict[str, object]], all_ids: set[str]) -> None:
    for name, split in splits.items():
        train = set(split["train"])
        validation = set(split["validation"])
        test = set(split["test"])
        if not train or not validation or not test:
            raise ValueError(f"{name} has an empty subset")
        if train & validation or train & test or validation & test:
            raise ValueError(f"{name} subsets overlap")
        if train | validation | test != all_ids:
            raise ValueError(f"{name} does not cover all P418 conditions")


def main() -> int:
    condition_records = records()
    splits = make_splits(condition_records)
    validate(splits, {str(item["condition_id"]) for item in condition_records})
    source = rows()["P418"]
    payload = {
        "source_parameter_id": "P418",
        "source_title": source["source_title"],
        "source_doi": source["source_url_or_doi"],
        "condition_count": len(condition_records),
        "conditions": condition_records,
        "splits": splits,
        "comparison_rule": (
            "Every model uses identical condition lists, train-only normalization, and the same "
            "field metrics. Architecture settings are numerical choices, not physical parameters."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: len(value) for key, value in split.items() if isinstance(value, list)} for name, split in splits.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
