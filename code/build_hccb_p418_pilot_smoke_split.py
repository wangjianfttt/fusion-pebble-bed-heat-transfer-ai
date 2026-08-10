#!/usr/bin/env python3
"""Create a five-case software-test split from completed P418 conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TRAIN = (
    "u0p05_T300_q8p85",
    "u0p20_T700_q6p85",
    "u0p25_T900_q4p85",
)
VALIDATION = ("u0p05_T900_q8p85",)
TEST = ("u0p25_T300_q4p85",)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = json.loads(args.dataset_index.resolve().read_text(encoding="utf-8"))
    records = {str(item["condition_id"]): item for item in dataset["conditions"]}
    selected = TRAIN + VALIDATION + TEST
    missing = sorted(set(selected) - set(records))
    if missing:
        raise ValueError(f"pilot dataset misses {missing}")
    payload = {
        "status": "software_smoke_split_only",
        "source_doi": "https://doi.org/10.1016/j.ijheatmasstransfer.2023.124325",
        "explanation": (
            "All five cases are exact published P418 combinations. This split only "
            "tests the training program and must not be used to report model accuracy."
        ),
        "conditions": [records[item] for item in selected],
        "splits": {
            "pilot_smoke": {
                "train": list(TRAIN),
                "validation": list(VALIDATION),
                "test": list(TEST),
                "question": "Does the complete training and evaluation program execute on actual 3D fields?",
            }
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
