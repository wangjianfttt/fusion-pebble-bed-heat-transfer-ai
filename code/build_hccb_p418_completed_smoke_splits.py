#!/usr/bin/env python3
"""Build a deterministic train/validation/test split from completed P418 cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--source-splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-case-count", type=int)
    args = parser.parse_args()

    matrix_root = args.matrix_root.resolve()
    source_path = args.source_splits.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    condition_by_id = {
        str(item["condition_id"]): item for item in source["conditions"]
    }
    completed = sorted(
        path.parent.name
        for path in matrix_root.glob("*/formal_sample_complete.json")
    )
    if args.expected_case_count is not None and len(completed) != args.expected_case_count:
        raise ValueError(
            f"completed case count {len(completed)} != expected {args.expected_case_count}"
        )
    if len(completed) < 3:
        raise ValueError("at least three completed conditions are required")
    unknown = sorted(set(completed) - set(condition_by_id))
    if unknown:
        raise ValueError(f"completed conditions are absent from source split file: {unknown}")

    # The final two lexicographic conditions are held out only for a software-path check.
    # Formal scientific comparisons continue to use the predeclared 60-condition splits.
    train = completed[:-2]
    validation = [completed[-2]]
    test = [completed[-1]]
    payload = {
        "source_parameter_id": source["source_parameter_id"],
        "source_title": source["source_title"],
        "source_doi": source["source_doi"],
        "condition_count": len(completed),
        "conditions": [condition_by_id[item] for item in completed],
        "splits": {
            "completed_smoke": {
                "train": train,
                "validation": validation,
                "test": test,
                "question": (
                    "Software-path check for mixed completion times; not a formal "
                    "accuracy split or a new physical experiment."
                ),
            }
        },
        "scope": "completed_case_software_smoke_only",
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
