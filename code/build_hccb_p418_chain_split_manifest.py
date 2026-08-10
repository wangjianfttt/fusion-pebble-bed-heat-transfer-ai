#!/usr/bin/env python3
"""Record how steady-condition roles and transient-curve roles meet in the fused model."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from hccb_p418_chain_roles import endpoint_novelty_class


def build_rows(
    steady_splits: dict[str, object],
    transient_plan: dict[str, object],
    transient_splits: dict[str, object],
    steady_split_name: str,
) -> list[dict[str, str]]:
    split = steady_splits["splits"][steady_split_name]
    steady_role: dict[str, str] = {}
    for role in ("train", "validation", "test"):
        for condition_id in split[role]:
            if condition_id in steady_role:
                raise ValueError(f"steady condition {condition_id} appears more than once")
            steady_role[str(condition_id)] = role
    sequences = {
        str(row["sequence_id"]): row for row in transient_plan["sequences"]
    }
    rows: list[dict[str, str]] = []
    for transient_split_name, roles in transient_splits["splits"].items():
        seen: set[str] = set()
        for transient_role in ("train", "validation", "test"):
            for sequence_id in roles[transient_role]:
                if sequence_id in seen:
                    raise ValueError(
                        f"{sequence_id} is repeated in transient split {transient_split_name}"
                    )
                seen.add(sequence_id)
                sequence = sequences[str(sequence_id)]
                source = str(sequence["source_condition_id"])
                target = str(sequence["target_condition_id"])
                if source not in steady_role or target not in steady_role:
                    raise ValueError(f"{sequence_id} uses a condition outside the steady split")
                source_role = steady_role[source]
                target_role = steady_role[target]
                rows.append(
                    {
                        "steady_split_name": steady_split_name,
                        "transient_split_name": str(transient_split_name),
                        "transient_role": transient_role,
                        "sequence_id": str(sequence_id),
                        "physical_step_family": str(sequence["family"]),
                        "source_condition_id": source,
                        "source_steady_role": source_role,
                        "target_condition_id": target,
                        "target_steady_role": target_role,
                        "endpoint_novelty_class": endpoint_novelty_class(
                            source_role, target_role
                        ),
                    }
                )
        if seen != set(sequences):
            missing = sorted(set(sequences).difference(seen))
            raise ValueError(
                f"transient split {transient_split_name} does not cover: {missing}"
            )
    return rows


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    test_rows = [row for row in rows if row["transient_role"] == "test"]
    return {
        "status": "completed_p418_fused_chain_split_manifest",
        "row_count": len(rows),
        "transient_test_row_count": len(test_rows),
        "test_curve_counts_by_endpoint_novelty": dict(
            sorted(Counter(row["endpoint_novelty_class"] for row in test_rows).items())
        ),
        "strict_end_to_end_test_curves": [
            {
                "transient_split_name": row["transient_split_name"],
                "sequence_id": row["sequence_id"],
                "physical_step_family": row["physical_step_family"],
            }
            for row in test_rows
            if row["endpoint_novelty_class"] == "both_steady_endpoints_unseen"
        ],
        "interpretation": (
            "A transient test curve is called strict end-to-end only when the curve is held "
            "out from transient fitting and both of its steady endpoints are absent from "
            "steady-PINN fitting. Other held-out curves remain valid transient tests but do "
            "not test unseen steady endpoints simultaneously."
        ),
        "new_physical_parameters": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steady-splits", type=Path, required=True)
    parser.add_argument("--transient-plan", type=Path, required=True)
    parser.add_argument("--transient-splits", type=Path, required=True)
    parser.add_argument("--steady-split-name", default="interleaved_all_ranges")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    rows = build_rows(
        json.loads(args.steady_splits.read_text(encoding="utf-8")),
        json.loads(args.transient_plan.read_text(encoding="utf-8")),
        json.loads(args.transient_splits.read_text(encoding="utf-8")),
        args.steady_split_name,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = summarize(rows)
    payload.update(
        {
            "steady_split_name": args.steady_split_name,
            "manifest_csv": str(args.output_csv.resolve()),
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
