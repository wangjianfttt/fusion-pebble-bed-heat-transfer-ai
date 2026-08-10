#!/usr/bin/env python3
"""Compare the corrected-flow preflight with its repeated formal case."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_values(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(numeric_values(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix] = float(value)
    return result


def load_case(case: Path) -> tuple[dict[str, Any], dict[str, Any], Path, str]:
    marker = json.loads((case / "formal_sample_complete.json").read_text(encoding="utf-8"))
    time_name = str(marker["time"])
    summary_path = case / f"cht_result_summary_{time_name}.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sample = Path(marker["training_sample"])
    if not sample.is_file():
        raise FileNotFoundError(f"training sample is missing: {sample}")
    return marker, summary, sample, sha256(sample)


def compare(preflight: Path, formal: Path) -> dict[str, Any]:
    pre_marker, pre_summary, pre_sample, pre_actual_hash = load_case(preflight)
    formal_marker, formal_summary, formal_sample, formal_actual_hash = load_case(formal)

    pre_values = numeric_values(pre_summary)
    formal_values = numeric_values(formal_summary)
    shared = sorted(set(pre_values) & set(formal_values))
    rows = []
    for quantity in shared:
        pre_value = pre_values[quantity]
        formal_value = formal_values[quantity]
        absolute = abs(formal_value - pre_value)
        scale = max(abs(pre_value), abs(formal_value))
        rows.append(
            {
                "quantity": quantity,
                "preflight": pre_value,
                "formal": formal_value,
                "absolute_difference": absolute,
                "relative_difference": absolute / scale if scale else 0.0,
            }
        )

    pre_recorded_hash = str(pre_marker["training_sample_sha256"])
    formal_recorded_hash = str(formal_marker["training_sample_sha256"])
    missing_in_formal = sorted(set(pre_values) - set(formal_values))
    missing_in_preflight = sorted(set(formal_values) - set(pre_values))
    exact = (
        bool(pre_summary.get("solver_finished"))
        and bool(formal_summary.get("solver_finished"))
        and str(pre_marker["time"]) == str(formal_marker["time"])
        and not missing_in_formal
        and not missing_in_preflight
        and all(row["absolute_difference"] == 0.0 for row in rows)
        and pre_recorded_hash == pre_actual_hash
        and formal_recorded_hash == formal_actual_hash
        and pre_actual_hash == formal_actual_hash
    )
    return {
        "status": (
            "corrected_preflight_exactly_reproduced_by_first_formal_case"
            if exact
            else "preflight_and_first_formal_case_differ"
        ),
        "preflight_case": str(preflight.resolve()),
        "formal_case": str(formal.resolve()),
        "completion_time_s": float(formal_marker["time"]),
        "compared_numeric_quantity_count": len(rows),
        "maximum_absolute_difference": max(
            (row["absolute_difference"] for row in rows), default=0.0
        ),
        "maximum_relative_difference": max(
            (row["relative_difference"] for row in rows), default=0.0
        ),
        "missing_in_formal": missing_in_formal,
        "missing_in_preflight": missing_in_preflight,
        "training_sample_bytes": formal_sample.stat().st_size,
        "training_sample_sha256": formal_actual_hash,
        "training_sample_hashes_identical": pre_actual_hash == formal_actual_hash,
        "training_sample_recorded_hashes_match_files": (
            pre_recorded_hash == pre_actual_hash and formal_recorded_hash == formal_actual_hash
        ),
        "numeric_comparison": rows,
        "new_physical_parameters": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-case", required=True, type=Path)
    parser.add_argument("--formal-case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.preflight_case, args.formal_case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "corrected_preflight_exactly_reproduced_by_first_formal_case":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
