#!/usr/bin/env python3
"""Separate actual P418 calculation inputs from older comparison references."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from hccb_p418_source_contract import CASE_PHYSICS_PARAMETER_IDS

ACTUAL_INPUT_IDS = CASE_PHYSICS_PARAMETER_IDS

REFERENCE_ONLY = {
    "P391": (
        "Published pressure-drop and maximum-temperature values for a different, "
        "full-domain pore-scale case. They are comparison values and are not imposed "
        "on the present local-crop calculation."
    ),
    "P392": (
        "Older boundary-description entry. The present calculation uses the wall "
        "temperature and boundary layout stated by P425 and P427."
    ),
}


def parameter_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["parameter_id"]: row for row in csv.DictReader(handle)}


def update_case(
    case: Path, *, sample_directory: str, rows: dict[str, dict[str, str]]
) -> dict[str, object]:
    metadata_path = case / "cht_smoke_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    before = list(metadata.get("parameter_ids", []))
    missing = sorted(set(ACTUAL_INPUT_IDS) - set(rows))
    if missing:
        raise ValueError(f"parameter manifest misses {missing}")
    metadata["parameter_ids"] = list(ACTUAL_INPUT_IDS)
    metadata["literature_comparison_parameter_ids"] = list(REFERENCE_ONLY)
    metadata["literature_comparison_parameter_notes"] = REFERENCE_ONLY
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    sample_metadata_path = case / sample_directory / "metadata.json"
    sample_updated = False
    if sample_metadata_path.exists():
        sample = json.loads(sample_metadata_path.read_text(encoding="utf-8"))
        sample["literature_parameters"] = [rows[item] for item in ACTUAL_INPUT_IDS]
        sample["literature_comparison_parameters"] = [
            {"parameter": rows[item], "use": REFERENCE_ONLY[item]}
            for item in REFERENCE_ONLY
        ]
        sample_metadata_path.write_text(
            json.dumps(sample, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        sample_updated = True
    return {
        "condition_id": metadata["operating_condition_id"],
        "parameter_ids_before": before,
        "calculation_input_parameter_ids": list(ACTUAL_INPUT_IDS),
        "comparison_only_parameter_ids": list(REFERENCE_ONLY),
        "sample_metadata_updated": sample_updated,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--parameter-manifest", type=Path, required=True)
    parser.add_argument("--sample-directory", default="training_sample_300_schema3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = parameter_rows(args.parameter_manifest.resolve())
    cases = sorted(args.matrix_root.resolve().glob("u*_T*_q*"))
    if not cases:
        raise FileNotFoundError("no P418 case directories were found")
    records = [
        update_case(case, sample_directory=args.sample_directory, rows=rows)
        for case in cases
        if (case / "cht_smoke_metadata.json").exists()
    ]
    payload = {
        "status": "p418_parameter_roles_separated",
        "case_count": len(records),
        "explanation": (
            "P391 and P392 are kept as literature comparison records but are not "
            "listed as numerical inputs to the local-crop P418 calculations."
        ),
        "cases": records,
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
