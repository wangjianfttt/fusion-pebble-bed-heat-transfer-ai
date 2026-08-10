#!/usr/bin/env python3
"""Select one P418 loss-balancing candidate using validation curves only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_root = args.candidate_root.resolve()
    sources_path = args.sources.resolve()
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    declared = [row["candidate_id"] for row in sources["formal_candidates"]]
    records: list[dict[str, object]] = []
    common: dict[str, object] | None = None
    for candidate_id in declared:
        summary_path = candidate_root / candidate_id / "selection_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("evaluation_stage") != "selection":
            raise ValueError(f"{candidate_id} was not run in validation-only selection mode")
        if summary.get("test_evaluated") or "test" in summary.get("metrics", {}):
            raise ValueError(f"{candidate_id} read the independent test curves too early")
        if summary["loss_balancing"]["candidate_id"] != candidate_id:
            raise ValueError(f"{candidate_id} summary identifies another candidate")
        comparable = {
            "dataset_index": summary["dataset_index"],
            "input_file_sha256": summary["input_file_sha256"],
            "split_name": summary["split_name"],
            "split_sequence_ids": summary["split_sequence_ids"],
            "seed": summary["seed"],
            "architecture": summary["architecture"],
            "physics_terms": summary["physics_terms"],
            "training_normalization_sequence_ids": summary[
                "training_normalization_sequence_ids"
            ],
        }
        if common is None:
            common = comparable
        elif comparable != common:
            raise ValueError("loss-balancing candidates do not share one data/model setting")
        records.append(
            {
                "candidate_id": candidate_id,
                "validation_selection_score": float(
                    summary["best_validation_selection_score"]
                ),
                "best_epoch": int(summary["best_epoch"]),
                "summary_path": str(summary_path),
                "summary_sha256": sha256(summary_path),
            }
        )

    selected = min(
        records,
        key=lambda row: (
            row["validation_selection_score"],
            row["candidate_id"],
        ),
    )
    output = {
        "status": "p418_loss_balancing_selected_on_validation_only",
        "selection_metric": (
            "equal mean of dimensionless state, face-flux and physics validation groups"
        ),
        "source_file": str(sources_path),
        "source_file_sha256": sha256(sources_path),
        "candidate_records": records,
        "selected_candidate_id": selected["candidate_id"],
        "selected_validation_score": selected["validation_selection_score"],
        "selected_summary_path": selected["summary_path"],
        "selected_summary_sha256": selected["summary_sha256"],
        "common_data_and_model_setting": common,
        "independent_test_read": False,
        "next_step": (
            "Resume only the selected candidate with --evaluation-stage final and "
            "--selected-method-record pointing to this file."
        ),
        "new_physical_parameters": [],
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
