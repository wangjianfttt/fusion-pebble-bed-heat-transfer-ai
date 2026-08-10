#!/usr/bin/env python3
"""Select one seed101 checkpoint per architecture using validation data only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ARCHITECTURES = ("pinn_data_only", "pinn", "graph", "transolver")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validation_score(summary: dict[str, object]) -> float:
    value = float(summary["best_validation_total_loss"])
    if not value >= 0.0:
        raise ValueError("validation loss must be finite and non-negative")
    return value


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def load_candidate(
    path: Path,
    *,
    architecture: str,
    split_name: str,
    reference_split: dict[str, object] | None,
    reference_fingerprint: str | None,
) -> tuple[dict[str, object], dict[str, object], str]:
    summary_path = path.resolve()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("status") != "conservative_mixed_operator_training_complete":
        raise ValueError(f"incomplete seed101 training result: {summary_path}")
    if payload.get("architecture") != architecture or payload.get("split_name") != split_name:
        raise ValueError(f"wrong architecture or split in {summary_path}")
    split = payload.get("split_case_ids")
    fingerprint = payload.get("run_provenance", {}).get(
        "common_comparison_fingerprint"
    )
    if not isinstance(split, dict) or not fingerprint:
        raise ValueError(f"missing split or physical-data fingerprint: {summary_path}")
    if reference_split is not None and split != reference_split:
        raise ValueError("initial and extended training use different condition splits")
    if reference_fingerprint is not None and fingerprint != reference_fingerprint:
        raise ValueError("initial and extended training use different physical data")
    checkpoint = summary_path.parent / "best.pt"
    if not checkpoint.is_file():
        raise ValueError(f"missing validation-selected checkpoint: {checkpoint}")
    candidate = {
        "summary": str(summary_path),
        "checkpoint": str(checkpoint.resolve()),
        "epochs": int(payload["epochs"]),
        "selected_epoch": int(payload["best_epoch"]),
        "best_validation_total_loss": validation_score(payload),
    }
    return candidate, split, str(fingerprint)


def build(
    *,
    project_root: Path,
    initial_epochs: int,
    split_name: str,
    followup_plan: Path,
    result_namespace: str = "hccb_p418_60",
) -> dict[str, object]:
    root = project_root.resolve()
    plan = json.loads(followup_plan.resolve().read_text(encoding="utf-8"))
    if plan.get("status") != "source_epoch_followup_plan_ready":
        raise ValueError("source-epoch follow-up plan is not ready")
    followups: dict[str, Path] = {}
    for row in plan.get("runs", []):
        if row["split"] != split_name:
            continue
        architecture = str(row["architecture"])
        if architecture in followups:
            raise ValueError(f"duplicate follow-up for {architecture}")
        followups[architecture] = (
            root / str(row["followup_result_directory"]) / "summary.json"
        )

    selected_models: dict[str, object] = {}
    for architecture in ARCHITECTURES:
        initial_path = (
            root
            / "results"
            / f"{result_namespace}_{architecture}_{split_name}_{initial_epochs}epoch"
            / "summary.json"
        )
        initial, split, fingerprint = load_candidate(
            initial_path,
            architecture=architecture,
            split_name=split_name,
            reference_split=None,
            reference_fingerprint=None,
        )
        candidates = [{"role": "initial", **initial}]
        if architecture in followups:
            extended, _, _ = load_candidate(
                followups[architecture],
                architecture=architecture,
                split_name=split_name,
                reference_split=split,
                reference_fingerprint=fingerprint,
            )
            candidates.append({"role": "source_length", **extended})
        selected = min(
            candidates,
            key=lambda row: (float(row["best_validation_total_loss"]), int(row["epochs"])),
        )
        summary_path = Path(str(selected["summary"]))
        checkpoint_path = Path(str(selected["checkpoint"]))
        selected_models[architecture] = {
            "selection_data": "seed101 validation conditions only",
            "independent_test_used_for_selection": False,
            "candidates": [
                {
                    **row,
                    "summary": relative(Path(str(row["summary"])), root),
                    "checkpoint": relative(Path(str(row["checkpoint"])), root),
                }
                for row in candidates
            ],
            "selected_summary": relative(summary_path, root),
            "selected_summary_sha256": sha256(summary_path),
            "selected_checkpoint": relative(checkpoint_path, root),
            "selected_checkpoint_sha256": sha256(checkpoint_path),
            "selected_epochs": int(selected["epochs"]),
            "selected_epoch": int(selected["selected_epoch"]),
            "selected_validation_total_loss": float(
                selected["best_validation_total_loss"]
            ),
            "split_case_ids": split,
            "common_comparison_fingerprint": fingerprint,
        }

    fingerprints = {
        record["common_comparison_fingerprint"] for record in selected_models.values()
    }
    split_sets = {
        json.dumps(record["split_case_ids"], sort_keys=True)
        for record in selected_models.values()
    }
    if len(fingerprints) != 1 or len(split_sets) != 1:
        raise ValueError("architectures do not use the same seed101 data and split")
    return {
        "status": "cross_packing_seed101_model_sources_selected",
        "result_namespace": result_namespace,
        "split_name": split_name,
        "selection_data": "seed101 validation conditions only",
        "independent_test_used_for_selection": False,
        "seed202_fields_read": False,
        "seed303_fields_read": False,
        "models": selected_models,
        "new_physical_parameter_values_added": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--result-namespace", default="hccb_p418_60")
    parser.add_argument("--initial-epochs", type=int, default=100)
    parser.add_argument("--split-name", default="interleaved_all_ranges")
    parser.add_argument("--followup-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(
        project_root=args.project_root,
        result_namespace=args.result_namespace,
        initial_epochs=args.initial_epochs,
        split_name=args.split_name,
        followup_plan=args.followup_plan,
    )
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
