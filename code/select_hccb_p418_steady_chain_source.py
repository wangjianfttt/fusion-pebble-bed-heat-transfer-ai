#!/usr/bin/env python3
"""Choose the steady PINN used to initialise the thermal-step chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validation_score(summary_path: Path, payload: dict[str, object]) -> float:
    if "best_validation_total_loss" in payload:
        return float(payload["best_validation_total_loss"])
    best_epoch = int(payload["best_epoch"])
    history_path = summary_path.parent / "training_history.jsonl"
    if not history_path.is_file():
        raise ValueError(f"steady summary has no validation score or history: {summary_path}")
    rows = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if int(row["epoch"]) == best_epoch]
    if len(selected) != 1:
        raise ValueError(f"cannot locate selected validation epoch {best_epoch}: {history_path}")
    return float(selected[0]["validation"]["total_loss"])


def choose(
    *,
    base_summary: Path,
    followup_plan: Path,
    project_root: Path,
    architecture: str,
    split_name: str,
) -> dict[str, object]:
    base_summary = base_summary.resolve()
    base = json.loads(base_summary.read_text(encoding="utf-8"))
    if base.get("architecture") != architecture or base.get("split_name") != split_name:
        raise ValueError("base steady model does not match the requested architecture and split")
    candidates = [("initial", base_summary, base)]
    plan = json.loads(followup_plan.resolve().read_text(encoding="utf-8"))
    matching = [
        row
        for row in plan.get("runs", [])
        if row["architecture"] == architecture and row["split"] == split_name
    ]
    if len(matching) > 1:
        raise ValueError("follow-up plan repeats the requested architecture and split")
    if matching:
        followup_summary = (
            project_root.resolve()
            / str(matching[0]["followup_result_directory"])
            / "summary.json"
        )
        followup = json.loads(followup_summary.read_text(encoding="utf-8"))
        if followup.get("architecture") != architecture or followup.get("split_name") != split_name:
            raise ValueError("follow-up steady model does not match the requested architecture and split")
        if base.get("split_case_ids") != followup.get("split_case_ids"):
            raise ValueError("initial and follow-up steady models use different condition splits")
        base_fingerprint = base.get("run_provenance", {}).get(
            "common_comparison_fingerprint"
        )
        followup_fingerprint = followup.get("run_provenance", {}).get(
            "common_comparison_fingerprint"
        )
        if not base_fingerprint or base_fingerprint != followup_fingerprint:
            raise ValueError("initial and follow-up steady models use different physical data")
        candidates.append(("followup", followup_summary, followup))

    scored = [
        {
            "role": role,
            "summary": str(path),
            "epochs": int(payload["epochs"]),
            "selected_epoch": int(payload["best_epoch"]),
            "best_validation_total_loss": validation_score(path, payload),
        }
        for role, path, payload in candidates
    ]
    selected = min(scored, key=lambda row: row["best_validation_total_loss"])
    return {
        "status": "steady_PINN_chain_source_selected",
        "architecture": architecture,
        "split_name": split_name,
        "selection_data": "validation conditions only",
        "independent_test_used_for_selection": False,
        "candidates": scored,
        "selected_summary": selected["summary"],
        "selected_epochs": selected["epochs"],
        "selected_epoch": selected["selected_epoch"],
        "selected_validation_total_loss": selected["best_validation_total_loss"],
        "new_physical_parameters": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-summary", type=Path, required=True)
    parser.add_argument("--followup-plan", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--architecture", default="pinn")
    parser.add_argument("--split-name", default="interleaved_all_ranges")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = choose(
        base_summary=args.base_summary,
        followup_plan=args.followup_plan,
        project_root=args.project_root,
        architecture=args.architecture,
        split_name=args.split_name,
    )
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
