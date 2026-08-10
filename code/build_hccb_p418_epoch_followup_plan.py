#!/usr/bin/env python3
"""Build source-epoch follow-up runs from the first P418 training comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ALLOWED_ARCHITECTURES = {"pinn_data_only", "pinn", "graph", "transolver"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--convergence-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.convergence_summary.read_text(encoding="utf-8"))
    requested = int(source["requested_epochs"])
    result_prefix = str(source["result_prefix"])
    runs: list[dict[str, object]] = []
    for record in source.get("models", []):
        architecture = str(record["architecture"])
        if architecture not in ALLOWED_ARCHITECTURES:
            raise ValueError(f"unsupported architecture: {architecture}")
        if int(record["completed_epochs"]) != requested:
            raise ValueError(f"inconsistent completed epochs for {architecture}")
        if not bool(record["best_epoch_is_final_epoch"]):
            continue
        source_epochs = int(record["published_source_epochs"])
        if source_epochs <= requested:
            raise ValueError(
                f"source schedule does not extend the first comparison: {architecture}"
            )
        split = str(record["split"])
        runs.append(
            {
                "architecture": architecture,
                "split": split,
                "initial_epochs": requested,
                "followup_epochs": source_epochs,
                "initial_result_directory": (
                    f"results/{result_prefix}_{architecture}_{split}_{requested}epoch"
                ),
                "followup_result_directory": (
                    f"results/{result_prefix}_{architecture}_{split}_{source_epochs}epoch"
                ),
                "reason": "best validation loss occurs at the final epoch of the first comparison",
            }
        )
    payload = {
        "status": "source_epoch_followup_plan_ready",
        "convergence_summary": str(args.convergence_summary),
        "result_prefix": result_prefix,
        "initial_epochs": requested,
        "followup_run_count": len(runs),
        "runs": runs,
        "interpretation": (
            "Each listed model is rerun from scratch with the epoch count recorded in its "
            "archived source. The first and source-length runs are both retained, and the "
            "longer run is not assumed to be more accurate."
        ),
        "new_physical_parameters": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
