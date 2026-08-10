#!/usr/bin/env python3
"""Rebuild final comparison, tables, text and figures after loss-weight selection."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from hccb_p418_selected_fixed_flow_chain import (
    STRICT_SPLIT,
    selected_model_directories,
)


FINAL_JOB_IDS = (
    "summarize_model_comparison",
    "build_transient_performance_table",
    "build_transient_cost_table",
    "build_transient_result_text",
    "plot_transient_model_comparison",
    "plot_openfoam_model_field_comparison",
)


def load_jobs(manifest: Path) -> list[dict]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("formal manifest has no jobs")
    selected = [job for job in jobs if job.get("job_id") in FINAL_JOB_IDS]
    if tuple(str(job["job_id"]) for job in selected) != FINAL_JOB_IDS:
        raise ValueError("formal manifest final-output jobs differ from the registered order")
    return selected


def require_final_outputs(root: Path, result_dir: Path) -> None:
    summary_path = result_dir / "model_comparison/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed_p418_physical_step_model_comparison":
        raise ValueError("final transient comparison is incomplete")
    if summary.get("strict_split_loss_balancing_stage") != "validation_selected":
        raise ValueError("final transient comparison did not use validation-selected weights")
    expected = {
        root / "figures/hccb_p418_transient_model_comparison.json": (
            "complete_formal_p418_transient_model_comparison_figure"
        ),
        root / "figures/hccb_p418_openfoam_model_field_comparison.json": (
            "complete_same_scale_openfoam_model_field_comparison"
        ),
    }
    for path, status in expected.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != status:
            raise ValueError(f"unexpected final-output status in {path}")
        if payload.get("strict_split_loss_balancing_stage") != "validation_selected":
            raise ValueError(f"final output did not use selected loss weights: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    result_dir = args.result_dir.resolve()
    selected_model_directories(result_dir, STRICT_SPLIT)
    jobs = load_jobs(args.manifest.resolve())
    if not args.execute:
        print(json.dumps({"job_ids": list(FINAL_JOB_IDS)}, indent=2))
        return 0
    for job in jobs:
        subprocess.run(
            str(job["command"]),
            shell=True,
            executable="/bin/bash",
            cwd=root,
            check=True,
        )
    require_final_outputs(root, result_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
