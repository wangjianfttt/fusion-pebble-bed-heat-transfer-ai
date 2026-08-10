#!/usr/bin/env python3
"""Check whether a saved steady-model result matches the current inputs and code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hccb_p418_comparison_contract import run_provenance, validate_split_and_statistics


def implementation_files(code_dir: Path, architecture: str) -> tuple[Path, ...]:
    if architecture == "response_surface":
        return (
            code_dir / "train_hccb_p418_regional_response_surface.py",
            code_dir / "hccb_p418_comparison_contract.py",
            code_dir / "train_hccb_p418_conservative_mixed_operator.py",
        )
    return (
        code_dir / "train_hccb_p418_conservative_mixed_operator.py",
        code_dir / "hccb_p418_comparison_contract.py",
        code_dir / "hccb_p418_conservative_mixed_operator.py",
        code_dir / "train_hccb_p418_regional_operator.py",
        code_dir / "hccb_p418_parametric_regional_operator.py",
        code_dir / "hccb_p418_coordinate_pinn.py",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--architecture",
        choices=("response_surface", "pinn_data_only", "pinn", "graph", "transolver"),
        required=True,
    )
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--state-targets", type=Path, required=True)
    parser.add_argument("--mass-targets", type=Path, required=True)
    parser.add_argument("--energy-targets", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--training-statistics", type=Path, required=True)
    parser.add_argument("--training-seed", type=int)
    args = parser.parse_args()

    if not args.summary.is_file():
        return 1
    saved = json.loads(args.summary.read_text(encoding="utf-8"))
    if saved.get("architecture") != args.architecture:
        return 1
    if saved.get("split_name") != args.split_name:
        return 1
    recorded_epochs = (
        saved.get("comparison_requested_epochs")
        if args.architecture == "response_surface"
        else saved.get("epochs")
    )
    if int(recorded_epochs if recorded_epochs is not None else -1) != args.epochs:
        return 1
    if args.architecture != "response_surface" and args.training_seed is not None:
        if int(saved.get("training_seed", -1)) != args.training_seed:
            return 1

    with np.load(args.state_targets.resolve(), allow_pickle=False) as loaded:
        condition_ids = loaded["condition_id"].astype(str)
    split_case_ids, _ = validate_split_and_statistics(
        split_file=args.split_file,
        training_statistics=args.training_statistics,
        split_name=args.split_name,
        condition_ids=condition_ids,
    )
    code_dir = Path(__file__).resolve().parent
    current = run_provenance(
        architecture=args.architecture,
        comparison_epochs=args.epochs,
        split_name=args.split_name,
        split_case_ids=split_case_ids,
        common_inputs={
            "state_targets": args.state_targets,
            "mass_targets": args.mass_targets,
            "energy_targets": args.energy_targets,
            "split_file": args.split_file,
            "training_statistics": args.training_statistics,
        },
        implementation_files=implementation_files(code_dir, args.architecture),
    )
    saved_provenance = saved.get("run_provenance", {})
    if saved_provenance.get("run_fingerprint") != current["run_fingerprint"]:
        return 1
    print(
        f"current result: {args.architecture} {args.split_name} "
        f"{current['run_fingerprint']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
