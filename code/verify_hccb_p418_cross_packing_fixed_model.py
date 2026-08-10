#!/usr/bin/env python3
"""Confirm that seed303 uses the exact model previously evaluated on seed202."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def verify(
    *,
    selection_path: Path,
    model_sources_path: Path,
    project_root: Path,
    architecture: str,
    seed202_result_path: Path,
) -> dict[str, object]:
    selection_path = selection_path.resolve()
    model_sources_path = model_sources_path.resolve()
    seed202_result_path = seed202_result_path.resolve()
    root = project_root.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "seed202_architecture_fixed_before_seed303":
        raise ValueError("seed202 architecture selection is not complete")
    if selection.get("selected_architecture") != architecture:
        raise ValueError("requested architecture differs from the seed202 selection")
    if selection.get("seed303_fields_read") is not False:
        raise ValueError("seed202 selection does not prove that seed303 was unseen")
    sources_hash = sha256(model_sources_path)
    if selection.get("seed101_model_sources_sha256") != sources_hash:
        raise ValueError("seed101 model-source map changed after seed202 selection")

    sources = json.loads(model_sources_path.read_text(encoding="utf-8"))
    if sources.get("status") != "cross_packing_seed101_model_sources_selected":
        raise ValueError("seed101 model-source map is not ready")
    record = sources.get("models", {}).get(architecture)
    if not record:
        raise ValueError("fixed architecture is absent from the model-source map")
    checkpoint = resolve(root, str(record["selected_checkpoint"]))
    summary = resolve(root, str(record["selected_summary"]))
    for path, recorded, label in (
        (checkpoint, record["selected_checkpoint_sha256"], "checkpoint"),
        (summary, record["selected_summary_sha256"], "training summary"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"validation-selected {label} is missing: {path}")
        if sha256(path) != recorded:
            raise ValueError(f"validation-selected {label} changed before seed303")

    seed202 = json.loads(seed202_result_path.read_text(encoding="utf-8"))
    if seed202.get("status") != "cross_packing_conservative_evaluation_complete":
        raise ValueError("seed202 result is not complete")
    if seed202.get("architecture") != architecture or int(
        seed202.get("packing_seed", -1)
    ) != 202:
        raise ValueError("seed202 result does not match the fixed architecture")
    if seed202.get("checkpoint_sha256") != record["selected_checkpoint_sha256"]:
        raise ValueError("seed303 checkpoint differs from the one used on seed202")
    if seed202.get("training_summary_sha256") != record["selected_summary_sha256"]:
        raise ValueError("seed303 training summary differs from the one used on seed202")

    registered = selection.get("seed101_checkpoint_selection", {}).get(architecture)
    if not registered:
        raise ValueError("seed202 selection lacks the fixed seed101 checkpoint record")
    if int(registered["selected_epochs"]) != int(record["selected_epochs"]):
        raise ValueError("seed202 selection and model-source map use different training lengths")
    if float(registered["selected_validation_total_loss"]) != float(
        record["selected_validation_total_loss"]
    ):
        raise ValueError("seed202 selection and model-source map use different validation results")

    return {
        "status": "seed303_uses_exact_seed202_model",
        "architecture": architecture,
        "checkpoint_sha256": record["selected_checkpoint_sha256"],
        "training_summary_sha256": record["selected_summary_sha256"],
        "seed101_model_sources_sha256": sources_hash,
        "seed202_result_sha256": sha256(seed202_result_path),
        "seed303_fields_read": False,
        "new_physical_parameter_values_added": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--model-sources", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--seed202-result", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = verify(
        selection_path=args.selection,
        model_sources_path=args.model_sources,
        project_root=args.project_root,
        architecture=args.architecture,
        seed202_result_path=args.seed202_result,
    )
    if args.output is not None:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
