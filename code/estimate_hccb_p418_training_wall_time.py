#!/usr/bin/env python3
"""Project formal P418 CPU training time from measured real-field batches."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ARCHITECTURES = ("pinn_data_only", "pinn", "graph", "transolver")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--split-names", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")

    references: dict[str, dict[str, object]] = {}
    for architecture in ARCHITECTURES:
        path = args.reference_root / architecture / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("epochs", 0)) != 1:
            raise ValueError(f"timing reference for {architecture} must use one epoch")
        if float(payload.get("optimization_seconds_per_update", 0.0)) <= 0.0:
            raise ValueError(f"missing measured update time for {architecture}")
        references[architecture] = payload

    splits = json.loads(args.split_file.read_text(encoding="utf-8"))["splits"]
    rows: list[dict[str, object]] = []
    totals: dict[str, float] = {}
    for split_name in args.split_names:
        split = splits[split_name]
        train_count = len(split["train"])
        validation_count = len(split["validation"])
        total_count = train_count + validation_count + len(split["test"])
        for architecture, reference in references.items():
            reference_ids = reference["split_case_ids"]
            reference_validation_count = len(reference_ids["validation"])
            reference_total_count = sum(len(reference_ids[role]) for role in ("train", "validation", "test"))
            reference_epochs = int(reference["epochs"])
            batch_size = int(reference["effective_batch_size"])
            updates = args.epochs * math.ceil(train_count / batch_size)
            optimization = updates * float(reference["optimization_seconds_per_update"])
            validation_per_case_epoch = float(reference["validation_seconds"]) / (
                reference_epochs * reference_validation_count
            )
            validation = validation_per_case_epoch * validation_count * args.epochs
            evaluation_per_case = float(reference["final_evaluation_seconds"]) / reference_total_count
            evaluation = evaluation_per_case * total_count
            projected = optimization + validation + evaluation
            rows.append(
                {
                    "split_name": split_name,
                    "architecture": architecture,
                    "epochs": args.epochs,
                    "train_conditions": train_count,
                    "validation_conditions": validation_count,
                    "test_conditions": len(split["test"]),
                    "effective_batch_size": batch_size,
                    "microbatch_size": reference.get("microbatch_size"),
                    "device": reference.get("device", "unknown"),
                    "measured_peak_gpu_memory_GB": reference.get("peak_gpu_memory_GB"),
                    "parameter_updates": updates,
                    "measured_optimization_s_per_update": reference["optimization_seconds_per_update"],
                    "projected_optimization_h": optimization / 3600.0,
                    "projected_validation_h": validation / 3600.0,
                    "projected_final_evaluation_h": evaluation / 3600.0,
                    "projected_total_h": projected / 3600.0,
                }
            )
            totals[split_name] = totals.get(split_name, 0.0) + projected

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "projected_training_wall_time.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "measured_training_time_projection",
        "epochs": args.epochs,
        "split_names": args.split_names,
        "architecture_count": len(ARCHITECTURES),
        "projected_serial_hours_by_split": {
            name: value / 3600.0 for name, value in totals.items()
        },
        "table": csv_path.name,
        "interpretation": (
            "Linear projection from one-epoch real-field measurements on the recorded device. It is a scheduling "
            "estimate under the current shared-machine load, not a guaranteed runtime or a model-speed result."
        ),
        "excluded_time": [
            "OpenFOAM field generation",
            "initial dataset assembly",
            "file checksum calculation",
            "response-surface fit",
        ],
        "new_physical_parameters": [],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
