#!/usr/bin/env python3
"""Compare two training lengths on the same P418 conditions and architecture."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: Path) -> dict[str, object]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if payload.get("architecture") is None or payload.get("split_name") is None:
        raise ValueError(f"missing architecture or split name in {path}")
    return payload


def result_row(label: str, payload: dict[str, object], source: Path) -> dict[str, object]:
    validation = payload["evaluations"]["validation"]
    test = payload["evaluations"]["test"]
    case = test["cases"][0]
    errors = case["engineering_absolute_errors"]
    return {
        "label": label,
        "epochs": int(payload["epochs"]),
        "best_epoch": int(payload["best_epoch"]),
        "validation_total_loss": float(validation["metrics"]["total_loss"]),
        "test_state_normalized_rmse": float(test["metrics"]["state_normalized_rmse"]),
        "test_outlet_temperature_error_K": float(errors["outlet_temperature_K"]),
        "test_solid_maximum_temperature_error_K": float(
            errors["solid_maximum_temperature_K"]
        ),
        "test_cooling_wall_heat_error_W": float(
            errors["cooling_wall_heat_into_fluid_W"]
        ),
        "test_global_energy_imbalance_over_generated_power": float(
            case["global_energy_imbalance_over_generated_power"]
        ),
        "training_wall_time_s": float(payload["training_seconds"]),
        "peak_gpu_memory_GB": float(payload["peak_gpu_memory_GB"]),
        "summary_file": str(source.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--short-summary", type=Path, required=True)
    parser.add_argument("--long-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    short = load(args.short_summary)
    long = load(args.long_summary)
    for key in ("architecture", "split_name"):
        if short[key] != long[key]:
            raise ValueError(f"training-length comparison changed {key}")
    short_fingerprint = short["run_provenance"]["common_comparison_fingerprint"]
    long_fingerprint = long["run_provenance"]["common_comparison_fingerprint"]
    if short_fingerprint != long_fingerprint:
        raise ValueError("training-length comparison changed fields, split, or scaling")
    if int(long["epochs"]) <= int(short["epochs"]):
        raise ValueError("long schedule must contain more epochs than short schedule")

    rows = [
        result_row("short", short, args.short_summary),
        result_row("long", long, args.long_summary),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = args.output_dir / "training_extension_comparison.csv"
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    delta = {
        key: rows[1][key] - rows[0][key]
        for key in (
            "validation_total_loss",
            "test_state_normalized_rmse",
            "test_outlet_temperature_error_K",
            "test_solid_maximum_temperature_error_K",
            "test_cooling_wall_heat_error_W",
            "test_global_energy_imbalance_over_generated_power",
            "training_wall_time_s",
        )
    }
    summary = {
        "status": "completed_small_condition_training_extension_check",
        "architecture": short["architecture"],
        "split_name": short["split_name"],
        "train_condition_count": len(short["split_case_ids"]["train"]),
        "short_epochs": rows[0]["epochs"],
        "long_epochs": rows[1]["epochs"],
        "long_minus_short": delta,
        "interpretation_cn": (
            "延长训练后必须同时检查三维状态、出口温度、颗粒最高温度和热量收支；"
            "一个总损失下降不足以证明工程量预测同时改善。"
        ),
        "table": table.name,
        "new_physical_parameters": [],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
