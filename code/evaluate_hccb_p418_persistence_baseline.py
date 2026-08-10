#!/usr/bin/env python3
"""Evaluate the initial-temperature persistence baseline on complete P418 steps."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hccb_p418_comparison_contract import file_record
from train_hccb_p418_regional_dmdc import (
    field_metrics,
    load_sequence,
    records,
    regional_geometry,
    split_lists,
)


def evaluate(
    root: Path,
    source_records: dict[str, dict],
    sequence_ids: list[str],
    node_type: np.ndarray,
    volume: np.ndarray,
    centroid: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, np.ndarray]]]:
    """Repeat the registered initial temperature at every reporting time."""
    outputs: list[dict[str, np.ndarray]] = []
    start = time.perf_counter()
    for sequence_id in sequence_ids:
        (
            time_s,
            condition,
            target,
            fixed_hydrodynamics,
            internal_mass_flux,
            boundary_mass_flux,
        ) = load_sequence(root, source_records[sequence_id])
        prediction = np.broadcast_to(target[0], target.shape).copy()
        outputs.append(
            {
                "sequence_id": np.asarray(sequence_id),
                "time_s": time_s,
                "condition": condition,
                "fixed_hydrodynamics": fixed_hydrodynamics.astype(np.float32),
                "internal_mass_flux": internal_mass_flux.astype(np.float64),
                "boundary_mass_flux": boundary_mass_flux.astype(np.float64),
                "prediction": prediction.astype(np.float32),
                "target": target.astype(np.float32),
            }
        )
    elapsed = time.perf_counter() - start
    metrics = field_metrics(
        np.stack([row["prediction"] for row in outputs]),
        np.stack([row["target"] for row in outputs]),
        node_type,
        volume,
        centroid,
    )
    metrics.update(
        {
            "inference_seconds": elapsed,
            "inference_seconds_per_curve": elapsed / len(sequence_ids),
        }
    )
    return metrics, outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    begin = time.perf_counter()
    index_path = args.dataset_index.resolve()
    root = index_path.parent
    index = json.loads(index_path.read_text(encoding="utf-8"))
    source_records = records(index)
    if len(source_records) != 12 or not all(
        bool(row["complete"]) for row in source_records.values()
    ):
        raise ValueError(
            "formal persistence comparison requires all 12 complete thermal-step curves"
        )
    split = split_lists(
        set(source_records), args.splits.resolve(), args.split_name
    )
    node_type, volume, centroid = regional_geometry(root, index)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_role: dict[str, dict[str, float]] = {}
    prediction_files: dict[str, str] = {}
    prediction_file_records: dict[str, dict[str, object]] = {}
    for role in ("train", "validation", "test"):
        metrics, outputs = evaluate(
            root,
            source_records,
            split[role],
            node_type,
            volume,
            centroid,
        )
        metrics_by_role[role] = metrics
        path = args.output_dir / f"{role}_temperature_predictions.npz"
        np.savez_compressed(
            path,
            sequence_id=np.asarray([str(item["sequence_id"]) for item in outputs]),
            time_s=np.stack([item["time_s"] for item in outputs]),
            condition_physical=np.stack([item["condition"] for item in outputs]),
            fixed_hydrodynamics_physical=np.stack(
                [item["fixed_hydrodynamics"] for item in outputs]
            ),
            fluid_internal_mass_flux_kg_s=np.stack(
                [item["internal_mass_flux"] for item in outputs]
            ),
            fluid_boundary_mass_flux_kg_s=np.stack(
                [item["boundary_mass_flux"] for item in outputs]
            ),
            temperature_prediction_K=np.stack(
                [item["prediction"] for item in outputs]
            ),
            temperature_target_K=np.stack([item["target"] for item in outputs]),
            node_type=node_type,
            node_volume_m3=volume,
            node_centroid_m=centroid,
        )
        prediction_files[role] = path.name
        prediction_file_records[role] = file_record(path)

    summary = {
        "status": "completed_p418_initial_temperature_persistence_baseline",
        "dataset_index": str(index_path),
        "split_name": args.split_name,
        "split_case_counts": {role: len(ids) for role, ids in split.items()},
        "split_case_ids": split,
        "temperature_metric_definition": (
            "regional-volume-weighted RMSE, reported separately for fluid and solid"
        ),
        "metrics": metrics_by_role,
        "prediction_files": prediction_files,
        "prediction_file_records": prediction_file_records,
        "model_parameter_count": 0,
        "model_storage_scalar_count": 0,
        "training_seconds": 0.0,
        "total_evaluation_seconds": time.perf_counter() - begin,
        "compute_device": "cpu_numpy",
        "selection_split": "not_applicable",
        "selection_metric": "none; this baseline has no fitted parameters",
        "new_physical_parameters": [],
        "scientific_scope": (
            "No-dynamics baseline that repeats each trajectory's registered "
            "initial temperature field at all subsequent reporting times. It "
            "uses no training, validation or test target after t=0."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
