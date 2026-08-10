#!/usr/bin/env python3
"""Train a volume-weighted DMDc baseline on P418 regional thermal steps."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from hccb_p418_transient_hotspot_metrics import solid_transient_hotspot_metrics


DMD_CITATION = {
    "paper": "Dynamic Mode Decomposition with Control",
    "authors": "Joshua L. Proctor, Steven L. Brunton and J. Nathan Kutz",
    "venue": "SIAM Journal on Applied Dynamical Systems 15 (2016) 142-161",
    "doi": "10.1137/15M1013857",
    "paper_url": "https://epubs.siam.org/doi/10.1137/15M1013857",
}
DEFAULT_RANKS = (4, 8, 12, 16, 24, 32)


def records(index: dict) -> dict[str, dict]:
    output = {str(row["sequence_id"]): row for row in index["sequences"]}
    if len(output) != int(index["sequence_count"]):
        raise ValueError("sequence records are duplicated or incomplete")
    return output


def split_lists(index_ids: set[str], path: Path, name: str) -> dict[str, list[str]]:
    source = json.loads(path.read_text(encoding="utf-8"))["splits"][name]
    split = {role: [str(value) for value in source[role]] for role in ("train", "validation", "test")}
    used = set().union(*(set(value) for value in split.values()))
    if used != index_ids:
        raise ValueError("DMDc split and sequence index differ")
    if any(
        set(split[a]) & set(split[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("DMDc curves overlap across split roles")
    return split


def load_sequence(
    root: Path, record: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(root / str(record["sequence_file"]), allow_pickle=False) as data:
        time_s = data["time_s"].astype(np.float64)
        condition = data["condition_physical"].astype(np.float64)
        state = data["state_physical"].astype(np.float64)
        temperature = state[..., 4]
        fixed_hydrodynamics = state[0, :, :4]
        internal_mass_flux = data["fluid_internal_mass_flux_kg_s"].astype(np.float64)
        boundary_mass_flux = data["fluid_boundary_mass_flux_kg_s"].astype(np.float64)
    if temperature.ndim != 2 or len(time_s) != len(temperature):
        raise ValueError(f"invalid temperature history for {record['sequence_id']}")
    delta = np.diff(time_s)
    if len(delta) == 0 or np.any(~np.isfinite(delta)) or np.any(delta <= 0.0):
        raise ValueError("DMDc requires finite, strictly increasing physical times")
    if not np.array_equal(
        state[..., :4],
        np.broadcast_to(fixed_hydrodynamics[None], state[..., :4].shape),
    ):
        raise ValueError("DMDc thermal-step data do not contain fixed hydrodynamics")
    return (
        time_s,
        condition,
        temperature,
        fixed_hydrodynamics,
        internal_mass_flux,
        boundary_mass_flux,
    )


def regional_geometry(root: Path, index: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(root / str(index["regional_geometry_file"]), allow_pickle=False) as data:
        node_type = data["node_type"].astype(np.int8)
        volume = data["node_volume_m3"].astype(np.float64)
        centroid = data["node_centroid_m"].astype(np.float64)
    if np.any(volume <= 0) or not set(np.unique(node_type)).issubset({0, 1}):
        raise ValueError("invalid regional material types or volumes")
    if centroid.shape != (len(node_type), 3) or not np.all(np.isfinite(centroid)):
        raise ValueError("invalid regional node centroids")
    return node_type, volume, centroid


def field_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    node_type: np.ndarray,
    volume: np.ndarray,
    centroid: np.ndarray,
) -> dict[str, float]:
    error = prediction - target
    result = {}
    for material, name in ((0, "fluid"), (1, "solid")):
        selected = node_type == material
        weight = volume[selected]
        square = np.square(error[..., selected])
        result[f"{name}_temperature_RMSE_K"] = float(
            np.sqrt(np.sum(square * weight) / (np.prod(square.shape[:-1]) * weight.sum()))
        )
    result["maximum_absolute_temperature_error_K"] = float(np.max(np.abs(error)))
    result.update(
        solid_transient_hotspot_metrics(prediction, target, node_type, centroid)
    )
    return result


def fit_dmdc(
    training: list[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ],
    volume: np.ndarray,
    rank: int,
) -> dict[str, np.ndarray | int]:
    all_temperature = np.concatenate([item[2] for item in training], axis=0)
    centre = all_temperature.mean(axis=0)
    weight = np.sqrt(volume / volume.mean())
    snapshots = ((all_temperature - centre) * weight).T
    left, singular, _ = np.linalg.svd(snapshots, full_matrices=False)
    usable = int(np.count_nonzero(singular > singular[0] * 1.0e-12)) if singular[0] > 0 else 1
    chosen_rank = min(rank, usable, left.shape[1])
    basis = left[:, :chosen_rank]

    condition_values = np.stack([item[1] for item in training])
    condition_mean = condition_values.mean(axis=0)
    condition_std = condition_values.std(axis=0)
    condition_std[condition_std < 1.0e-12] = 1.0
    midpoint_states, time_derivatives, controls = [], [], []
    for time_s, condition, temperature, _, _, _ in training:
        reduced = basis.T @ (((temperature - centre) * weight).T)
        delta_t = np.diff(time_s)
        midpoint_states.append(0.5 * (reduced[:, 1:] + reduced[:, :-1]))
        time_derivatives.append((reduced[:, 1:] - reduced[:, :-1]) / delta_t[None, :])
        control = (condition - condition_mean) / condition_std
        control = np.concatenate((control, np.ones(1)))
        controls.append(np.repeat(control[:, None], temperature.shape[0] - 1, axis=1))
    x = np.concatenate(midpoint_states, axis=1)
    y = np.concatenate(time_derivatives, axis=1)
    u = np.concatenate(controls, axis=1)
    omega = np.concatenate((x, u), axis=0)
    operator = np.linalg.lstsq(omega.T, y.T, rcond=None)[0].T
    return {
        "centre": centre,
        "weight": weight,
        "basis": basis,
        "A": operator[:, :chosen_rank],
        "B": operator[:, chosen_rank:],
        "condition_mean": condition_mean,
        "condition_std": condition_std,
        "rank": chosen_rank,
        "singular_values": singular,
        "time_form": "continuous_time_midpoint_derivative",
    }


def predict(
    model: dict, condition: np.ndarray, initial: np.ndarray, time_s: np.ndarray
) -> np.ndarray:
    delta_t = np.diff(np.asarray(time_s, dtype=np.float64))
    if len(delta_t) == 0 or np.any(~np.isfinite(delta_t)) or np.any(delta_t <= 0.0):
        raise ValueError("continuous-time DMDc prediction needs increasing physical times")
    centre = model["centre"]
    weight = model["weight"]
    basis = model["basis"]
    reduced = basis.T @ ((initial - centre) * weight)
    control = (condition - model["condition_mean"]) / model["condition_std"]
    control = np.concatenate((control, np.ones(1)))
    rank = int(model["rank"])
    augmented = np.zeros((rank + len(control), rank + len(control)), dtype=np.float64)
    augmented[:rank, :rank] = model["A"]
    augmented[:rank, rank:] = model["B"]
    augmented_state = np.concatenate((reduced, control))
    output = [initial.copy()]
    for step_size in delta_t:
        augmented_state = expm(augmented * step_size) @ augmented_state
        reduced = augmented_state[:rank]
        temperature = centre + (basis @ reduced) / weight
        output.append(temperature)
    return np.stack(output)


def evaluate(
    model: dict,
    root: Path,
    source_records: dict[str, dict],
    sequence_ids: list[str],
    node_type: np.ndarray,
    volume: np.ndarray,
    centroid: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, np.ndarray]]]:
    outputs = []
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
        prediction = predict(model, condition, target[0], time_s)
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
    parser.add_argument("--rank-candidates", default=",".join(map(str, DEFAULT_RANKS)))
    args = parser.parse_args()

    begin = time.perf_counter()
    index_path = args.dataset_index.resolve()
    root = index_path.parent
    index = json.loads(index_path.read_text(encoding="utf-8"))
    source_records = records(index)
    if len(source_records) != 12 or not all(bool(row["complete"]) for row in source_records.values()):
        raise ValueError("formal DMDc comparison requires all 12 complete thermal-step curves")
    split = split_lists(set(source_records), args.splits.resolve(), args.split_name)
    node_type, volume, centroid = regional_geometry(root, index)
    training = [load_sequence(root, source_records[value]) for value in split["train"]]
    time_steps = np.concatenate([np.diff(item[0]) for item in training])
    maximum_rank = min(len(node_type), sum(len(item[0]) for item in training))
    candidate_ranks = sorted(
        {int(value) for value in args.rank_candidates.split(",") if 0 < int(value) <= maximum_rank}
    )
    if not candidate_ranks:
        raise ValueError("no admissible DMDc rank candidate")

    validation_scan = []
    fitted = {}
    for rank in candidate_ranks:
        model = fit_dmdc(training, volume, rank)
        metrics, _ = evaluate(
            model, root, source_records, split["validation"], node_type, volume, centroid
        )
        validation_scan.append({"requested_rank": rank, "fitted_rank": int(model["rank"]), **metrics})
        fitted[rank] = model
    selected = min(validation_scan, key=lambda row: row["solid_temperature_RMSE_K"])
    selected_rank = int(selected["requested_rank"])
    model = fitted[selected_rank]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_role = {}
    prediction_files = {}
    for role in ("train", "validation", "test"):
        metrics, outputs = evaluate(
            model, root, source_records, split[role], node_type, volume, centroid
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
            temperature_prediction_K=np.stack([item["prediction"] for item in outputs]),
            temperature_target_K=np.stack([item["target"] for item in outputs]),
            node_type=node_type,
            node_volume_m3=volume,
            node_centroid_m=centroid,
        )
        prediction_files[role] = path.name
    np.savez_compressed(
        args.output_dir / "dmdc_model.npz",
        centre_temperature_K=model["centre"],
        volume_weight=model["weight"],
        pod_basis=model["basis"],
        state_operator=model["A"],
        control_operator=model["B"],
        condition_mean=model["condition_mean"],
        condition_std=model["condition_std"],
        singular_values=model["singular_values"],
    )
    model_storage_scalar_count = sum(
        int(np.asarray(model[name]).size)
        for name in (
            "centre",
            "weight",
            "basis",
            "A",
            "B",
            "condition_mean",
            "condition_std",
            "singular_values",
        )
    )
    summary = {
        "status": "completed_p418_regional_dmdc",
        "dataset_index": str(index_path),
        "split_name": args.split_name,
        "split_case_counts": {role: len(values) for role, values in split.items()},
        "split_case_ids": split,
        "temperature_metric_definition": "regional-volume-weighted RMSE, reported separately for fluid and solid",
        "time_step_range_s": [float(time_steps.min()), float(time_steps.max())],
        "time_step_unique_count": int(len(np.unique(time_steps))),
        "dynamics_form": model["time_form"],
        "time_integration": "exact augmented linear-system matrix exponential at each recorded interval",
        "rank_candidates": candidate_ranks,
        "validation_rank_scan": validation_scan,
        "selection_split": "validation",
        "selection_metric": "regional-volume-weighted solid-temperature RMSE in K",
        "selected_rank": int(model["rank"]),
        "model_storage_scalar_count": model_storage_scalar_count,
        "compute_device": "cpu_numpy",
        "metrics": metrics_by_role,
        "prediction_files": prediction_files,
        "training_seconds": time.perf_counter() - begin,
        "algorithm_source": DMD_CITATION,
        "new_physical_parameters": [],
        "scientific_scope": (
            "Classical continuous-time linear reduced-order baseline fitted only to training thermal-step curves. "
            "Midpoint derivatives admit the registered nonuniform physical output times, and prediction advances "
            "the augmented state-control system over each original interval without resampling. "
            "POD rank is selected using validation curves; test curves are not used in fitting or selection."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
