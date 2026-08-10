#!/usr/bin/env python3
"""Fit a continuous-time DMDc baseline to the six P418 outlet observables.

This baseline uses exactly the same condition vectors, complete-curve splits,
physical time coordinates and six targets as the observable Transformer.
Training curves fit the model, validation curves select the reduced rank, and
test curves are used only once for final evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from train_hccb_p418_transient_observable_transformer import (
    TARGET_NAMES,
    select_columns,
    split_indices,
)


DEFAULT_RANKS = (1, 2, 3, 4, 5, 6)
DMD_CITATION = {
    "paper": "Dynamic Mode Decomposition with Control",
    "authors": "Joshua L. Proctor, Steven L. Brunton and J. Nathan Kutz",
    "venue": "SIAM Journal on Applied Dynamical Systems 15 (2016) 142-161",
    "doi": "10.1137/15M1013857",
    "paper_url": "https://epubs.siam.org/doi/10.1137/15M1013857",
}


def load_formal_data(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=True) as data:
        case_ids_all = [str(value) for value in data["case_id"]]
        complete = data["complete"].astype(bool)
        time_mask_all = data["time_mask"].astype(bool)
        fully_reported = time_mask_all.sum(axis=1) == time_mask_all.shape[1]
        available = np.flatnonzero(complete & fully_reported)
        if len(available) != 12:
            raise ValueError(
                "formal physical-step observable DMDc requires 12 complete curves, "
                f"found {len(available)}"
            )
        return {
            "case_ids": [case_ids_all[index] for index in available],
            "conditions": data["conditions"][available].astype(np.float64),
            "condition_names": [str(value) for value in data["condition_names"]],
            "time_s": data["time_s"][available].astype(np.float64),
            "time_mask": time_mask_all[available],
            "targets": select_columns(data)[available].astype(np.float64),
        }


def fit_model(
    conditions: np.ndarray,
    time_s: np.ndarray,
    targets: np.ndarray,
    indices: list[int],
    rank: int,
) -> dict[str, np.ndarray | int | str]:
    training = targets[indices]
    target_mean = training.reshape(-1, training.shape[-1]).mean(axis=0)
    target_std = training.reshape(-1, training.shape[-1]).std(axis=0)
    target_std[target_std < 1.0e-12] = 1.0
    normalized = (training - target_mean) / target_std

    snapshots = normalized.reshape(-1, normalized.shape[-1]).T
    left, singular, _ = np.linalg.svd(snapshots, full_matrices=False)
    usable = (
        int(np.count_nonzero(singular > singular[0] * 1.0e-12))
        if singular[0] > 0.0
        else 1
    )
    chosen_rank = min(int(rank), usable, left.shape[1])
    basis = left[:, :chosen_rank]

    condition_train = conditions[indices]
    condition_mean = condition_train.mean(axis=0)
    condition_std = condition_train.std(axis=0)
    condition_std[condition_std < 1.0e-12] = 1.0

    midpoint_states: list[np.ndarray] = []
    derivatives: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    for local_index, case_index in enumerate(indices):
        reduced = basis.T @ normalized[local_index].T
        delta_t = np.diff(time_s[case_index])
        if np.any(~np.isfinite(delta_t)) or np.any(delta_t <= 0.0):
            raise ValueError("physical times must be finite and strictly increasing")
        midpoint_states.append(0.5 * (reduced[:, 1:] + reduced[:, :-1]))
        derivatives.append((reduced[:, 1:] - reduced[:, :-1]) / delta_t[None, :])
        control = (conditions[case_index] - condition_mean) / condition_std
        control = np.concatenate((control, np.ones(1)))
        controls.append(np.repeat(control[:, None], len(delta_t), axis=1))

    state_matrix = np.concatenate(midpoint_states, axis=1)
    derivative_matrix = np.concatenate(derivatives, axis=1)
    control_matrix = np.concatenate(controls, axis=1)
    regression = np.concatenate((state_matrix, control_matrix), axis=0)
    operator = np.linalg.lstsq(
        regression.T, derivative_matrix.T, rcond=None
    )[0].T
    return {
        "target_mean": target_mean,
        "target_std": target_std,
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
    model: dict[str, np.ndarray | int | str],
    condition: np.ndarray,
    initial_target: np.ndarray,
    time_s: np.ndarray,
) -> np.ndarray:
    delta_t = np.diff(time_s)
    if np.any(~np.isfinite(delta_t)) or np.any(delta_t <= 0.0):
        raise ValueError("prediction times must be finite and strictly increasing")
    target_mean = np.asarray(model["target_mean"])
    target_std = np.asarray(model["target_std"])
    basis = np.asarray(model["basis"])
    reduced = basis.T @ ((initial_target - target_mean) / target_std)
    control = (
        (condition - np.asarray(model["condition_mean"]))
        / np.asarray(model["condition_std"])
    )
    control = np.concatenate((control, np.ones(1)))
    rank = int(model["rank"])
    augmented = np.zeros((rank + len(control), rank + len(control)))
    augmented[:rank, :rank] = np.asarray(model["A"])
    augmented[:rank, rank:] = np.asarray(model["B"])
    augmented_state = np.concatenate((reduced, control))
    output = [initial_target.copy()]
    transition_cache: dict[float, np.ndarray] = {}
    for step_size in delta_t:
        cache_key = float(np.round(step_size, 12))
        transition = transition_cache.get(cache_key)
        if transition is None:
            transition = expm(augmented * float(step_size))
            transition_cache[cache_key] = transition
        augmented_state = transition @ augmented_state
        normalized_target = basis @ augmented_state[:rank]
        target = target_mean + target_std * normalized_target
        if not np.all(np.isfinite(target)):
            raise FloatingPointError("observable DMDc prediction became non-finite")
        output.append(target)
    return np.stack(output)


def evaluate(
    model: dict[str, np.ndarray | int | str],
    conditions: np.ndarray,
    time_s: np.ndarray,
    targets: np.ndarray,
    indices: list[int],
) -> tuple[dict[str, float], np.ndarray]:
    begin = time.perf_counter()
    predictions = np.stack(
        [
            predict(
                model,
                conditions[index],
                targets[index, 0],
                time_s[index],
            )
            for index in indices
        ]
    )
    elapsed = time.perf_counter() - begin
    truth = targets[indices]
    residual = predictions - truth
    normalized = residual / np.asarray(model["target_std"])
    metrics = {
        "normalized_trajectory_MSE": float(np.mean(normalized**2)),
        "inference_seconds": elapsed,
        "inference_seconds_per_curve": elapsed / len(indices),
    }
    for target_index, name in enumerate(TARGET_NAMES):
        metrics[f"{name}_RMSE"] = float(
            np.sqrt(np.mean(residual[..., target_index] ** 2))
        )
    return metrics, predictions


def casewise_rows(
    case_ids: list[str],
    indices: list[int],
    truth: np.ndarray,
    prediction: np.ndarray,
    role: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for local_index, case_index in enumerate(indices):
        for target_index, name in enumerate(TARGET_NAMES):
            residual = (
                prediction[local_index, :, target_index]
                - truth[case_index, :, target_index]
            )
            rows.append(
                {
                    "split": role,
                    "condition_id": case_ids[case_index],
                    "target": name,
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                    "maximum_absolute_error": float(np.max(np.abs(residual))),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rank-candidates", default=",".join(map(str, DEFAULT_RANKS))
    )
    args = parser.parse_args()

    begin = time.perf_counter()
    source = load_formal_data(args.data.resolve())
    case_ids = list(source["case_ids"])
    conditions = np.asarray(source["conditions"])
    time_s = np.asarray(source["time_s"])
    targets = np.asarray(source["targets"])
    split = split_indices(
        case_ids,
        args.splits.resolve(),
        args.split_name,
        require_complete=True,
    )
    rank_candidates = sorted(
        {
            int(value)
            for value in args.rank_candidates.split(",")
            if 0 < int(value) <= len(TARGET_NAMES)
        }
    )
    if not rank_candidates:
        raise ValueError("no admissible observable DMDc rank candidate")

    validation_scan: list[dict[str, object]] = []
    fitted: dict[int, dict[str, np.ndarray | int | str]] = {}
    for rank in rank_candidates:
        model = fit_model(
            conditions, time_s, targets, split["train"], rank
        )
        try:
            with np.errstate(over="raise", invalid="raise"):
                metrics, _ = evaluate(
                    model,
                    conditions,
                    time_s,
                    targets,
                    split["validation"],
                )
        except (FloatingPointError, OverflowError, ValueError) as error:
            validation_scan.append(
                {
                    "requested_rank": rank,
                    "fitted_rank": int(model["rank"]),
                    "stable_on_validation_curves": False,
                    "failure_reason": f"{type(error).__name__}: {error}",
                }
            )
            continue
        validation_scan.append(
            {
                "requested_rank": rank,
                "fitted_rank": int(model["rank"]),
                "stable_on_validation_curves": True,
                **metrics,
            }
        )
        fitted[rank] = model

    stable = [
        row for row in validation_scan if row["stable_on_validation_curves"]
    ]
    if not stable:
        raise ValueError("all observable DMDc ranks are unstable on validation curves")
    selected = min(stable, key=lambda row: row["normalized_trajectory_MSE"])
    selected_requested_rank = int(selected["requested_rank"])
    model = fitted[selected_requested_rank]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_role: dict[str, dict[str, float]] = {}
    prediction_files: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    for role in ("train", "validation", "test"):
        metrics, prediction = evaluate(
            model, conditions, time_s, targets, split[role]
        )
        metrics_by_role[role] = metrics
        path = args.output_dir / f"{role}_observable_predictions.npz"
        np.savez_compressed(
            path,
            case_id=np.asarray([case_ids[index] for index in split[role]]),
            time_s=time_s[split[role]],
            target=targets[split[role]].astype(np.float32),
            prediction=prediction.astype(np.float32),
            target_names=np.asarray(TARGET_NAMES),
        )
        prediction_files[role] = path.name
        rows.extend(
            casewise_rows(
                case_ids, split[role], targets, prediction, role
            )
        )

    with (args.output_dir / "casewise_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        args.output_dir / "dmdc_model.npz",
        target_mean=np.asarray(model["target_mean"]),
        target_std=np.asarray(model["target_std"]),
        pod_basis=np.asarray(model["basis"]),
        state_operator=np.asarray(model["A"]),
        control_operator=np.asarray(model["B"]),
        condition_mean=np.asarray(model["condition_mean"]),
        condition_std=np.asarray(model["condition_std"]),
        singular_values=np.asarray(model["singular_values"]),
    )
    storage_count = sum(
        int(np.asarray(model[name]).size)
        for name in (
            "target_mean",
            "target_std",
            "basis",
            "A",
            "B",
            "condition_mean",
            "condition_std",
            "singular_values",
        )
    )
    summary = {
        "status": "completed_p418_observable_dmdc",
        "data": str(args.data.resolve()),
        "split_name": args.split_name,
        "split_case_counts": {
            role: len(indices) for role, indices in split.items()
        },
        "split_case_ids": {
            role: [case_ids[index] for index in indices]
            for role, indices in split.items()
        },
        "target_names": TARGET_NAMES,
        "condition_names": source["condition_names"],
        "selection_split": "validation",
        "selection_metric": "normalized trajectory MSE across the same six observable targets",
        "rank_candidates": rank_candidates,
        "validation_rank_scan": validation_scan,
        "selected_rank": int(model["rank"]),
        "selected_requested_rank": selected_requested_rank,
        "dynamics_form": model["time_form"],
        "time_integration": (
            "exact augmented linear-system matrix exponential at each original interval"
        ),
        "model_storage_scalar_count": storage_count,
        "compute_device": "cpu_numpy",
        "training_seconds": time.perf_counter() - begin,
        "metrics": metrics_by_role,
        "prediction_files": prediction_files,
        "algorithm_source": DMD_CITATION,
        "new_physical_parameters": [],
        "scientific_scope": (
            "Direct classical baseline for the observable Transformer. It uses the "
            "same six condition-dependent outlet and thermal trajectories, identical "
            "complete-curve splits and original nonuniform physical times. Training "
            "curves fit the continuous-time model, validation curves select rank and "
            "test curves are excluded from fitting and selection."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
