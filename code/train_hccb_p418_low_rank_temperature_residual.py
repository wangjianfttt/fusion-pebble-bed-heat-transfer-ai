#!/usr/bin/env python3
"""Training-only low-rank correction of deterministic P418 temperature residuals."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from hccb_p418_transient_hotspot_metrics import solid_transient_hotspot_metrics


ROLES = ("train", "validation", "test")


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    required = {
        "sequence_id",
        "time_s",
        "condition_physical",
        "condition_normalized",
        "fixed_hydrodynamics_physical",
        "fluid_internal_mass_flux_kg_s",
        "fluid_boundary_mass_flux_kg_s",
        "baseline_temperature_normalized",
        "target_temperature_normalized",
        "node_type",
        "node_volume_m3",
        "node_centroid_m",
        "temperature_mean_K_by_node_type",
        "temperature_std_K_by_node_type",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"deterministic prediction file lacks {sorted(missing)}")
        return {name: data[name].copy() for name in required}


def validate_deterministic_prediction_contract(
    *,
    summary: dict[str, object],
    splits: dict[str, dict[str, np.ndarray]],
    prediction_dir: Path,
    run_role: str,
) -> None:
    """Require the formal POD correction to use the complete physical dataset."""
    if run_role == "software_smoke":
        return
    if summary.get("status") != "completed_p418_spatiotemporal_regional_operator":
        raise ValueError("formal POD input is not a completed spatiotemporal model")
    if summary.get("run_role") not in {"formal", "formal_factorized"}:
        raise ValueError("formal POD input must come from a formal deterministic run")
    if summary.get("physics_mode") != "energy_and_flux":
        raise ValueError("formal POD input must be the physics-constrained model")
    if summary.get("selection_split") != "validation":
        raise ValueError("deterministic model was not selected with validation curves")
    if summary.get("new_physical_parameters") != []:
        raise ValueError("deterministic model unexpectedly introduced physical parameters")

    recorded = summary.get("split_case_ids", {})
    prediction_files = summary.get("temporal_temperature_prediction_files", {})
    seen: set[str] = set()
    for role in ROLES:
        actual = [str(value) for value in splits[role]["sequence_id"]]
        expected = [str(value) for value in recorded.get(role, [])]
        if actual != expected:
            raise ValueError(f"{role} POD curves differ from the deterministic split")
        overlap = seen.intersection(actual)
        if overlap:
            raise ValueError(f"complete curves overlap across POD roles: {sorted(overlap)}")
        seen.update(actual)
        expected_file = f"{role}_temporal_temperature_predictions.npz"
        if prediction_files.get(role) != expected_file:
            raise ValueError(f"deterministic summary records a different {role} prediction file")
        if not (prediction_dir / expected_file).is_file():
            raise FileNotFoundError(prediction_dir / expected_file)
    if len(seen) != 12:
        raise ValueError(
            f"formal POD correction requires all 12 thermal-step curves, found {len(seen)}"
        )


def feature_scale(data: dict[str, np.ndarray]) -> np.ndarray:
    node_type = data["node_type"].astype(np.int64)
    volume = data["node_volume_m3"].astype(np.float64)
    temperature_std = data["temperature_std_K_by_node_type"].astype(np.float64)
    if np.any(volume <= 0.0) or np.any(temperature_std <= 0.0):
        raise ValueError("node volume and temperature scale must be positive")
    return temperature_std[node_type] * np.sqrt(volume / volume.sum())


def weighted_residual_matrix(data: dict[str, np.ndarray], scale: np.ndarray) -> np.ndarray:
    baseline = data["baseline_temperature_normalized"].astype(np.float64)
    target = data["target_temperature_normalized"].astype(np.float64)
    if baseline.shape != target.shape or baseline.ndim != 4 or baseline.shape[-1] != 1:
        raise ValueError("temperature histories must have shape [curve,time,node,1]")
    residual = target - baseline
    residual[:, 0] = 0.0
    weighted = residual[..., 0] * scale[None, None, :]
    return weighted.reshape(len(weighted), -1)


def fit_low_rank_family(
    residual: np.ndarray,
    condition: np.ndarray,
) -> dict[str, np.ndarray | int]:
    if residual.ndim != 2 or condition.ndim != 2 or len(residual) != len(condition):
        raise ValueError("residual and condition arrays must contain the same training curves")
    mean = residual.mean(axis=0)
    centered = residual - mean
    _, singular, modes = np.linalg.svd(centered, full_matrices=False)
    tolerance = np.finfo(float).eps * max(centered.shape) * (singular[0] if len(singular) else 0.0)
    available_rank = min(len(residual) - 1, int(np.count_nonzero(singular > tolerance)))
    design = np.column_stack((np.ones(len(condition)), condition.astype(np.float64)))
    scores = centered @ modes[:available_rank].T
    regression = np.linalg.lstsq(design, scores, rcond=None)[0]
    return {
        "mean": mean,
        "modes": modes[:available_rank],
        "regression": regression,
        "available_rank": available_rank,
    }


def predict_weighted_residual(
    fitted: dict[str, np.ndarray | int], condition: np.ndarray, rank: int
) -> np.ndarray:
    available = int(fitted["available_rank"])
    if rank < 0 or rank > available:
        raise ValueError(f"rank {rank} is outside 0--{available}")
    mean = np.asarray(fitted["mean"], dtype=np.float64)
    prediction = np.broadcast_to(mean, (len(condition), len(mean))).copy()
    if rank:
        design = np.column_stack((np.ones(len(condition)), condition.astype(np.float64)))
        regression = np.asarray(fitted["regression"], dtype=np.float64)[:, :rank]
        modes = np.asarray(fitted["modes"], dtype=np.float64)[:rank]
        prediction += (design @ regression) @ modes
    return prediction


def corrected_temperature(
    data: dict[str, np.ndarray], weighted_residual: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    baseline = data["baseline_temperature_normalized"].astype(np.float64)
    curve_count, time_count, node_count, _ = baseline.shape
    weighted = weighted_residual.reshape(curve_count, time_count, node_count)
    normalized_residual = weighted / scale[None, None, :]
    normalized_residual[:, 0] = 0.0
    return baseline + normalized_residual[..., None]


def temperature_metrics(
    prediction: np.ndarray, data: dict[str, np.ndarray]
) -> dict[str, float]:
    target = data["target_temperature_normalized"].astype(np.float64)
    node_type = data["node_type"].astype(np.int64)
    volume = data["node_volume_m3"].astype(np.float64)
    temperature_std = data["temperature_std_K_by_node_type"].astype(np.float64)
    temperature_mean = data["temperature_mean_K_by_node_type"].astype(np.float64)
    error_k = (prediction - target)[..., 0] * temperature_std[node_type][None, None, :]

    def rmse(selected: np.ndarray) -> float:
        selected_volume = volume[selected]
        square = np.square(error_k[..., selected]) * selected_volume[None, None, :]
        return float(math.sqrt(square.sum() / (prediction.shape[0] * prediction.shape[1] * selected_volume.sum())))

    result = {
        "fluid_temperature_RMSE_K": rmse(node_type == 0),
        "solid_temperature_RMSE_K": rmse(node_type == 1),
        "maximum_absolute_temperature_error_K": float(np.max(np.abs(error_k))),
    }
    node_scale = temperature_std[node_type][None, None, :]
    node_offset = temperature_mean[node_type][None, None, :]
    prediction_k = prediction[..., 0] * node_scale + node_offset
    target_k = target[..., 0] * node_scale + node_offset
    result.update(
        solid_transient_hotspot_metrics(
            prediction_k,
            target_k,
            node_type,
            data["node_centroid_m"],
        )
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument(
        "--run-role",
        choices=("software_smoke", "formal"),
        default="formal",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    prediction_dir = args.prediction_dir.resolve()
    deterministic_summary_path = prediction_dir / "summary.json"
    if not deterministic_summary_path.is_file():
        raise FileNotFoundError(deterministic_summary_path)
    deterministic_summary = json.loads(
        deterministic_summary_path.read_text(encoding="utf-8")
    )
    data = {
        role: load_predictions(prediction_dir / f"{role}_temporal_temperature_predictions.npz")
        for role in ROLES
    }
    validate_deterministic_prediction_contract(
        summary=deterministic_summary,
        splits=data,
        prediction_dir=prediction_dir,
        run_role=args.run_role,
    )
    scale = feature_scale(data["train"])
    for role in ROLES[1:]:
        if not np.array_equal(data[role]["node_type"], data["train"]["node_type"]):
            raise ValueError(f"node types differ in {role}")
        if not np.allclose(data[role]["node_volume_m3"], data["train"]["node_volume_m3"]):
            raise ValueError(f"node volumes differ in {role}")
    training_residual = weighted_residual_matrix(data["train"], scale)
    fitted = fit_low_rank_family(training_residual, data["train"]["condition_normalized"])
    validation_candidates = []
    for rank in range(int(fitted["available_rank"]) + 1):
        residual = predict_weighted_residual(fitted, data["validation"]["condition_normalized"], rank)
        prediction = corrected_temperature(data["validation"], residual, scale)
        metrics = temperature_metrics(prediction, data["validation"])
        validation_candidates.append({"rank": rank, **metrics})
    selected_rank = min(
        validation_candidates,
        key=lambda row: (row["solid_temperature_RMSE_K"], row["rank"]),
    )["rank"]
    training_seconds = time.perf_counter() - started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    prediction_files = {}
    inference_started = time.perf_counter()
    for role in ROLES:
        role_started = time.perf_counter()
        residual = predict_weighted_residual(
            fitted, data[role]["condition_normalized"], selected_rank
        )
        prediction = corrected_temperature(data[role], residual, scale)
        role_seconds = time.perf_counter() - role_started
        role_metrics = temperature_metrics(prediction, data[role])
        role_metrics.update(
            {
                "inference_seconds": role_seconds,
                "inference_seconds_per_curve": role_seconds / len(prediction),
            }
        )
        metrics[role] = role_metrics
        path = args.output_dir / f"{role}_low_rank_temperature_predictions.npz"
        np.savez_compressed(
            path,
            sequence_id=data[role]["sequence_id"],
            time_s=data[role]["time_s"],
            condition_physical=data[role]["condition_physical"],
            fixed_hydrodynamics_physical=data[role]["fixed_hydrodynamics_physical"],
            fluid_internal_mass_flux_kg_s=data[role][
                "fluid_internal_mass_flux_kg_s"
            ],
            fluid_boundary_mass_flux_kg_s=data[role][
                "fluid_boundary_mass_flux_kg_s"
            ],
            corrected_temperature_normalized=prediction.astype(np.float32),
            baseline_temperature_normalized=data[role]["baseline_temperature_normalized"],
            target_temperature_normalized=data[role]["target_temperature_normalized"],
            node_type=data[role]["node_type"],
            node_volume_m3=data[role]["node_volume_m3"],
            node_centroid_m=data[role]["node_centroid_m"],
            temperature_mean_K_by_node_type=data[role][
                "temperature_mean_K_by_node_type"
            ],
            temperature_std_K_by_node_type=data[role][
                "temperature_std_K_by_node_type"
            ],
        )
        prediction_files[role] = path.name
    inference_seconds = time.perf_counter() - inference_started

    mean = np.asarray(fitted["mean"])
    modes = np.asarray(fitted["modes"])[:selected_rank]
    regression = np.asarray(fitted["regression"])[:, :selected_rank]
    np.savez_compressed(
        args.output_dir / "low_rank_residual_model.npz",
        weighted_mean_residual=mean.astype(np.float32),
        weighted_spatial_temporal_modes=modes.astype(np.float32),
        condition_to_mode_coefficients=regression.astype(np.float32),
        selected_rank=np.asarray(selected_rank, dtype=np.int64),
    )
    summary = {
        "status": "completed_p418_low_rank_temperature_residual",
        "run_role": args.run_role,
        "split_name": args.split_name,
        "upstream_training_seed": deterministic_summary.get("seed"),
        "split_case_ids": deterministic_summary["split_case_ids"],
        "deterministic_prediction_dir": str(prediction_dir),
        "selection_split": "validation",
        "selection_metric": "validation regional-volume-weighted solid-temperature RMSE in K",
        "temperature_metric_definition": (
            "regional-volume-weighted RMSE, reported separately for fluid and solid"
        ),
        "selected_rank": int(selected_rank),
        "available_training_rank": int(fitted["available_rank"]),
        "validation_rank_candidates": validation_candidates,
        "metrics": metrics,
        "training_seconds": training_seconds,
        "total_inference_seconds": inference_seconds,
        "model_storage_scalar_count": int(mean.size + modes.size + regression.size),
        "model_size_definition": "stored low-rank mean, modes and condition coefficients",
        "compute_device": "cpu",
        "prediction_files": prediction_files,
        "initial_temperature_constraint": "correction is exactly zero at t=0",
        "regression": "ordinary least-squares minimum-norm solution; no fitted physical parameters",
        "new_physical_parameters": [],
        "scientific_scope": (
            "Deterministic low-rank residual baseline used to test whether diffusion adds value "
            "beyond repeatable training-curve discrepancy."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
