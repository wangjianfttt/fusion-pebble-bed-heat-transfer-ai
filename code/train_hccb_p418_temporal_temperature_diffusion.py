#!/usr/bin/env python3
"""Train temporal diffusion correction after the deterministic P418 operator."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from hccb_p418_regional_cht_adapter import load_p418_subface_geometry
from hccb_p418_regional_diffusion_refiner import make_velocity_training_pair
from hccb_p418_temporal_temperature_diffusion import (
    FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE,
    P418TemporalTemperatureResidualRefiner,
    sample_temporal_temperature_residual,
)
from hccb_p418_transient_hotspot_metrics import solid_transient_hotspot_metrics
from hccb_p418_transient_regional_physics import (
    assemble_p418_transient_regional_residual,
    dimensionless_transient_energy_loss,
)


FORMAL_TRAINING = {
    "epochs": 500,
    "batch_size": 8,
    "microbatch_size": 1,
    "activation_precision": "bfloat16",
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-5,
    "ema_decay": 0.995,
    "spatial_time_chunk_size": 1,
    "temporal_node_chunk_size": 2048,
}


def write_inference_progress(path: Path, payload: dict[str, object]) -> None:
    """Atomically expose long ensemble-inference progress without changing results."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
        "structural_features",
        "temperature_mean_K_by_node_type",
        "temperature_std_K_by_node_type",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"deterministic prediction file lacks {sorted(missing)}")
        return {name: data[name].copy() for name in required}


def training_time_scale(splits: dict[str, dict[str, np.ndarray]]) -> float:
    time_s = np.asarray(splits["train"]["time_s"], dtype=np.float64)
    if time_s.size == 0 or not np.all(np.isfinite(time_s)):
        raise ValueError("training curves contain invalid time values")
    if float(time_s.min()) < 0.0 or float(time_s.max()) <= 0.0:
        raise ValueError("training curves must start at non-negative time and extend above zero")
    return float(time_s.max())


def validate_deterministic_prediction_contract(
    *,
    summary: dict[str, object],
    splits: dict[str, dict[str, np.ndarray]],
    prediction_dir: Path,
    run_role: str,
) -> None:
    if not summary:
        if run_role == "software_smoke":
            return
        raise ValueError("formal diffusion training lacks the deterministic summary")
    if summary.get("status") != "completed_p418_spatiotemporal_regional_operator":
        raise ValueError("diffusion input is not a completed spatiotemporal model")
    if summary.get("run_role") not in {"formal", "formal_factorized"}:
        raise ValueError(
            "formal diffusion input must come from a formal deterministic run"
        )
    if summary.get("physics_mode") != "energy_and_flux":
        raise ValueError("formal diffusion input must be the physics-constrained model")
    if summary.get("selection_split") != "validation":
        raise ValueError("deterministic model was not selected with validation curves")
    if summary.get("new_physical_parameters") != []:
        raise ValueError("deterministic model unexpectedly introduced physical parameters")

    recorded = summary.get("split_case_ids", {})
    prediction_files = summary.get("temporal_temperature_prediction_files", {})
    seen: set[str] = set()
    for role in ("train", "validation", "test"):
        actual = [str(value) for value in splits[role]["sequence_id"]]
        expected = [str(value) for value in recorded.get(role, [])]
        if actual != expected:
            raise ValueError(f"{role} diffusion curves differ from the deterministic split")
        overlap = seen.intersection(actual)
        if overlap:
            raise ValueError(f"complete curves overlap across diffusion roles: {sorted(overlap)}")
        seen.update(actual)
        expected_file = f"{role}_temporal_temperature_predictions.npz"
        if prediction_files.get(role) != expected_file:
            raise ValueError(f"deterministic summary records a different {role} prediction file")
        if not (prediction_dir / expected_file).is_file():
            raise FileNotFoundError(prediction_dir / expected_file)
    if len(seen) != 12:
        raise ValueError(
            f"formal diffusion correction requires all 12 thermal-step curves, found {len(seen)}"
        )


def residual_scale(
    residual: np.ndarray, node_volume: np.ndarray
) -> float:
    square = np.square(residual[..., 0], dtype=np.float64)
    mean_square = np.sum(square * node_volume[None, None, :]) / (
        residual.shape[0] * residual.shape[1] * np.sum(node_volume)
    )
    return max(float(math.sqrt(mean_square)), 1.0e-6)


def weighted_velocity_loss(
    prediction: torch.Tensor, target: torch.Tensor, volume: torch.Tensor
) -> torch.Tensor:
    square = (prediction - target).square()[..., 0]
    return (square * volume[None, None, :]).sum() / (
        prediction.shape[0] * prediction.shape[1] * volume.sum()
    )


def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
            ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)
        for ema_buffer, buffer in zip(ema.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def diffusion_checkpoint_contract(
    args: argparse.Namespace,
    prediction_dir: Path,
    splits: dict[str, dict[str, np.ndarray]],
) -> dict[str, object]:
    """Return the data split and settings that must match after a restart."""
    return {
        "prediction_dir": str(prediction_dir),
        "split_case_ids": {
            role: [str(value) for value in data["sequence_id"]]
            for role, data in splits.items()
        },
        "run_role": args.run_role,
        "observation_mask": (
            str(args.observation_mask.resolve())
            if args.observation_mask is not None
            else None
        ),
        "observation_source": args.observation_source,
        "residual_geometry": (
            str(args.residual_geometry.resolve())
            if args.residual_geometry is not None
            else None
        ),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "microbatch_size": args.microbatch_size,
        "activation_precision": args.activation_precision,
        "hidden_dim": args.hidden_dim,
        "spatial_layers": args.spatial_layers,
        "spatial_attention_heads": args.spatial_attention_heads,
        "physics_slices": args.physics_slices,
        "temporal_layers": args.temporal_layers,
        "temporal_heads": args.temporal_heads,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "ema_decay": args.ema_decay,
        "spatial_time_chunk_size": args.spatial_time_chunk_size,
        "temporal_node_chunk_size": args.temporal_node_chunk_size,
        "seed": args.seed,
    }


def save_diffusion_training_checkpoint(
    path: Path,
    *,
    contract: dict[str, object],
    next_epoch: int,
    model: torch.nn.Module,
    ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    order_generator: torch.Generator,
    diffusion_generator: torch.Generator,
    best_validation: float,
    best_epoch: int | None,
    best_ema_state: dict[str, torch.Tensor] | None,
    history: list[dict[str, object]],
    validation_history: list[dict[str, float]],
    training_seconds: float,
) -> None:
    """Atomically save every state needed to continue the next epoch."""
    payload = {
        "contract": contract,
        "next_epoch": next_epoch,
        "model_state": model.state_dict(),
        "ema_state": ema.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "order_generator_state": order_generator.get_state(),
        "diffusion_generator_state": diffusion_generator.get_state(),
        "best_validation": best_validation,
        "best_epoch": best_epoch,
        "best_ema_state": best_ema_state,
        "history": history,
        "validation_history": validation_history,
        "training_seconds": training_seconds,
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
        "cuda_random_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_diffusion_training_checkpoint(
    path: Path,
    *,
    contract: dict[str, object],
    model: torch.nn.Module,
    ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    order_generator: torch.Generator,
    diffusion_generator: torch.Generator,
    device: torch.device,
) -> dict[str, object]:
    """Restore a completed-epoch checkpoint after checking data and settings."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("contract") != contract:
        raise ValueError(
            "diffusion checkpoint does not match the current data split or model settings"
        )
    model.load_state_dict(payload["model_state"])
    ema.load_state_dict(payload["ema_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    order_generator.set_state(payload["order_generator_state"].cpu())
    diffusion_generator.set_state(payload["diffusion_generator_state"].cpu())
    np.random.set_state(payload["numpy_random_state"])
    torch.random.set_rng_state(payload["torch_random_state"].cpu())
    cuda_state = payload.get("cuda_random_state_all")
    if device.type == "cuda" and cuda_state is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_state])
    return payload


def temperature_rmse_k(
    prediction: np.ndarray,
    target: np.ndarray,
    node_type: np.ndarray,
    node_volume: np.ndarray,
    temperature_std_by_type: np.ndarray,
    material: int | None = None,
) -> float:
    selected = np.ones(len(node_type), dtype=bool) if material is None else node_type == material
    scale = temperature_std_by_type[node_type[selected]]
    error_k = (
        prediction[..., selected, 0] - target[..., selected, 0]
    ) * scale[None, None, :]
    selected_volume = node_volume[selected]
    mse = np.sum(np.square(error_k) * selected_volume[None, None, :]) / (
        prediction.shape[0] * prediction.shape[1] * np.sum(selected_volume)
    )
    return float(math.sqrt(mse))


def interval_weighted_sums(
    *,
    covered: np.ndarray,
    width_k: np.ndarray,
    predictive_std_k: np.ndarray,
    node_volume: np.ndarray,
    selection: np.ndarray,
) -> np.ndarray:
    """Return covered, total, width and standard-deviation volume sums."""
    if covered.shape != width_k.shape or covered.shape != predictive_std_k.shape:
        raise ValueError("interval arrays must have identical [..,time,node] shapes")
    if selection.shape != covered.shape or selection.dtype != bool:
        raise ValueError("interval selection must be boolean and match the interval arrays")
    if covered.shape[-1] != len(node_volume):
        raise ValueError("interval node count differs from node volumes")
    weights = selection * node_volume.reshape((1,) * (selection.ndim - 1) + (-1,))
    return np.asarray(
        (
            np.sum(covered * weights),
            np.sum(weights),
            np.sum(width_k * weights),
            np.sum(predictive_std_k * weights),
        ),
        dtype=np.float64,
    )


def finalized_interval_metrics(
    sums: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float | None]:
    if sums.shape != (4,) or not np.all(np.isfinite(sums)) or sums[1] < 0.0:
        raise ValueError("invalid interval statistics")
    if sums[1] == 0.0:
        return {
            f"{prefix}_90pct_interval_coverage_fraction": None,
            f"{prefix}_90pct_interval_mean_width_K": None,
            f"{prefix}_predictive_std_K_volume_mean": None,
        }
    return {
        f"{prefix}_90pct_interval_coverage_fraction": float(sums[0] / sums[1]),
        f"{prefix}_90pct_interval_mean_width_K": float(sums[2] / sums[1]),
        f"{prefix}_predictive_std_K_volume_mean": float(sums[3] / sums[1]),
    }


def ensemble_crps_k(
    members: np.ndarray,
    target: np.ndarray,
    temperature_scale_by_node: np.ndarray,
) -> np.ndarray:
    """Compute ensemble CRPS in kelvin with an O(S log S) sorted formula."""
    if members.ndim != target.ndim + 1 or members.shape[1:] != target.shape:
        raise ValueError("ensemble members must add one sample axis to the target")
    if target.shape[-1] != 1 or target.shape[-2] != len(temperature_scale_by_node):
        raise ValueError("temperature target or node scale has an incompatible shape")
    scale_shape = (1,) * (members.ndim - 2) + (-1,)
    physical_members = members[..., 0] * temperature_scale_by_node.reshape(
        scale_shape
    )
    target_scale_shape = (1,) * (target.ndim - 2) + (-1,)
    physical_target = target[..., 0] * temperature_scale_by_node.reshape(
        target_scale_shape
    )
    first_term = np.mean(
        np.abs(physical_members - physical_target[None, ...]), axis=0
    )
    ordered = np.sort(physical_members, axis=0)
    sample_count = members.shape[0]
    coefficients = (
        2.0 * np.arange(1, sample_count + 1, dtype=np.float64)
        - sample_count
        - 1.0
    ).reshape((sample_count,) + (1,) * (ordered.ndim - 1))
    pair_term = np.sum(coefficients * ordered, axis=0) / float(sample_count**2)
    return first_term - pair_term


def weighted_selected_sums(
    values: np.ndarray,
    node_volume: np.ndarray,
    selection: np.ndarray,
) -> np.ndarray:
    if values.shape != selection.shape or selection.dtype != bool:
        raise ValueError("selected values and boolean selection must have equal shape")
    if values.shape[-1] != len(node_volume):
        raise ValueError("selected values differ from the node-volume array")
    weights = selection * node_volume.reshape((1,) * (selection.ndim - 1) + (-1,))
    return np.asarray((np.sum(values * weights), np.sum(weights)), dtype=np.float64)


def finalized_selected_mean(sums: np.ndarray) -> float | None:
    if sums.shape != (2,) or not np.all(np.isfinite(sums)) or sums[1] < 0.0:
        raise ValueError("invalid selected-value sums")
    return None if sums[1] == 0.0 else float(sums[0] / sums[1])


def validate_observation_input(
    run_role: str,
    observation_mask: Path | None,
    observation_source: str,
) -> None:
    if observation_mask is None:
        if observation_source != "none":
            raise ValueError("an observation source cannot be declared without a mask file")
        if run_role == "sparse_reconstruction":
            raise ValueError("sparse reconstruction requires a declared observation mask")
        return
    if run_role != "sparse_reconstruction":
        raise ValueError("an observation mask is only accepted for sparse reconstruction")
    if observation_source == "none":
        raise ValueError("an observation mask requires an explicit observation source")
    if observation_source == "external_experiment":
        raise ValueError(
            "this trainer does not yet ingest measured temperatures, measurement "
            "uncertainty or sensor response; external experiments cannot be replaced "
            "by exact OpenFOAM target values"
        )
    if observation_source != "computed_openfoam_target":
        raise ValueError(f"unsupported observation source {observation_source!r}")


def observation_masks(
    path: Path | None,
    splits: dict[str, dict[str, np.ndarray]],
    observation_source: str = "none",
) -> dict[str, np.ndarray]:
    if path is None:
        if observation_source != "none":
            raise ValueError("an observation source cannot be declared without a mask file")
        return {
            role: np.zeros_like(data["baseline_temperature_normalized"], dtype=bool)
            for role, data in splits.items()
        }
    if observation_source != "computed_openfoam_target":
        raise ValueError(
            "current mask files are accepted only for computed OpenFOAM target points"
        )
    with np.load(path, allow_pickle=False) as data:
        required_metadata = {
            "observation_source_kind": "computed_openfoam_target",
            "observed_values_kind": (
                "target_temperature_normalized_from_openfoam_reference"
            ),
        }
        for key, expected in required_metadata.items():
            if key not in data.files:
                raise ValueError(f"observation-mask file lacks {key}")
            actual = str(np.asarray(data[key]).item())
            if actual != expected:
                raise ValueError(
                    f"observation-mask {key} is {actual!r}, expected {expected!r}"
                )
        result = {}
        for role, values in splits.items():
            key = f"{role}_mask"
            if key not in data.files:
                raise ValueError(f"observation-mask file lacks {key}")
            mask = data[key].astype(bool)
            if mask.shape != values["baseline_temperature_normalized"].shape:
                raise ValueError(f"{key} shape differs from {role} temperature histories")
            result[role] = mask
    return result


def unobserved_dynamic_selection(
    observation_mask: np.ndarray,
    node_type: np.ndarray,
    *,
    material: int | None = None,
) -> np.ndarray:
    """Select dynamic locations that were not supplied as observations."""
    if observation_mask.ndim != 3 or observation_mask.shape[-1] != 1:
        raise ValueError("one curve observation mask must have shape [time,node,1]")
    if observation_mask.dtype != bool or observation_mask.shape[1] != len(node_type):
        raise ValueError("observation mask and node types are incompatible")
    selected = ~observation_mask[..., 0].copy()
    selected[0, :] = False
    if material is not None:
        selected &= node_type[None, :] == material
    return selected


def physical_temperature_state(
    temperature_normalized: np.ndarray,
    fixed_hydrodynamics: np.ndarray,
    node_type: np.ndarray,
    temperature_mean_by_type: np.ndarray,
    temperature_std_by_type: np.ndarray,
) -> np.ndarray:
    """Combine one temperature history with the fixed target velocity/pressure field."""
    if temperature_normalized.ndim != 3 or temperature_normalized.shape[-1] != 1:
        raise ValueError("temperature history must have shape [time,node,1]")
    if fixed_hydrodynamics.shape != (temperature_normalized.shape[1], 4):
        raise ValueError("fixed hydrodynamics must have shape [node,4]")
    temperature_k = (
        temperature_normalized[..., 0]
        * temperature_std_by_type[node_type][None, :]
        + temperature_mean_by_type[node_type][None, :]
    )
    hydrodynamics = np.broadcast_to(
        fixed_hydrodynamics[None, :, :],
        (temperature_normalized.shape[0], temperature_normalized.shape[1], 4),
    )
    return np.concatenate((hydrodynamics, temperature_k[..., None]), axis=-1).astype(
        np.float32
    )


def energy_residual_summary(
    deterministic_mse: float, refined_mse: float, reference_mse: float
) -> dict[str, float]:
    """Return comparable energy-equation residual magnitudes and ratios."""
    values = (deterministic_mse, refined_mse, reference_mse)
    if any(value < 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("energy residual mean squares must be finite and non-negative")
    tiny = np.finfo(np.float64).tiny
    deterministic_rmse = math.sqrt(deterministic_mse)
    refined_rmse = math.sqrt(refined_mse)
    reference_rmse = math.sqrt(reference_mse)
    return {
        "deterministic_absolute_energy_equation_normalized_RMSE": deterministic_rmse,
        "diffusion_refined_absolute_energy_equation_normalized_RMSE": refined_rmse,
        "openfoam_reference_absolute_energy_equation_normalized_RMSE": reference_rmse,
        "diffusion_to_deterministic_energy_residual_ratio": refined_rmse
        / max(deterministic_rmse, tiny),
        "diffusion_to_openfoam_reference_energy_residual_ratio": refined_rmse
        / max(reference_rmse, tiny),
    }


def projection_aware_energy_difference_mse(
    predicted_residual,
    reference_residual,
    step_condition: torch.Tensor,
    fluid_volume_m3: torch.Tensor,
    solid_volume_m3: torch.Tensor,
) -> torch.Tensor:
    """Volume-weighted energy-equation difference from the regional reference."""
    source_scale = (step_condition[:, 5] * 1.0e6).clamp_min(
        torch.finfo(step_condition.dtype).tiny
    )
    fluid_difference = (
        predicted_residual.fluid_energy_w_m3
        - reference_residual.fluid_energy_w_m3
    ) / source_scale[:, None, None]
    solid_difference = (
        predicted_residual.solid_energy_w_m3
        - reference_residual.solid_energy_w_m3
    ) / source_scale[:, None, None]

    def weighted_mean_square(values: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
        weights = volume / volume.sum()
        return (values.square() * weights).sum(dim=-1).mean()

    return 0.5 * (
        weighted_mean_square(fluid_difference, fluid_volume_m3)
        + weighted_mean_square(solid_difference, solid_volume_m3)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--observation-mask", type=Path)
    parser.add_argument(
        "--observation-source",
        choices=("none", "computed_openfoam_target", "external_experiment"),
        default="none",
    )
    parser.add_argument("--residual-geometry", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run-role",
        choices=("software_smoke", "computed_residual_benchmark", "sparse_reconstruction"),
        default="computed_residual_benchmark",
    )
    parser.add_argument("--epochs", type=int, default=FORMAL_TRAINING["epochs"])
    parser.add_argument("--batch-size", type=int, default=FORMAL_TRAINING["batch_size"])
    parser.add_argument(
        "--microbatch-size", type=int, default=FORMAL_TRAINING["microbatch_size"]
    )
    parser.add_argument(
        "--activation-precision",
        choices=("float32", "bfloat16"),
        default=FORMAL_TRAINING["activation_precision"],
    )
    parser.add_argument("--hidden-dim", type=int, default=FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE["hidden_dim"])
    parser.add_argument("--spatial-layers", type=int, default=FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE["spatial_layers"])
    parser.add_argument("--spatial-attention-heads", type=int, default=FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE["spatial_attention_heads"])
    parser.add_argument("--physics-slices", type=int, default=FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE["physics_slices"])
    parser.add_argument("--temporal-layers", type=int, default=FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE["temporal_layers"])
    parser.add_argument("--temporal-heads", type=int, default=FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE["temporal_heads"])
    parser.add_argument("--learning-rate", type=float, default=FORMAL_TRAINING["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=FORMAL_TRAINING["weight_decay"])
    parser.add_argument("--ema-decay", type=float, default=FORMAL_TRAINING["ema_decay"])
    parser.add_argument(
        "--spatial-time-chunk-size",
        type=int,
        default=FORMAL_TRAINING["spatial_time_chunk_size"],
    )
    parser.add_argument(
        "--temporal-node-chunk-size",
        type=int,
        default=FORMAL_TRAINING["temporal_node_chunk_size"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--ensemble-samples", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    args = parser.parse_args()
    validate_observation_input(
        args.run_role,
        args.observation_mask,
        args.observation_source,
    )
    if args.run_role != "software_smoke" and args.residual_geometry is None:
        raise ValueError(
            "formal diffusion comparison requires the regional residual geometry"
        )
    if args.ensemble_samples < 2:
        raise ValueError("diffusion comparison requires at least two stochastic samples")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive")
    if args.microbatch_size <= 0 or args.microbatch_size > args.batch_size:
        raise ValueError("microbatch size must be positive and no larger than effective batch size")
    if args.activation_precision == "bfloat16" and args.device != "cuda":
        raise ValueError("formal bfloat16 diffusion training requires CUDA")
    if args.run_role != "software_smoke":
        expected = {
            "epochs": FORMAL_TRAINING["epochs"],
            "batch_size": FORMAL_TRAINING["batch_size"],
            "microbatch_size": FORMAL_TRAINING["microbatch_size"],
            "activation_precision": FORMAL_TRAINING["activation_precision"],
            "hidden_dim": FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE["hidden_dim"],
            "spatial_layers": FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE["spatial_layers"],
            "spatial_attention_heads": FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE["spatial_attention_heads"],
            "physics_slices": FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE["physics_slices"],
            "temporal_layers": FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE["temporal_layers"],
            "temporal_heads": FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE["temporal_heads"],
            "learning_rate": FORMAL_TRAINING["learning_rate"],
            "weight_decay": FORMAL_TRAINING["weight_decay"],
            "ema_decay": FORMAL_TRAINING["ema_decay"],
            "spatial_time_chunk_size": FORMAL_TRAINING["spatial_time_chunk_size"],
            "temporal_node_chunk_size": FORMAL_TRAINING["temporal_node_chunk_size"],
        }
        changed = {name: (getattr(args, name), value) for name, value in expected.items() if getattr(args, name) != value}
        if changed:
            raise ValueError(f"diffusion settings differ from recorded algorithm sources: {changed}")

    device = torch.device(args.device)
    if args.threads <= 0:
        raise ValueError("PyTorch thread count must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    prediction_dir = args.prediction_dir.resolve()
    deterministic_summary_path = prediction_dir / "summary.json"
    if deterministic_summary_path.is_file():
        deterministic_summary = json.loads(
            deterministic_summary_path.read_text(encoding="utf-8")
        )
    elif args.run_role == "software_smoke":
        deterministic_summary = {}
    else:
        raise FileNotFoundError(
            "formal diffusion correction requires the deterministic model summary"
        )
    splits = {
        role: load_predictions(prediction_dir / f"{role}_temporal_temperature_predictions.npz")
        for role in ("train", "validation", "test")
    }
    validate_deterministic_prediction_contract(
        summary=deterministic_summary,
        splits=splits,
        prediction_dir=prediction_dir,
        run_role=args.run_role,
    )
    node_type = splits["train"]["node_type"].astype(np.int64)
    node_volume = splits["train"]["node_volume_m3"].astype(np.float32)
    structure = splits["train"]["structural_features"].astype(np.float32)
    temperature_mean = splits["train"]["temperature_mean_K_by_node_type"].astype(
        np.float32
    )
    temperature_std = splits["train"]["temperature_std_K_by_node_type"].astype(np.float32)
    for role, data in splits.items():
        if not np.array_equal(data["node_type"], node_type):
            raise ValueError(f"node types differ in {role}")
        if not np.allclose(data["node_volume_m3"], node_volume):
            raise ValueError(f"node volumes differ in {role}")
        if not np.allclose(data["structural_features"], structure):
            raise ValueError(f"structural features differ in {role}")
        if data["fixed_hydrodynamics_physical"].shape != (
            len(data["sequence_id"]),
            len(node_type),
            4,
        ):
            raise ValueError(f"fixed hydrodynamic field shape differs in {role}")
    residual_geometry = None
    if args.residual_geometry is not None:
        dataset_index = deterministic_summary.get("dataset_index")
        if not dataset_index:
            if args.run_role != "software_smoke":
                raise ValueError(
                    "formal diffusion energy evaluation requires the upstream dataset index"
                )
        else:
            index_payload = json.loads(
                Path(dataset_index).resolve().read_text(encoding="utf-8")
            )
            residual_geometry = load_p418_subface_geometry(
                args.residual_geometry.resolve(),
                fluid_patch_names=index_payload["boundary_patch_names"]["fluid"],
                solid_patch_names=index_payload["boundary_patch_names"]["solid"],
                device=device,
                dtype=torch.float32,
            )
    masks = observation_masks(
        args.observation_mask,
        splits,
        args.observation_source,
    )
    train_residual = (
        splits["train"]["target_temperature_normalized"]
        - splits["train"]["baseline_temperature_normalized"]
    ).astype(np.float32)
    scale = residual_scale(train_residual, node_volume)
    arrays = {}
    maximum_time = training_time_scale(splits)
    for role, data in splits.items():
        # Complete curves remain in CPU memory. One full curve already uses
        # most of the activation memory on the actual 46089-node graph.
        baseline = torch.as_tensor(data["baseline_temperature_normalized"])
        target = torch.as_tensor(data["target_temperature_normalized"])
        residual = (target - baseline) / scale
        mask = torch.as_tensor(masks[role], dtype=torch.bool)
        arrays[role] = {
            "baseline": baseline,
            "target": target,
            "residual": residual,
            "condition": torch.as_tensor(data["condition_normalized"]),
            "time": torch.as_tensor(
                (data["time_s"] / maximum_time).astype(np.float32)
            ),
            "mask": mask,
            "observed_residual": torch.where(mask, residual, torch.zeros_like(residual)),
        }
    structure_tensor = torch.as_tensor(structure, device=device)
    volume_tensor = torch.as_tensor(node_volume, device=device)
    model = P418TemporalTemperatureResidualRefiner(
        structural_dim=structure.shape[1],
        hidden_dim=args.hidden_dim,
        spatial_layers=args.spatial_layers,
        spatial_attention_heads=args.spatial_attention_heads,
        physics_slices=args.physics_slices,
        temporal_layers=args.temporal_layers,
        temporal_heads=args.temporal_heads,
        spatial_time_chunk_size=args.spatial_time_chunk_size,
        temporal_node_chunk_size=args.temporal_node_chunk_size,
    ).to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * math.ceil(len(train_residual) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate, total_steps=total_steps
    )
    order_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    diffusion_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "training_checkpoint.pt"
    checkpoint_contract = diffusion_checkpoint_contract(args, prediction_dir, splits)
    history: list[dict[str, object]] = []
    validation_history: list[dict[str, float]] = []
    validation_every = max(args.epochs // 10, 1)
    best_validation = math.inf
    best_epoch = None
    best_ema_state = None
    start_epoch = 0
    previous_training_seconds = 0.0
    if args.resume and checkpoint_path.is_file():
        resumed = load_diffusion_training_checkpoint(
            checkpoint_path,
            contract=checkpoint_contract,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            order_generator=order_generator,
            diffusion_generator=diffusion_generator,
            device=device,
        )
        start_epoch = int(resumed["next_epoch"])
        best_validation = float(resumed["best_validation"])
        best_epoch = (
            int(resumed["best_epoch"])
            if resumed["best_epoch"] is not None
            else None
        )
        best_ema_state = resumed["best_ema_state"]
        history = list(resumed["history"])
        validation_history = list(resumed["validation_history"])
        previous_training_seconds = float(resumed["training_seconds"])
        if start_epoch < 0 or start_epoch > args.epochs or len(history) != start_epoch:
            raise ValueError("diffusion checkpoint has an invalid completed-epoch count")

    def validation_velocity_loss() -> float:
        """Evaluate one fixed noising realization per validation curve."""
        ema.eval()
        losses = []
        with torch.no_grad():
            for index in range(len(arrays["validation"]["residual"])):
                baseline = arrays["validation"]["baseline"][index : index + 1].to(device)
                residual = arrays["validation"]["residual"][index : index + 1].to(device)
                condition = arrays["validation"]["condition"][index : index + 1].to(device)
                time_value = arrays["validation"]["time"][index : index + 1].to(device)
                observed_residual = arrays["validation"]["observed_residual"][
                    index : index + 1
                ].to(device)
                mask = arrays["validation"]["mask"][index : index + 1].to(device)
                step = torch.tensor(
                    [(args.seed + index) % 4], dtype=torch.long, device=device
                )
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + 2000 + index
                )
                noise = torch.randn(
                    residual.shape, generator=generator, device=device
                )
                noised, target_velocity = make_velocity_training_pair(
                    residual, step, noise=noise
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=args.activation_precision == "bfloat16",
                ):
                    prediction = ema(
                        baseline,
                        noised,
                        condition,
                        structure_tensor,
                        time_value,
                        observed_residual,
                        mask,
                        step,
                    )
                    loss = weighted_velocity_loss(
                        prediction, target_velocity, volume_tensor
                    )
                losses.append(float(loss.cpu()))
        return float(np.mean(losses))

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        order = torch.randperm(
            len(train_residual), generator=order_generator, device="cpu"
        )
        epoch_loss = 0.0
        seen = 0
        for start in range(0, len(order), args.batch_size):
            effective_index = order[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            for micro_start in range(0, len(effective_index), args.microbatch_size):
                index = effective_index[
                    micro_start : micro_start + args.microbatch_size
                ]
                baseline = arrays["train"]["baseline"][index].to(device)
                residual = arrays["train"]["residual"][index].to(device)
                condition = arrays["train"]["condition"][index].to(device)
                time_value = arrays["train"]["time"][index].to(device)
                observed_residual = arrays["train"]["observed_residual"][index].to(
                    device
                )
                mask = arrays["train"]["mask"][index].to(device)
                step = torch.randint(
                    0,
                    4,
                    (len(index),),
                    generator=diffusion_generator,
                    device=device,
                )
                noise = torch.randn(
                    residual.shape, generator=diffusion_generator, device=device
                )
                noised, target_velocity = make_velocity_training_pair(
                    residual, step, noise=noise
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=args.activation_precision == "bfloat16",
                ):
                    prediction = model(
                        baseline,
                        noised,
                        condition,
                        structure_tensor,
                        time_value,
                        observed_residual,
                        mask,
                        step,
                    )
                    loss = weighted_velocity_loss(
                        prediction, target_velocity, volume_tensor
                    )
                    accumulated_loss = loss * (
                        len(index) / len(effective_index)
                    )
                accumulated_loss.backward()
                epoch_loss += float(loss.detach().cpu()) * len(index)
                seen += len(index)
            optimizer.step()
            scheduler.step()
            update_ema(ema, model, args.ema_decay)
        validation_loss = None
        if epoch == 0 or (epoch + 1) % validation_every == 0 or epoch + 1 == args.epochs:
            validation_loss = validation_velocity_loss()
            validation_history.append(
                {"epoch": epoch + 1, "validation_velocity_loss": validation_loss}
            )
            if validation_loss < best_validation:
                best_validation = validation_loss
                best_epoch = epoch + 1
                best_ema_state = {
                    name: value.detach().cpu().clone()
                    for name, value in ema.state_dict().items()
                }
        history.append(
            {
                "epoch": epoch + 1,
                "training_velocity_loss": epoch_loss / seen,
                "validation_velocity_loss": validation_loss,
            }
        )
        if (epoch + 1) % args.checkpoint_every == 0 or epoch + 1 == args.epochs:
            save_diffusion_training_checkpoint(
                checkpoint_path,
                contract=checkpoint_contract,
                next_epoch=epoch + 1,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                order_generator=order_generator,
                diffusion_generator=diffusion_generator,
                best_validation=best_validation,
                best_epoch=best_epoch,
                best_ema_state=best_ema_state,
                history=history,
                validation_history=validation_history,
                training_seconds=(
                    previous_training_seconds + time.perf_counter() - started
                ),
            )
    training_seconds = previous_training_seconds + time.perf_counter() - started
    if best_ema_state is None or best_epoch is None:
        raise RuntimeError("diffusion training did not produce a validation-selected EMA model")
    ema.load_state_dict(best_ema_state)

    metrics = {}
    prediction_files = {}
    ema.eval()
    inference_progress_path = args.output_dir / "inference_progress.json"
    inference_started = time.perf_counter()
    total_case_count = sum(len(data["sequence_id"]) for data in splits.values())
    completed_case_count = 0
    write_inference_progress(
        inference_progress_path,
        {
            "status": "p418_temporal_diffusion_inference_running",
            "completed_case_count": completed_case_count,
            "total_case_count": total_case_count,
            "ensemble_samples_per_case": args.ensemble_samples,
            "completed_ensemble_members_in_current_case": 0,
            "elapsed_seconds": 0.0,
        },
    )
    for role, data in splits.items():
        refined_mean = []
        refined_std = []
        refined_lower = []
        refined_upper = []
        member_square_error_k = np.zeros(args.ensemble_samples, dtype=np.float64)
        member_solid_square_error_k = np.zeros(
            args.ensemble_samples, dtype=np.float64
        )
        convergence_sizes = [value for value in (8, 16, 32) if value <= args.ensemble_samples]
        if args.ensemble_samples not in convergence_sizes:
            convergence_sizes.append(args.ensemble_samples)
        convergence_square_error_k = {value: 0.0 for value in convergence_sizes}
        interval_covered_volume = 0.0
        interval_total_volume = 0.0
        interval_width_volume_k = 0.0
        predictive_std_volume_k = 0.0
        unobserved_dynamic_interval_sums = {
            "all": np.zeros(4, dtype=np.float64),
            "fluid": np.zeros(4, dtype=np.float64),
            "solid": np.zeros(4, dtype=np.float64),
        }
        unobserved_dynamic_crps_sums = {
            "all": np.zeros(2, dtype=np.float64),
            "fluid": np.zeros(2, dtype=np.float64),
            "solid": np.zeros(2, dtype=np.float64),
        }
        deterministic_energy_mse = 0.0
        refined_energy_mse = 0.0
        reference_energy_mse = 0.0
        deterministic_projection_energy_mse = 0.0
        refined_projection_energy_mse = 0.0
        member_projection_energy_mse = np.zeros(
            args.ensemble_samples, dtype=np.float64
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        evaluation_start = time.perf_counter()
        model_inference_seconds = 0.0
        for index in range(len(data["sequence_id"])):
            baseline_case = arrays[role]["baseline"][index : index + 1].to(device)
            condition_case = arrays[role]["condition"][index : index + 1].to(device)
            time_case = arrays[role]["time"][index : index + 1].to(device)
            observed_case = arrays[role]["observed_residual"][index : index + 1].to(
                device
            )
            mask_case = arrays[role]["mask"][index : index + 1].to(device)
            members = []
            if device.type == "cuda":
                torch.cuda.synchronize()
            model_inference_start = time.perf_counter()
            for sample_index in range(args.ensemble_samples):
                sample_seed = (
                    args.seed + 1000 + index * args.ensemble_samples + sample_index
                )
                initial_noise = torch.randn(
                    baseline_case.shape,
                    generator=torch.Generator(device=device).manual_seed(sample_seed),
                    device=device,
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=args.activation_precision == "bfloat16",
                ):
                    sampled = sample_temporal_temperature_residual(
                        ema,
                        baseline_case,
                        condition_case,
                        structure_tensor,
                        time_case,
                        observed_case,
                        mask_case,
                        initial_noise=initial_noise,
                    )
                members.append(
                    (baseline_case + scale * sampled)
                    .cpu()
                    .numpy()[0]
                )
                write_inference_progress(
                    inference_progress_path,
                    {
                        "status": "p418_temporal_diffusion_inference_running",
                        "role": role,
                        "sequence_id": str(data["sequence_id"][index]),
                        "case_index_within_role": index,
                        "case_count_within_role": len(data["sequence_id"]),
                        "completed_case_count": completed_case_count,
                        "total_case_count": total_case_count,
                        "ensemble_samples_per_case": args.ensemble_samples,
                        "completed_ensemble_members_in_current_case": sample_index + 1,
                        "elapsed_seconds": time.perf_counter() - inference_started,
                    },
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            model_inference_seconds += time.perf_counter() - model_inference_start
            member_array = np.stack(members, axis=0)
            target_case = data["target_temperature_normalized"][index]
            temperature_scale = temperature_std[node_type]
            error_k = (
                member_array[..., 0] - target_case[None, ..., 0]
            ) * temperature_scale[None, None, :]
            member_square_error_k += np.sum(
                np.square(error_k) * node_volume[None, None, :], axis=(1, 2)
            )
            solid_selection = node_type == 1
            member_solid_square_error_k += np.sum(
                np.square(error_k[..., solid_selection])
                * node_volume[solid_selection][None, None, :],
                axis=(1, 2),
            )
            for count in convergence_sizes:
                prefix_mean = member_array[:count].mean(axis=0)
                prefix_error_k = (
                    prefix_mean[..., 0] - target_case[..., 0]
                ) * temperature_scale[None, :]
                convergence_square_error_k[count] += float(
                    np.sum(np.square(prefix_error_k) * node_volume[None, :])
                )
            mean_case = member_array.mean(axis=0)
            std_case = member_array.std(axis=0)
            lower_case = np.quantile(member_array, 0.05, axis=0)
            upper_case = np.quantile(member_array, 0.95, axis=0)
            covered = (target_case >= lower_case) & (target_case <= upper_case)
            width_k = (upper_case[..., 0] - lower_case[..., 0]) * temperature_scale[None, :]
            std_k = std_case[..., 0] * temperature_scale[None, :]
            crps_k = ensemble_crps_k(
                member_array,
                target_case,
                temperature_scale,
            )
            interval_covered_volume += float(
                np.sum(covered[..., 0] * node_volume[None, :])
            )
            interval_total_volume += float(target_case.shape[0] * node_volume.sum())
            interval_width_volume_k += float(np.sum(width_k * node_volume[None, :]))
            predictive_std_volume_k += float(np.sum(std_k * node_volume[None, :]))
            for material_name, material_selection in (
                ("all", None),
                ("fluid", 0),
                ("solid", 1),
            ):
                selected = unobserved_dynamic_selection(
                    masks[role][index], node_type, material=material_selection
                )
                unobserved_dynamic_interval_sums[material_name] += interval_weighted_sums(
                    covered=covered[..., 0],
                    width_k=width_k,
                    predictive_std_k=std_k,
                    node_volume=node_volume,
                    selection=selected,
                )
                unobserved_dynamic_crps_sums[material_name] += weighted_selected_sums(
                    crps_k,
                    node_volume,
                    selected,
                )
            refined_mean.append(mean_case)
            refined_std.append(std_case)
            refined_lower.append(lower_case)
            refined_upper.append(upper_case)
            if residual_geometry is not None:
                energy_states = (
                    ("deterministic", data["baseline_temperature_normalized"][index]),
                    ("refined", mean_case),
                    ("reference", target_case),
                )
                case_energy = {}
                case_residual = {}
                for energy_name, temperature_history in energy_states:
                    state_np = physical_temperature_state(
                        temperature_history,
                        data["fixed_hydrodynamics_physical"][index],
                        node_type,
                        temperature_mean,
                        temperature_std,
                    )
                    state_tensor = torch.as_tensor(
                        state_np[None], dtype=torch.float32, device=device
                    )
                    condition_tensor = torch.as_tensor(
                        data["condition_physical"][index : index + 1],
                        dtype=torch.float32,
                        device=device,
                    )
                    time_tensor = torch.as_tensor(
                        data["time_s"][index : index + 1],
                        dtype=torch.float32,
                        device=device,
                    )
                    with torch.no_grad():
                        residual = assemble_p418_transient_regional_residual(
                            geometry=residual_geometry,
                            step_condition=condition_tensor,
                            state_physical=state_tensor,
                            time_s=time_tensor,
                            fluid_internal_mass_flux_kg_s=torch.as_tensor(
                                data["fluid_internal_mass_flux_kg_s"][
                                    index : index + 1
                                ],
                                dtype=torch.float32,
                                device=device,
                            ),
                            fluid_boundary_mass_flux_kg_s=torch.as_tensor(
                                data["fluid_boundary_mass_flux_kg_s"][
                                    index : index + 1
                                ],
                                dtype=torch.float32,
                                device=device,
                            ),
                        )
                        case_residual[energy_name] = residual
                        case_energy[energy_name] = float(
                            dimensionless_transient_energy_loss(
                                residual,
                                condition_tensor,
                                residual_geometry.fluid_mesh.cell_volume,
                                residual_geometry.solid_mesh.cell_volume,
                            ).cpu()
                        )
                deterministic_energy_mse += case_energy["deterministic"]
                refined_energy_mse += case_energy["refined"]
                reference_energy_mse += case_energy["reference"]
                deterministic_projection_energy_mse += float(
                    projection_aware_energy_difference_mse(
                        case_residual["deterministic"],
                        case_residual["reference"],
                        condition_tensor,
                        residual_geometry.fluid_mesh.cell_volume,
                        residual_geometry.solid_mesh.cell_volume,
                    ).cpu()
                )
                refined_projection_energy_mse += float(
                    projection_aware_energy_difference_mse(
                        case_residual["refined"],
                        case_residual["reference"],
                        condition_tensor,
                        residual_geometry.fluid_mesh.cell_volume,
                        residual_geometry.solid_mesh.cell_volume,
                    ).cpu()
                )
                internal_mass_flux_tensor = torch.as_tensor(
                    data["fluid_internal_mass_flux_kg_s"][index : index + 1],
                    dtype=torch.float32,
                    device=device,
                )
                boundary_mass_flux_tensor = torch.as_tensor(
                    data["fluid_boundary_mass_flux_kg_s"][index : index + 1],
                    dtype=torch.float32,
                    device=device,
                )
                if role == "test":
                    for sample_index, member_temperature in enumerate(member_array):
                        member_state = physical_temperature_state(
                            member_temperature,
                            data["fixed_hydrodynamics_physical"][index],
                            node_type,
                            temperature_mean,
                            temperature_std,
                        )
                        with torch.no_grad():
                            member_residual = assemble_p418_transient_regional_residual(
                                geometry=residual_geometry,
                                step_condition=condition_tensor,
                                state_physical=torch.as_tensor(
                                    member_state[None],
                                    dtype=torch.float32,
                                    device=device,
                                ),
                                time_s=time_tensor,
                                fluid_internal_mass_flux_kg_s=internal_mass_flux_tensor,
                                fluid_boundary_mass_flux_kg_s=boundary_mass_flux_tensor,
                            )
                            member_projection_energy_mse[sample_index] += float(
                                projection_aware_energy_difference_mse(
                                    member_residual,
                                    case_residual["reference"],
                                    condition_tensor,
                                    residual_geometry.fluid_mesh.cell_volume,
                                    residual_geometry.solid_mesh.cell_volume,
                                ).cpu()
                            )
            completed_case_count += 1
            write_inference_progress(
                inference_progress_path,
                {
                    "status": "p418_temporal_diffusion_inference_running",
                    "role": role,
                    "sequence_id": str(data["sequence_id"][index]),
                    "completed_case_count": completed_case_count,
                    "total_case_count": total_case_count,
                    "ensemble_samples_per_case": args.ensemble_samples,
                    "completed_ensemble_members_in_current_case": args.ensemble_samples,
                    "elapsed_seconds": time.perf_counter() - inference_started,
                },
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        evaluation_seconds = time.perf_counter() - evaluation_start
        refined_np = np.stack(refined_mean, axis=0)
        refined_std_np = np.stack(refined_std, axis=0)
        refined_lower_np = np.stack(refined_lower, axis=0)
        refined_upper_np = np.stack(refined_upper, axis=0)
        baseline_np = data["baseline_temperature_normalized"]
        target_np = data["target_temperature_normalized"]
        member_denominator = (
            len(data["sequence_id"]) * target_np.shape[1] * node_volume.sum()
        )
        member_rmse_k = np.sqrt(member_square_error_k / member_denominator)
        member_solid_denominator = (
            len(data["sequence_id"])
            * target_np.shape[1]
            * node_volume[node_type == 1].sum()
        )
        member_solid_rmse_k = np.sqrt(
            member_solid_square_error_k / member_solid_denominator
        )
        convergence = {
            str(count): math.sqrt(convergence_square_error_k[count] / member_denominator)
            for count in convergence_sizes
        }
        metrics[role] = {
            "deterministic_temperature_RMSE_K": temperature_rmse_k(
                baseline_np, target_np, node_type, node_volume, temperature_std
            ),
            "diffusion_refined_temperature_RMSE_K": temperature_rmse_k(
                refined_np, target_np, node_type, node_volume, temperature_std
            ),
            "deterministic_fluid_temperature_RMSE_K": temperature_rmse_k(
                baseline_np, target_np, node_type, node_volume, temperature_std, material=0
            ),
            "deterministic_solid_temperature_RMSE_K": temperature_rmse_k(
                baseline_np, target_np, node_type, node_volume, temperature_std, material=1
            ),
            "diffusion_refined_fluid_temperature_RMSE_K": temperature_rmse_k(
                refined_np, target_np, node_type, node_volume, temperature_std, material=0
            ),
            "diffusion_refined_solid_temperature_RMSE_K": temperature_rmse_k(
                refined_np, target_np, node_type, node_volume, temperature_std, material=1
            ),
            "diffusion_member_RMSE_K_median": float(np.median(member_rmse_k)),
            "diffusion_member_RMSE_K_p95": float(np.quantile(member_rmse_k, 0.95)),
            "diffusion_member_RMSE_K_minimum": float(member_rmse_k.min()),
            "diffusion_member_RMSE_K_maximum": float(member_rmse_k.max()),
            "diffusion_member_solid_RMSE_K_median": float(
                np.median(member_solid_rmse_k)
            ),
            "diffusion_member_solid_RMSE_K_p95": float(
                np.quantile(member_solid_rmse_k, 0.95)
            ),
            "diffusion_predictive_std_K_volume_mean": (
                predictive_std_volume_k / interval_total_volume
            ),
            "diffusion_90pct_interval_coverage_fraction": (
                interval_covered_volume / interval_total_volume
            ),
            "diffusion_90pct_interval_mean_width_K": (
                interval_width_volume_k / interval_total_volume
            ),
            "ensemble_mean_RMSE_K_by_sample_count": convergence,
            "observation_count": int(masks[role].sum()),
            "inference_seconds": model_inference_seconds,
            "inference_seconds_per_curve": model_inference_seconds
            / len(data["sequence_id"]),
            "evaluation_seconds_including_metrics": evaluation_seconds,
        }
        node_scale = temperature_std[node_type][None, None, :]
        node_offset = temperature_mean[node_type][None, None, :]
        target_temperature_k = target_np[..., 0] * node_scale + node_offset
        deterministic_temperature_k = baseline_np[..., 0] * node_scale + node_offset
        refined_temperature_k = refined_np[..., 0] * node_scale + node_offset
        deterministic_hotspot = solid_transient_hotspot_metrics(
            deterministic_temperature_k,
            target_temperature_k,
            node_type,
            data["node_centroid_m"],
        )
        refined_hotspot = solid_transient_hotspot_metrics(
            refined_temperature_k,
            target_temperature_k,
            node_type,
            data["node_centroid_m"],
        )
        metrics[role].update(
            {f"deterministic_{key}": value for key, value in deterministic_hotspot.items()}
        )
        metrics[role].update(
            {f"diffusion_refined_{key}": value for key, value in refined_hotspot.items()}
        )
        metrics[role].update(
            finalized_interval_metrics(
                unobserved_dynamic_interval_sums["all"],
                prefix="diffusion_unobserved_dynamic",
            )
        )
        metrics[role]["diffusion_unobserved_dynamic_CRPS_K"] = finalized_selected_mean(
            unobserved_dynamic_crps_sums["all"]
        )
        metrics[role][
            "diffusion_unobserved_dynamic_fluid_CRPS_K"
        ] = finalized_selected_mean(unobserved_dynamic_crps_sums["fluid"])
        metrics[role][
            "diffusion_unobserved_dynamic_solid_CRPS_K"
        ] = finalized_selected_mean(unobserved_dynamic_crps_sums["solid"])
        metrics[role].update(
            finalized_interval_metrics(
                unobserved_dynamic_interval_sums["fluid"],
                prefix="diffusion_unobserved_dynamic_fluid",
            )
        )
        metrics[role].update(
            finalized_interval_metrics(
                unobserved_dynamic_interval_sums["solid"],
                prefix="diffusion_unobserved_dynamic_solid",
            )
        )
        if residual_geometry is not None:
            count = len(data["sequence_id"])
            metrics[role].update(
                energy_residual_summary(
                    deterministic_energy_mse / count,
                    refined_energy_mse / count,
                    reference_energy_mse / count,
                )
            )
            deterministic_projection_energy_rmse = math.sqrt(
                deterministic_projection_energy_mse / count
            )
            refined_projection_energy_rmse = math.sqrt(
                refined_projection_energy_mse / count
            )
            metrics[role].update(
                {
                    "deterministic_projection_aware_energy_equation_normalized_RMSE": deterministic_projection_energy_rmse,
                    "diffusion_refined_projection_aware_energy_equation_normalized_RMSE": refined_projection_energy_rmse,
                }
            )
            if role == "test":
                member_projection_energy_rmse = np.sqrt(
                    member_projection_energy_mse / count
                )
                metrics[role].update(
                    {
                    "diffusion_member_projection_aware_energy_equation_normalized_RMSE_median": float(
                        np.median(member_projection_energy_rmse)
                    ),
                    "diffusion_member_projection_aware_energy_equation_normalized_RMSE_p95": float(
                        np.quantile(member_projection_energy_rmse, 0.95)
                    ),
                    "diffusion_member_projection_aware_energy_equation_normalized_RMSE_minimum": float(
                        member_projection_energy_rmse.min()
                    ),
                    "diffusion_member_projection_aware_energy_equation_normalized_RMSE_maximum": float(
                        member_projection_energy_rmse.max()
                    ),
                    "diffusion_member_joint_temperature_energy_improvement_fraction": float(
                        np.mean(
                            (
                                member_solid_rmse_k
                                < metrics[role][
                                    "deterministic_solid_temperature_RMSE_K"
                                ]
                            )
                            & (
                                member_projection_energy_rmse
                                <= deterministic_projection_energy_rmse
                            )
                        )
                    ),
                    "diffusion_member_sample_count": int(args.ensemble_samples),
                    }
                )
        output_path = args.output_dir / f"{role}_refined_temperature.npz"
        np.savez_compressed(
            output_path,
            sequence_id=data["sequence_id"],
            time_s=data["time_s"],
            condition_physical=data["condition_physical"],
            fixed_hydrodynamics_physical=data["fixed_hydrodynamics_physical"],
            fluid_internal_mass_flux_kg_s=data["fluid_internal_mass_flux_kg_s"],
            fluid_boundary_mass_flux_kg_s=data["fluid_boundary_mass_flux_kg_s"],
            refined_temperature_normalized=refined_np,
            refined_temperature_std_normalized=refined_std_np,
            refined_temperature_q05_normalized=refined_lower_np,
            refined_temperature_q95_normalized=refined_upper_np,
            target_temperature_normalized=target_np,
            observation_mask=masks[role],
            node_type=node_type,
            node_volume_m3=node_volume,
            node_centroid_m=data["node_centroid_m"],
            temperature_mean_K_by_node_type=temperature_mean,
            temperature_std_K_by_node_type=temperature_std,
        )
        prediction_files[role] = output_path.name
    torch.save(ema.state_dict(), args.output_dir / "ema_model_state.pt")
    summary = {
        "status": "completed_p418_temporal_temperature_diffusion",
        "run_role": args.run_role,
        "split_name": deterministic_summary.get("split_name"),
        "seed": args.seed,
        "upstream_training_seed": deterministic_summary.get("seed"),
        "residual_scale_in_normalized_temperature": scale,
        "time_normalization_maximum_s": maximum_time,
        "time_normalization_source": "training_curves_only",
        "ensemble_samples": args.ensemble_samples,
        "ensemble_seed_rule": "seed + 1000 + case_index * ensemble_samples + sample_index",
        "ensemble_member_energy_evaluation": (
            "each stochastic temperature prediction is compared with the regionalized "
            "OpenFOAM energy equation; the ensemble mean alone is not used to represent "
            "member-level physical consistency"
        ),
        "observation_input": {
            "mask_file": (
                str(args.observation_mask.resolve())
                if args.observation_mask is not None
                else None
            ),
            "role": (
                "computed_openfoam_sparse_points"
                if args.observation_mask is not None
                else "none_full_field_computed_residual_benchmark"
            ),
            "source_kind": args.observation_source,
            "observed_values_kind": (
                "target_temperature_normalized_from_openfoam_reference"
                if args.observation_mask is not None
                else None
            ),
            "hard_conditioning_is_exact": args.observation_mask is not None,
            "external_measurements_supported_by_this_trainer": False,
            "computed_openfoam_targets_are_measurements": False,
            "unobserved_metric_rule": (
                "exclude t=0 and every supplied observation; report fluid, solid and "
                "combined dynamic locations separately"
            ),
        },
        "split_case_counts": {role: len(data["sequence_id"]) for role, data in splits.items()},
        "split_case_ids": {
            role: [str(value) for value in data["sequence_id"]] for role, data in splits.items()
        },
        "deterministic_prediction_dir": str(prediction_dir),
        "energy_residual_geometry": (
            str(args.residual_geometry.resolve())
            if args.residual_geometry is not None
            else None
        ),
        "temperature_metric_definition": (
            "regional-volume-weighted RMSE, reported separately for fluid and solid"
        ),
        "metrics": metrics,
        "prediction_files": prediction_files,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "compute_device": str(device),
        "activation_precision": args.activation_precision,
        "effective_batch_size": args.batch_size,
        "microbatch_size": args.microbatch_size,
        "spatial_time_chunk_size": args.spatial_time_chunk_size,
        "temporal_node_chunk_size": args.temporal_node_chunk_size,
        "gradient_accumulation_rule": (
            "volume-weighted microbatch losses are weighted by curve count, "
            "then one optimizer and EMA update is applied per effective batch"
        ),
        "torch_num_threads": torch.get_num_threads(),
        "training_seconds": training_seconds,
        "training_resumed_from_epoch": start_epoch,
        "training_checkpoint": checkpoint_path.name,
        "checkpoint_every_epochs": args.checkpoint_every,
        "selection_split": "validation",
        "selection_metric": "fixed-noise volume-weighted diffusion velocity loss",
        "selection_evaluation_interval_epochs": validation_every,
        "selected_epoch": best_epoch,
        "best_validation_velocity_loss": best_validation,
        "validation_selection_history": validation_history,
        "training_history": history,
        "architecture": {
            "hidden_dim": args.hidden_dim,
            "spatial_layers": args.spatial_layers,
            "spatial_attention_heads": args.spatial_attention_heads,
            "physics_slices": args.physics_slices,
            "temporal_layers": args.temporal_layers,
            "temporal_heads": args.temporal_heads,
            "refinement_steps": 3,
            "spatial_time_chunk_size": args.spatial_time_chunk_size,
            "temporal_node_chunk_size": args.temporal_node_chunk_size,
        },
        "algorithm_sources": {
            "diffusion_schedule": "PDE-Refiner, NeurIPS 2023",
            "spatial_attention": "Transolver, ICML 2024",
            "partial_observation_role": "DiffusionPDE, NeurIPS 2024",
            "source_files": [
                "parameters/hccb_p418_ai_architecture_sources.json",
                "parameters/apd006_tdem_diffusion_route_contract.yaml",
                "parameters/hccb_p418_mgnt_temporal_pino_contract.yaml",
            ],
        },
        "physical_parameter_sources": (
            "inherited unchanged from each deterministic prediction file"
        ),
        "new_physical_parameters": [],
        "corrected_state_channels": ["temperature"],
        "fixed_state_channels": ["velocity_x", "velocity_y", "velocity_z", "pressure"],
        "initial_condition_rule": (
            "temperature residual is exactly zero at t=0; a conflicting supplied "
            "observation cannot alter the deterministic initial temperature"
        ),
        "scientific_scope": (
            "Temperature-residual correction after the deterministic regional graph--Transformer. "
            "The deterministic, diffusion-refined and OpenFOAM temperature fields are also "
            "evaluated with the same transient fluid/solid energy equations. A computed "
            "residual benchmark is not an experimental posterior claim."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_inference_progress(
        inference_progress_path,
        {
            "status": "completed_p418_temporal_diffusion_inference",
            "completed_case_count": total_case_count,
            "total_case_count": total_case_count,
            "ensemble_samples_per_case": args.ensemble_samples,
            "completed_ensemble_members_in_current_case": args.ensemble_samples,
            "elapsed_seconds": time.perf_counter() - inference_started,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
