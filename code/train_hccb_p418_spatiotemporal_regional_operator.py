#!/usr/bin/env python3
"""Train the regional graph--Transformer on complete P418 thermal steps."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from hccb_p418_comparison_contract import file_record, sha256_file
from hccb_p418_fixed_flow_loss_balancing import (
    balanced_fixed_flow_loss,
    build_fixed_flow_loss_balancer,
    load_fixed_flow_candidate,
)
from hccb_p418_regional_cht_adapter import load_p418_subface_geometry
from hccb_p418_spatiotemporal_regional_operator import (
    FORMAL_ARCHITECTURE,
    HCCBP418SpatiotemporalRegionalOperator,
    P418ThermalStepRegionalGraph,
    SPATIAL_TEMPORAL_MODES,
    TEMPERATURE_OUTPUT_MODES,
)
from hccb_p418_transient_hotspot_metrics import solid_transient_hotspot_metrics
from hccb_p418_transient_regional_physics import (
    assemble_p418_transient_regional_residual,
    dimensionless_transient_energy_loss,
    volume_weighted_mean_square,
)
from hccb_source_backed_thermophysical import load_hccb_thermophysical_parameters


FORMAL_TRAINING = {
    "epochs": 500,
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-5,
    "data_weight": 5.0,
    "edge_flux_weight": 1.0,
    "energy_weight": 1.0,
}

P418_TRANSIENT_PARAMETER_IDS = (
    "P070", "P071", "P092", "P388", "P389", "P403", "P418",
    "P424", "P425", "P426", "P427", "P428", "P429", "P430", "P431",
)


FLUID_TEMPERATURE_RANGE_K = (300.0, 1000.0)
SOLID_TEMPERATURE_RANGE_K = (
    load_hccb_thermophysical_parameters().solid_cp_temperature_range_k
)

TEMPERATURE_OUTPUT_BOUNDS_K = np.asarray(
    (
        FLUID_TEMPERATURE_RANGE_K,
        SOLID_TEMPERATURE_RANGE_K,
    ),
    dtype=np.float32,
)
TEMPERATURE_OUTPUT_BOUND_SOURCE_IDS = ("P424", "P428", "P429")

PHYSICS_REFERENCE_FIELDS = (
    "fluid_energy_w_m3",
    "solid_energy_w_m3",
    "fluid_internal_energy_flux_w",
    "solid_internal_heat_flux_w",
)

FIXED_FLOW_LOSS_BALANCING_SOURCES = (
    Path(__file__).resolve().parents[1]
    / "parameters/hccb_p418_fixed_flow_loss_balancing_candidates.json"
)


def sequence_records(index: dict[str, object]) -> dict[str, dict[str, object]]:
    records = {str(row["sequence_id"]): row for row in index["sequences"]}
    if len(records) != int(index["sequence_count"]):
        raise ValueError("sequence records are duplicated or incomplete")
    return records


def selected_split(
    sequence_ids: set[str], split_path: Path, split_name: str
) -> dict[str, list[str]]:
    split = json.loads(split_path.read_text(encoding="utf-8"))["splits"][split_name]
    result = {role: [str(value) for value in split[role]] for role in ("train", "validation", "test")}
    unused = [str(value) for value in split.get("unused", [])]
    groups = {**result, "unused": unused}
    covered = set().union(*map(set, groups.values()))
    if covered != sequence_ids:
        raise ValueError(
            f"split and regional sequences differ: missing={sorted(sequence_ids-covered)}, "
            f"extra={sorted(covered-sequence_ids)}"
        )
    role_names = tuple(groups)
    if any(
        set(groups[role_names[left]]) & set(groups[role_names[right]])
        for left in range(len(role_names))
        for right in range(left + 1, len(role_names))
    ):
        raise ValueError("complete thermal-step curves overlap across split roles")
    return result


def load_sequence(
    root: Path, record: dict[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(root / str(record["sequence_file"]), allow_pickle=False) as data:
        time_s = data["time_s"].astype(np.float32)
        condition = data["condition_physical"].astype(np.float32)
        state = data["state_physical"].astype(np.float32)
        internal_mass_flux = data["fluid_internal_mass_flux_kg_s"].astype(np.float32)
        boundary_mass_flux = data["fluid_boundary_mass_flux_kg_s"].astype(np.float32)
    if state.ndim != 3 or state.shape[-1] != 5 or len(time_s) != len(state):
        raise ValueError(f"invalid regional sequence {record['sequence_id']}")
    return time_s, condition, state, internal_mass_flux, boundary_mass_flux


def physics_time_chunks(
    time_count: int, chunk_size: int
) -> list[tuple[slice, slice, slice, float]]:
    """Return overlapped slices that preserve the three-point time derivative.

    Each tuple contains the extended input slice, the local core slice, the
    matching global core slice, and the core fraction of the complete history.
    One neighbour is retained on each internal side, so cropped derivatives are
    identical to evaluating the full reported history at once.
    """
    if time_count < 3:
        raise ValueError("chunked transient physics requires at least three times")
    if chunk_size < 2:
        raise ValueError("physics time chunks must contain at least two core times")
    chunks = []
    for core_start in range(0, time_count, chunk_size):
        core_end = min(time_count, core_start + chunk_size)
        extended_start = max(0, core_start - 1)
        extended_end = min(time_count, core_end + 1)
        local_start = core_start - extended_start
        local_end = local_start + core_end - core_start
        chunks.append(
            (
                slice(extended_start, extended_end),
                slice(local_start, local_end),
                slice(core_start, core_end),
                (core_end - core_start) / time_count,
            )
        )
    return chunks


def residual_loss_view(residual, time_slice: slice) -> SimpleNamespace:
    """Keep only the residual fields used by the formal physics objective."""
    return SimpleNamespace(
        **{
            name: getattr(residual, name)[:, time_slice]
            for name in PHYSICS_REFERENCE_FIELDS
        }
    )


def time_slice_mass_flux(values: torch.Tensor, time_slice: slice) -> torch.Tensor:
    """Slice a time-dependent face flux while retaining a fixed face flux."""
    if values.ndim == 2:
        return values
    if values.ndim == 3:
        return values[:, time_slice]
    raise ValueError("face flux must have [batch,face] or [batch,time,face] shape")


def training_statistics(
    root: Path,
    records: dict[str, dict[str, object]],
    training_ids: list[str],
    node_type: np.ndarray,
) -> dict[str, np.ndarray | float]:
    conditions = []
    state_sum = np.zeros((2, 5), dtype=np.float64)
    state_square = np.zeros((2, 5), dtype=np.float64)
    state_count = np.zeros((2, 5), dtype=np.int64)
    valid = {0: np.arange(5), 1: np.asarray([4])}
    maximum_time = 0.0
    for sequence_id in training_ids:
        time_s, condition, state, _, _ = load_sequence(root, records[sequence_id])
        conditions.append(condition)
        maximum_time = max(maximum_time, float(time_s.max()))
        for material in (0, 1):
            selected = state[:, node_type == material]
            for channel in valid[material]:
                values = selected[..., channel].astype(np.float64)
                state_sum[material, channel] += values.sum()
                state_square[material, channel] += np.square(values).sum()
                state_count[material, channel] += values.size
    condition_values = np.stack(conditions).astype(np.float64)
    condition_mean = condition_values.mean(axis=0)
    condition_std = condition_values.std(axis=0)
    condition_std[condition_std < 1.0e-12] = 1.0
    state_mean = np.zeros((2, 5), dtype=np.float64)
    state_std = np.ones((2, 5), dtype=np.float64)
    populated = state_count > 0
    state_mean[populated] = state_sum[populated] / state_count[populated]
    variance = np.zeros_like(state_mean)
    variance[populated] = (
        state_square[populated] / state_count[populated]
        - np.square(state_mean[populated])
    )
    state_std[populated] = np.sqrt(np.maximum(variance[populated], 0.0))
    state_std[state_std < 1.0e-12] = 1.0
    if maximum_time <= 0:
        raise ValueError("training curves contain no positive time")
    return {
        "condition_mean": condition_mean.astype(np.float32),
        "condition_std": condition_std.astype(np.float32),
        "state_mean": state_mean.astype(np.float32),
        "state_std": state_std.astype(np.float32),
        "maximum_time_s": maximum_time,
    }


def state_scale_tensors(
    statistics: dict[str, np.ndarray | float], node_type: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    mean_by_type = torch.as_tensor(statistics["state_mean"], device=device)
    std_by_type = torch.as_tensor(statistics["state_std"], device=device)
    return mean_by_type[node_type], std_by_type[node_type]


def temperature_data_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    graph: P418ThermalStepRegionalGraph,
) -> torch.Tensor:
    error = (prediction[..., 4] - target[..., 4]).square()
    losses = []
    for material in (0, 1):
        selected = graph.node_type == material
        weight = graph.volume_m3[selected]
        weight = weight / weight.sum()
        losses.append((error[:, :, selected] * weight[None, None, :]).sum(dim=-1).mean())
    return 0.5 * (losses[0] + losses[1])


def area_weighted_flux_density_mean_square(
    flux_difference_w: torch.Tensor,
    face_area_m2: torch.Tensor,
    power_scale_w: torch.Tensor,
) -> torch.Tensor:
    """Dimensionless face-flux error invariant to geometric face splitting."""
    if flux_difference_w.ndim < 3 or flux_difference_w.shape[-1] != len(face_area_m2):
        raise ValueError("flux difference and face-area shapes differ")
    if torch.any(~torch.isfinite(face_area_m2)) or torch.any(face_area_m2 <= 0.0):
        raise ValueError("internal face areas must be finite and positive")
    if power_scale_w.ndim != 1 or power_scale_w.shape[0] != flux_difference_w.shape[0]:
        raise ValueError("power scale must contain one value per condition")
    total_area = face_area_m2.sum()
    reference_flux_density = power_scale_w / total_area
    normalized_density_error = (
        flux_difference_w
        / face_area_m2
        / reference_flux_density[:, None, None]
    )
    area_weight = face_area_m2 / total_area
    return (
        normalized_density_error.square() * area_weight
    ).sum(dim=-1).mean()


def physics_difference_losses(
    predicted,
    reference: dict[str, torch.Tensor],
    condition_physical: torch.Tensor,
    fluid_volume_m3: torch.Tensor,
    solid_volume_m3: torch.Tensor,
    fluid_internal_face_area_m2: torch.Tensor,
    solid_internal_face_area_m2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_power = (
        condition_physical[:, 5] * 1.0e6 * solid_volume_m3.sum()
    ).clamp_min(torch.finfo(condition_physical.dtype).tiny)
    source_density = (condition_physical[:, 5] * 1.0e6).clamp_min(
        torch.finfo(condition_physical.dtype).tiny
    )
    fluid_flux_difference = (
        predicted.fluid_internal_energy_flux_w - reference["fluid_internal_energy_flux_w"]
    )
    solid_flux_difference = (
        predicted.solid_internal_heat_flux_w - reference["solid_internal_heat_flux_w"]
    )
    edge_flux_loss = 0.5 * (
        area_weighted_flux_density_mean_square(
            fluid_flux_difference, fluid_internal_face_area_m2, source_power
        )
        + area_weighted_flux_density_mean_square(
            solid_flux_difference, solid_internal_face_area_m2, source_power
        )
    )
    fluid_energy = (
        predicted.fluid_energy_w_m3 - reference["fluid_energy_w_m3"]
    ) / source_density[:, None, None]
    solid_energy = (
        predicted.solid_energy_w_m3 - reference["solid_energy_w_m3"]
    ) / source_density[:, None, None]
    energy_loss = 0.5 * (
        volume_weighted_mean_square(fluid_energy, fluid_volume_m3)
        + volume_weighted_mean_square(solid_energy, solid_volume_m3)
    )
    return edge_flux_loss, energy_loss


def physics_training_losses(
    predicted,
    reference: dict[str, torch.Tensor],
    condition_physical: torch.Tensor,
    fluid_volume_m3: torch.Tensor,
    solid_volume_m3: torch.Tensor,
    fluid_internal_face_area_m2: torch.Tensor,
    solid_internal_face_area_m2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return flux, projection-aware energy, and absolute-residual diagnostic.

    Regional volume averaging does not commute with the nonlinear face-flux
    operator, so even a projected OpenFOAM reference field has a non-zero local
    residual.  The trainable energy term therefore minimizes the difference
    between predicted and projected-reference equation residuals.  The absolute
    predicted residual remains visible as a diagnostic and is never presented
    as the native-mesh OpenFOAM closure.
    """
    edge_flux_loss, projection_aware_energy = physics_difference_losses(
        predicted,
        reference,
        condition_physical,
        fluid_volume_m3,
        solid_volume_m3,
        fluid_internal_face_area_m2,
        solid_internal_face_area_m2,
    )
    absolute_energy_balance = dimensionless_transient_energy_loss(
        predicted, condition_physical, fluid_volume_m3, solid_volume_m3
    )
    return edge_flux_loss, projection_aware_energy, absolute_energy_balance


def validation_selection_score(
    metrics: dict[str, float], physics_mode: str
) -> float:
    """Select data-only and PINN checkpoints with their declared objectives."""
    if physics_mode == "energy_and_flux":
        return float(metrics["weighted_physics_objective"])
    if physics_mode == "data_only":
        return float(metrics["normalized_temperature_data_MSE"])
    raise ValueError(f"unknown physics mode {physics_mode}")


def loss_balancing_validation_score(metrics: dict[str, float]) -> float:
    """Compare weighting candidates with one weight-independent validation score."""
    values = np.asarray(
        [
            metrics["normalized_temperature_data_MSE"],
            metrics["reference_edge_energy_flux_normalized_RMSE"] ** 2,
            metrics[
                "projection_aware_energy_equation_normalized_RMSE"
            ] ** 2,
        ],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("loss-balancing validation groups must be finite and nonnegative")
    return float(np.mean(values))


def energy_residual_rmse_ratio(
    prediction_mse: float, reference_mse: float
) -> tuple[float, float, float]:
    """Return prediction/reference RMSE and their finite ratio."""
    if prediction_mse < 0 or reference_mse < 0:
        raise ValueError("energy residual mean squares must be non-negative")
    prediction_rmse = math.sqrt(prediction_mse)
    reference_rmse = math.sqrt(reference_mse)
    denominator = max(reference_rmse, np.finfo(np.float64).eps)
    return prediction_rmse, reference_rmse, prediction_rmse / denominator


def formal_configuration_check(args: argparse.Namespace) -> None:
    expected = {
        "epochs": FORMAL_TRAINING["epochs"],
        "learning_rate": FORMAL_TRAINING["learning_rate"],
        "weight_decay": FORMAL_TRAINING["weight_decay"],
        "hidden_dim": FORMAL_ARCHITECTURE["hidden_dim"],
        "local_pre_iterations": FORMAL_ARCHITECTURE["local_pre_iterations"],
        "physics_attention_blocks": FORMAL_ARCHITECTURE["physics_attention_blocks"],
        "local_post_iterations": FORMAL_ARCHITECTURE["local_post_iterations"],
        "physics_attention_heads": FORMAL_ARCHITECTURE["physics_attention_heads"],
        "physics_slices": FORMAL_ARCHITECTURE["physics_slices"],
        "temporal_layers": FORMAL_ARCHITECTURE["temporal_layers"],
        "temporal_heads": FORMAL_ARCHITECTURE["temporal_heads"],
        "temperature_output_mode": "literature_bounded_residual",
    }
    changed = {name: (getattr(args, name), value) for name, value in expected.items() if getattr(args, name) != value}
    if changed:
        raise ValueError(f"formal architecture differs from the recorded literature contract: {changed}")
    expected_mode = "data_only" if args.run_role == "formal_data_only" else "energy_and_flux"
    if args.physics_mode != expected_mode:
        raise ValueError(
            f"{args.run_role} requires physics_mode={expected_mode}, got {args.physics_mode}"
        )
    expected_spatial_temporal_mode = (
        "factorized_static_spatial"
        if args.run_role == "formal_factorized"
        else "repeated_query_spatial"
    )
    if args.spatial_temporal_mode != expected_spatial_temporal_mode:
        raise ValueError(
            f"{args.run_role} requires spatial_temporal_mode="
            f"{expected_spatial_temporal_mode}, got {args.spatial_temporal_mode}"
        )
    if args.loss_balancing_candidate_id is None:
        if args.evaluation_stage != "final" or args.selected_method_record is not None:
            raise ValueError(
                "the unchanged fixed-weight run does not use loss-balancing selection"
            )
        return
    if (
        args.run_role not in {"formal", "formal_factorized"}
        or args.physics_mode != "energy_and_flux"
    ):
        raise ValueError(
            "fixed-flow loss balancing is used only for formal energy-and-flux models"
        )
    source_path = args.loss_balancing_sources.resolve()
    if source_path != FIXED_FLOW_LOSS_BALANCING_SOURCES.resolve():
        raise ValueError("formal loss balancing must use the recorded project source file")
    load_fixed_flow_candidate(source_path, args.loss_balancing_candidate_id)
    if args.evaluation_stage == "selection":
        if args.run_role != "formal":
            raise ValueError(
                "validation-only loss-weight selection uses the repeated-query model"
            )
        if args.selected_method_record is not None:
            raise ValueError("validation-only selection does not accept a selected-method record")
        return
    if args.selected_method_record is None:
        raise ValueError("final evaluation requires the validation-only selection record")
    record_path = args.selected_method_record.resolve()
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("status") != "p418_loss_balancing_selected_on_validation_only":
        raise ValueError("loss-balancing selection record has an unexpected status")
    if record.get("independent_test_read") is not False:
        raise ValueError("loss-balancing selection must not read independent test curves")
    if record.get("source_file_sha256") != sha256_file(source_path):
        raise ValueError("loss-balancing candidates changed after validation selection")
    if record.get("selected_candidate_id") != args.loss_balancing_candidate_id:
        raise ValueError("final run differs from the validation-selected candidate")
    selected_summary = Path(str(record["selected_summary_path"])).resolve()
    if sha256_file(selected_summary) != record.get("selected_summary_sha256"):
        raise ValueError("validation-only selection summary changed before final evaluation")
    selected = json.loads(selected_summary.read_text(encoding="utf-8"))
    if (
        selected.get("evaluation_stage") != "selection"
        or selected.get("test_evaluated") is not False
        or "test" in selected.get("metrics", {})
    ):
        raise ValueError("selected loss-balancing summary is not validation-only")
    expected_inputs = {
        "dataset_index": sha256_file(args.dataset_index.resolve()),
        "split_file": sha256_file(args.splits.resolve()),
        "residual_geometry": sha256_file(args.residual_geometry.resolve()),
        "loss_balancing_sources": sha256_file(source_path),
    }
    if selected.get("input_file_sha256") != expected_inputs:
        raise ValueError("data, split, geometry or candidates changed after selection")


def checkpoint_contract(
    args: argparse.Namespace,
    index_path: Path,
    split: dict[str, list[str]],
) -> dict[str, object]:
    """Return the settings that must remain identical after a restart."""
    contract = {
        "dataset_index": str(index_path),
        "split_name": args.split_name,
        "split_sequence_ids": split,
        "split_case_ids": split,
        "run_role": args.run_role,
        "physics_mode": args.physics_mode,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "local_pre_iterations": args.local_pre_iterations,
        "physics_attention_blocks": args.physics_attention_blocks,
        "local_post_iterations": args.local_post_iterations,
        "physics_attention_heads": args.physics_attention_heads,
        "physics_slices": args.physics_slices,
        "temporal_layers": args.temporal_layers,
        "temporal_heads": args.temporal_heads,
        "spatial_time_chunk_size": args.spatial_time_chunk_size,
        "temporal_node_chunk_size": args.temporal_node_chunk_size,
        "spatial_temporal_mode": args.spatial_temporal_mode,
        "temperature_output_mode": args.temperature_output_mode,
        "physics_computation_device": args.physics_device,
        "physics_time_chunk_size": args.physics_time_chunk_size,
        "physics_loss_revision": (
            "projection_aware_volume_energy_and_area_flux_v3"
            if args.physics_mode == "energy_and_flux"
            else "data_only_v1"
        ),
        "validation_selection_rule": (
            "weighted_temperature_flux_projection_aware_energy"
            if args.physics_mode == "energy_and_flux"
            else "normalized_temperature_mse"
        ),
        "seed": args.seed,
    }
    if args.loss_balancing_candidate_id is not None:
        contract["loss_balancing"] = {
            "candidate_id": args.loss_balancing_candidate_id,
            "source_file": str(args.loss_balancing_sources.resolve()),
            "source_file_sha256": sha256_file(
                args.loss_balancing_sources.resolve()
            ),
            "validation_selection_rule": (
                "equal_mean_of_temperature_edge_flux_and_projection_energy_MSE"
            ),
        }
    return contract


def save_training_checkpoint(
    path: Path,
    *,
    contract: dict[str, object],
    next_epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best_validation: float,
    best_state: dict[str, torch.Tensor] | None,
    history: list[dict[str, float]],
    training_seconds: float,
    loss_balancer=None,
) -> None:
    """Atomically save every state needed to continue the next epoch."""
    payload = {
        "contract": contract,
        "next_epoch": next_epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_validation": best_validation,
        "best_state": best_state,
        "history": history,
        "training_seconds": training_seconds,
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
        "cuda_random_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    if loss_balancer is not None:
        payload["loss_balancer_state"] = copy.deepcopy(
            loss_balancer.state_dict()
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_training_checkpoint(
    path: Path,
    *,
    contract: dict[str, object],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    loss_balancer=None,
) -> dict[str, object]:
    """Restore a completed-epoch checkpoint after verifying its data and settings."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("contract") != contract:
        raise ValueError("training checkpoint does not match the current data, split or model settings")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    if loss_balancer is not None:
        if "loss_balancer_state" not in payload:
            raise ValueError("training checkpoint lacks loss-balancer state")
        loss_balancer.load_state_dict(payload["loss_balancer_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.random.set_rng_state(payload["torch_random_state"].cpu())
    cuda_state = payload.get("cuda_random_state_all")
    if device.type == "cuda" and cuda_state is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_state])
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split-name", default="direction_down_test")
    parser.add_argument("--residual-geometry", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run-role",
        choices=("smoke", "formal_data_only", "formal", "formal_factorized"),
        default="formal",
    )
    parser.add_argument("--physics-mode", choices=("data_only", "energy_and_flux"), default="energy_and_flux")
    parser.add_argument("--epochs", type=int, default=FORMAL_TRAINING["epochs"])
    parser.add_argument("--learning-rate", type=float, default=FORMAL_TRAINING["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=FORMAL_TRAINING["weight_decay"])
    parser.add_argument("--hidden-dim", type=int, default=FORMAL_ARCHITECTURE["hidden_dim"])
    parser.add_argument("--local-pre-iterations", type=int, default=FORMAL_ARCHITECTURE["local_pre_iterations"])
    parser.add_argument("--physics-attention-blocks", type=int, default=FORMAL_ARCHITECTURE["physics_attention_blocks"])
    parser.add_argument("--local-post-iterations", type=int, default=FORMAL_ARCHITECTURE["local_post_iterations"])
    parser.add_argument("--physics-attention-heads", type=int, default=FORMAL_ARCHITECTURE["physics_attention_heads"])
    parser.add_argument("--physics-slices", type=int, default=FORMAL_ARCHITECTURE["physics_slices"])
    parser.add_argument("--temporal-layers", type=int, default=FORMAL_ARCHITECTURE["temporal_layers"])
    parser.add_argument("--temporal-heads", type=int, default=FORMAL_ARCHITECTURE["temporal_heads"])
    parser.add_argument("--spatial-time-chunk-size", type=int, default=1)
    parser.add_argument("--temporal-node-chunk-size", type=int, default=4096)
    parser.add_argument(
        "--spatial-temporal-mode",
        choices=SPATIAL_TEMPORAL_MODES,
        default="repeated_query_spatial",
    )
    parser.add_argument(
        "--temperature-output-mode",
        choices=TEMPERATURE_OUTPUT_MODES,
        default="additive_normalized",
        help=(
            "The formal literature-bounded residual mode maps a normalized "
            "temperature increment to the available heating or cooling range. "
            "It preserves a unit local gradient at a temperature bound while "
            "keeping fluid/solid outputs inside the registered P424 and "
            "P428--P429 ranges."
        ),
    )
    parser.add_argument(
        "--physics-device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
        help=(
            "Device used for the regional conservation residual. It follows "
            "the available CUDA device by default so formal comparisons use "
            "the same implementation path; CPU remains available explicitly "
            "for memory-limited diagnostic runs."
        ),
    )
    parser.add_argument(
        "--physics-time-chunk-size",
        type=int,
        default=8,
        help=(
            "Core time points per regional-residual chunk. Internal chunks add "
            "one neighbouring point on each side to preserve the three-point "
            "time derivative exactly."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--loss-balancing-candidate-id")
    parser.add_argument(
        "--loss-balancing-sources",
        type=Path,
        default=FIXED_FLOW_LOSS_BALANCING_SOURCES,
    )
    parser.add_argument(
        "--evaluation-stage",
        choices=("selection", "final"),
        default="final",
    )
    parser.add_argument("--selected-method-record", type=Path)
    args = parser.parse_args()
    if args.torch_threads is not None:
        if args.torch_threads <= 0:
            raise ValueError("--torch-threads must be positive")
        torch.set_num_threads(args.torch_threads)
    formal_roles = {"formal", "formal_data_only", "formal_factorized"}
    if args.run_role in formal_roles:
        formal_configuration_check(args)
    if args.run_role in formal_roles and args.residual_geometry is None:
        raise ValueError("formal field comparison requires --residual-geometry")
    if args.physics_mode == "energy_and_flux" and args.residual_geometry is None:
        raise ValueError("energy-and-flux training requires --residual-geometry")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive")
    if args.physics_time_chunk_size < 2:
        raise ValueError("--physics-time-chunk-size must be at least two")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.physics_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--physics-device=cuda requires an available CUDA device")
    physics_device = torch.device(args.physics_device)
    index_path = args.dataset_index.resolve()
    root = index_path.parent
    index = json.loads(index_path.read_text(encoding="utf-8"))
    records = sequence_records(index)
    if args.run_role in formal_roles and (
        len(records) != 12 or not all(bool(row["complete"]) for row in records.values())
    ):
        raise ValueError("formal training requires all 12 complete P418 thermal-step curves")
    split = selected_split(set(records), args.splits.resolve(), args.split_name)
    graph = P418ThermalStepRegionalGraph.from_npz(
        root / str(index["regional_geometry_file"]), device=device
    )
    statistics = training_statistics(
        root, records, split["train"], graph.node_type.detach().cpu().numpy()
    )
    condition_mean = torch.as_tensor(statistics["condition_mean"], device=device)
    condition_std = torch.as_tensor(statistics["condition_std"], device=device)
    state_mean, state_std = state_scale_tensors(statistics, graph.node_type, device)
    residual_geometry = None
    if args.residual_geometry is not None:
        residual_geometry = load_p418_subface_geometry(
            args.residual_geometry.resolve(),
            fluid_patch_names=index["boundary_patch_names"]["fluid"],
            solid_patch_names=index["boundary_patch_names"]["solid"],
            device=physics_device,
            dtype=torch.float32,
        )
    model = HCCBP418SpatiotemporalRegionalOperator(
        condition_dim=len(index["condition_names"]),
        hidden_dim=args.hidden_dim,
        local_pre_iterations=args.local_pre_iterations,
        physics_attention_blocks=args.physics_attention_blocks,
        local_post_iterations=args.local_post_iterations,
        physics_attention_heads=args.physics_attention_heads,
        physics_slices=args.physics_slices,
        temporal_layers=args.temporal_layers,
        temporal_heads=args.temporal_heads,
        spatial_time_chunk_size=args.spatial_time_chunk_size,
        temporal_node_chunk_size=args.temporal_node_chunk_size,
        spatial_temporal_mode=args.spatial_temporal_mode,
        boundary_role_count=graph.boundary_role_count,
        temperature_output_mode=args.temperature_output_mode,
        temperature_mean_k_by_node_type=torch.as_tensor(
            np.asarray(statistics["state_mean"])[:, 4], device=device
        ),
        temperature_std_k_by_node_type=torch.as_tensor(
            np.asarray(statistics["state_std"])[:, 4], device=device
        ),
        temperature_bounds_k_by_node_type=torch.as_tensor(
            TEMPERATURE_OUTPUT_BOUNDS_K, device=device
        ),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0.0
    )
    loss_balancer = None
    if args.loss_balancing_candidate_id is not None:
        loss_balancer = build_fixed_flow_loss_balancer(
            source_path=args.loss_balancing_sources.resolve(),
            candidate_id=args.loss_balancing_candidate_id,
            seed=args.seed,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "training_checkpoint.pt"
    contract = checkpoint_contract(args, index_path, split)
    reference_cache: dict[str, dict[str, object]] = {}

    def tensors(sequence_id: str):
        (
            time_np,
            condition_np,
            state_np,
            internal_mass_flux_np,
            boundary_mass_flux_np,
        ) = load_sequence(root, records[sequence_id])
        time_s = torch.as_tensor(time_np, device=device).unsqueeze(0)
        condition = torch.as_tensor(condition_np, device=device).unsqueeze(0)
        state = torch.as_tensor(state_np, device=device).unsqueeze(0)
        normalized_state = (state - state_mean[None, None]) / state_std[None, None]
        normalized_condition = (condition - condition_mean) / condition_std
        normalized_time = time_s / float(statistics["maximum_time_s"])
        internal_mass_flux = torch.as_tensor(
            internal_mass_flux_np, device=device
        ).unsqueeze(0)
        boundary_mass_flux = torch.as_tensor(
            boundary_mass_flux_np, device=device
        ).unsqueeze(0)
        return (
            time_s,
            condition,
            state,
            normalized_state,
            normalized_condition,
            normalized_time,
            internal_mass_flux,
            boundary_mass_flux,
        )

    def reference_physics(
        sequence_id: str,
        time_s,
        condition,
        state,
        internal_mass_flux,
        boundary_mass_flux,
    ):
        if sequence_id not in reference_cache:
            assert residual_geometry is not None
            physics_condition = condition.to(physics_device)
            chunks = []
            absolute_energy_mse = torch.zeros((), dtype=state.dtype)
            for extended, local_core, _, weight in physics_time_chunks(
                state.shape[1], args.physics_time_chunk_size
            ):
                with torch.no_grad():
                    residual = assemble_p418_transient_regional_residual(
                        geometry=residual_geometry,
                        step_condition=physics_condition,
                        state_physical=state[:, extended].to(physics_device),
                        time_s=time_s[:, extended].to(physics_device),
                        fluid_internal_mass_flux_kg_s=time_slice_mass_flux(
                            internal_mass_flux, extended
                        ).to(physics_device),
                        fluid_boundary_mass_flux_kg_s=time_slice_mass_flux(
                            boundary_mass_flux, extended
                        ).to(physics_device),
                    )
                    core = residual_loss_view(residual, local_core)
                    chunks.append(
                        {
                            name: getattr(core, name).detach().cpu()
                            for name in PHYSICS_REFERENCE_FIELDS
                        }
                    )
                    absolute_energy_mse += weight * (
                        dimensionless_transient_energy_loss(
                            core,
                            physics_condition,
                            residual_geometry.fluid_mesh.cell_volume,
                            residual_geometry.solid_mesh.cell_volume,
                        )
                        .detach()
                        .cpu()
                    )
            reference_cache[sequence_id] = {
                "chunks": chunks,
                "absolute_energy_mse": absolute_energy_mse,
            }
        return reference_cache[sequence_id]

    def chunked_physics_losses(
        sequence_id: str,
        prediction: torch.Tensor,
        time_s: torch.Tensor,
        condition: torch.Tensor,
        state: torch.Tensor,
        internal_mass_flux: torch.Tensor,
        boundary_mass_flux: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the unchanged full-history objective in bounded time chunks."""
        assert residual_geometry is not None
        reference = reference_physics(
            sequence_id,
            time_s,
            condition,
            state,
            internal_mass_flux,
            boundary_mass_flux,
        )
        reference_chunks = reference["chunks"]
        if not isinstance(reference_chunks, list):
            raise TypeError("cached reference chunks have an invalid type")
        fluid_volume = residual_geometry.fluid_mesh.cell_volume
        solid_volume = residual_geometry.solid_mesh.cell_volume
        fluid_area = torch.linalg.vector_norm(
            residual_geometry.fluid_mesh.internal_area_vector, dim=1
        )
        solid_area = torch.linalg.vector_norm(
            residual_geometry.solid_mesh.internal_area_vector, dim=1
        )
        physics_condition = condition.to(physics_device)
        totals = [prediction.new_zeros(()) for _ in range(3)]

        for chunk_index, (extended, local_core, _, weight) in enumerate(
            physics_time_chunks(prediction.shape[1], args.physics_time_chunk_size)
        ):
            reference_chunk = {
                name: reference_chunks[chunk_index][name].to(physics_device)
                for name in PHYSICS_REFERENCE_FIELDS
            }
            chunk_state = prediction[:, extended].to(physics_device)
            chunk_time = time_s[:, extended].to(physics_device)
            chunk_internal_flux = time_slice_mass_flux(
                internal_mass_flux, extended
            ).to(physics_device)
            chunk_boundary_flux = time_slice_mass_flux(
                boundary_mass_flux, extended
            ).to(physics_device)

            def evaluate_chunk(
                state_chunk: torch.Tensor,
                time_chunk: torch.Tensor,
                condition_chunk: torch.Tensor,
                internal_flux_chunk: torch.Tensor,
                boundary_flux_chunk: torch.Tensor,
                reference_fluid_energy: torch.Tensor,
                reference_solid_energy: torch.Tensor,
                reference_fluid_flux: torch.Tensor,
                reference_solid_flux: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                residual = assemble_p418_transient_regional_residual(
                    geometry=residual_geometry,
                    step_condition=condition_chunk,
                    state_physical=state_chunk,
                    time_s=time_chunk,
                    fluid_internal_mass_flux_kg_s=internal_flux_chunk,
                    fluid_boundary_mass_flux_kg_s=boundary_flux_chunk,
                )
                core = residual_loss_view(residual, local_core)
                return physics_training_losses(
                    core,
                    {
                        "fluid_energy_w_m3": reference_fluid_energy,
                        "solid_energy_w_m3": reference_solid_energy,
                        "fluid_internal_energy_flux_w": reference_fluid_flux,
                        "solid_internal_heat_flux_w": reference_solid_flux,
                    },
                    condition_chunk,
                    fluid_volume,
                    solid_volume,
                    fluid_area,
                    solid_area,
                )

            chunk_inputs = (
                chunk_state,
                chunk_time,
                physics_condition,
                chunk_internal_flux,
                chunk_boundary_flux,
                reference_chunk["fluid_energy_w_m3"],
                reference_chunk["solid_energy_w_m3"],
                reference_chunk["fluid_internal_energy_flux_w"],
                reference_chunk["solid_internal_heat_flux_w"],
            )
            if torch.is_grad_enabled() and chunk_state.requires_grad:
                losses = checkpoint(
                    evaluate_chunk, *chunk_inputs, use_reentrant=False
                )
            else:
                losses = evaluate_chunk(*chunk_inputs)
            for term_index, chunk_loss in enumerate(losses):
                totals[term_index] = (
                    totals[term_index]
                    + weight * chunk_loss.to(prediction.device)
                )

        return (
            totals[0],
            totals[1],
            totals[2],
            reference["absolute_energy_mse"].to(prediction.device),
        )

    if args.physics_mode == "energy_and_flux":
        physics_roles = (
            ("train", "validation")
            if args.loss_balancing_candidate_id is not None
            and args.evaluation_stage == "selection"
            else ("train", "validation", "test")
        )
        for sequence_id in sorted(
            set().union(*(set(split[role]) for role in physics_roles))
        ):
            (
                reference_time,
                reference_condition,
                reference_state,
                reference_normalized_state,
                reference_normalized_condition,
                reference_normalized_time,
                reference_internal_flux,
                reference_boundary_flux,
            ) = tensors(sequence_id)
            reference_physics(
                sequence_id,
                reference_time,
                reference_condition,
                reference_state,
                reference_internal_flux,
                reference_boundary_flux,
            )
            del (
                reference_time,
                reference_condition,
                reference_state,
                reference_normalized_state,
                reference_normalized_condition,
                reference_normalized_time,
                reference_internal_flux,
                reference_boundary_flux,
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def evaluate(role: str, include_physics: bool = False) -> dict[str, float]:
        model.eval()
        fluid_square = 0.0
        solid_square = 0.0
        fluid_weight = 0.0
        solid_weight = 0.0
        maximum = 0.0
        inference_seconds = 0.0
        normalized_temperature_square = 0.0
        edge_square = 0.0
        projection_energy_square = 0.0
        prediction_absolute_energy_square = 0.0
        reference_absolute_energy_square = 0.0
        hotspot_predictions = []
        hotspot_targets = []
        predicted_solid_minimum = math.inf
        predicted_solid_maximum = -math.inf
        predicted_solid_outside_count = 0
        predicted_solid_value_count = 0
        predicted_fluid_minimum = math.inf
        predicted_fluid_maximum = -math.inf
        predicted_fluid_outside_count = 0
        predicted_fluid_value_count = 0
        with torch.no_grad():
            for sequence_id in split[role]:
                (
                    time_s,
                    condition,
                    state,
                    normalized_state,
                    normalized_condition,
                    normalized_time,
                    internal_mass_flux,
                    boundary_mass_flux,
                ) = tensors(sequence_id)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                inference_start = time.perf_counter()
                prediction_norm = model(
                    normalized_state[:, 0], normalized_condition, normalized_time, graph
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                inference_seconds += time.perf_counter() - inference_start
                prediction = prediction_norm * state_std[None, None] + state_mean[None, None]
                normalized_temperature_square += float(
                    temperature_data_loss(
                        prediction_norm, normalized_state, graph
                    ).cpu()
                )
                error = prediction[..., 4] - state[..., 4]
                hotspot_predictions.append(prediction[0, ..., 4].cpu().numpy())
                hotspot_targets.append(state[0, ..., 4].cpu().numpy())
                solid_prediction = prediction[..., 4][..., graph.node_type == 1]
                fluid_prediction = prediction[..., 4][..., graph.node_type == 0]
                predicted_fluid_minimum = min(
                    predicted_fluid_minimum,
                    float(fluid_prediction.min().cpu()),
                )
                predicted_fluid_maximum = max(
                    predicted_fluid_maximum,
                    float(fluid_prediction.max().cpu()),
                )
                fluid_low_k, fluid_high_k = FLUID_TEMPERATURE_RANGE_K
                predicted_fluid_outside_count += int(
                    (
                        (fluid_prediction < fluid_low_k)
                        | (fluid_prediction > fluid_high_k)
                    )
                    .sum()
                    .cpu()
                )
                predicted_fluid_value_count += fluid_prediction.numel()
                predicted_solid_minimum = min(
                    predicted_solid_minimum,
                    float(solid_prediction.min().cpu()),
                )
                predicted_solid_maximum = max(
                    predicted_solid_maximum,
                    float(solid_prediction.max().cpu()),
                )
                low_k, high_k = SOLID_TEMPERATURE_RANGE_K
                predicted_solid_outside_count += int(
                    ((solid_prediction < low_k) | (solid_prediction > high_k))
                    .sum()
                    .cpu()
                )
                predicted_solid_value_count += solid_prediction.numel()
                fluid = error[..., graph.node_type == 0]
                solid = error[..., graph.node_type == 1]
                fluid_volume = graph.volume_m3[graph.node_type == 0]
                solid_volume = graph.volume_m3[graph.node_type == 1]
                fluid_square += float(
                    (fluid.square() * fluid_volume[None, None, :]).sum().cpu()
                )
                solid_square += float(
                    (solid.square() * solid_volume[None, None, :]).sum().cpu()
                )
                fluid_weight += float(fluid.shape[0] * fluid.shape[1] * fluid_volume.sum().cpu())
                solid_weight += float(solid.shape[0] * solid.shape[1] * solid_volume.sum().cpu())
                maximum = max(maximum, float(error.abs().max().cpu()))
                if include_physics:
                    if residual_geometry is None:
                        raise ValueError("physical evaluation requires residual geometry")
                    (
                        edge_loss,
                        projection_energy_loss,
                        absolute_energy_diagnostic,
                        reference_absolute_energy,
                    ) = chunked_physics_losses(
                        str(sequence_id),
                        prediction,
                        time_s,
                        condition,
                        state,
                        internal_mass_flux,
                        boundary_mass_flux,
                    )
                    edge_square += float(edge_loss.cpu())
                    projection_energy_square += float(
                        projection_energy_loss.cpu()
                    )
                    prediction_absolute_energy_square += float(
                        absolute_energy_diagnostic.cpu()
                    )
                    reference_absolute_energy_square += float(
                        reference_absolute_energy.cpu()
                    )
        result = {
            "fluid_temperature_RMSE_K": math.sqrt(fluid_square / fluid_weight),
            "solid_temperature_RMSE_K": math.sqrt(solid_square / solid_weight),
            "maximum_absolute_temperature_error_K": maximum,
            "normalized_temperature_data_MSE": (
                normalized_temperature_square / len(split[role])
            ),
            "normalized_temperature_data_RMSE": math.sqrt(
                normalized_temperature_square / len(split[role])
            ),
            "inference_seconds": inference_seconds,
            "inference_seconds_per_curve": inference_seconds / len(split[role]),
            "predicted_fluid_temperature_minimum_K": predicted_fluid_minimum,
            "predicted_fluid_temperature_maximum_K": predicted_fluid_maximum,
            "predicted_fluid_temperature_outside_registered_range_fraction": (
                predicted_fluid_outside_count / predicted_fluid_value_count
            ),
            "predicted_solid_temperature_minimum_K": predicted_solid_minimum,
            "predicted_solid_temperature_maximum_K": predicted_solid_maximum,
            "predicted_solid_temperature_outside_registered_range_fraction": (
                predicted_solid_outside_count / predicted_solid_value_count
            ),
        }
        result.update(
            solid_transient_hotspot_metrics(
                np.stack(hotspot_predictions),
                np.stack(hotspot_targets),
                graph.node_type.cpu().numpy(),
                graph.centroid_m.cpu().numpy(),
            )
        )
        if include_physics:
            edge_mse = edge_square / len(split[role])
            projection_energy_mse = projection_energy_square / len(split[role])
            absolute_energy_mse = (
                prediction_absolute_energy_square / len(split[role])
            )
            reference_absolute_energy_mse = (
                reference_absolute_energy_square / len(split[role])
            )
            (
                prediction_energy_rmse,
                reference_energy_rmse,
                prediction_reference_ratio,
            ) = energy_residual_rmse_ratio(
                absolute_energy_mse, reference_absolute_energy_mse
            )
            result.update(
                {
                    "reference_edge_energy_flux_normalized_RMSE": math.sqrt(
                        edge_mse
                    ),
                    "absolute_transient_energy_equation_normalized_RMSE": prediction_energy_rmse,
                    "openfoam_reference_transient_energy_equation_normalized_RMSE": reference_energy_rmse,
                    "prediction_to_openfoam_energy_residual_ratio": prediction_reference_ratio,
                    "projection_aware_energy_equation_normalized_RMSE": math.sqrt(
                        projection_energy_mse
                    ),
                    "weighted_physics_objective": (
                        FORMAL_TRAINING["data_weight"]
                        * result["normalized_temperature_data_MSE"]
                        + FORMAL_TRAINING["edge_flux_weight"] * edge_mse
                        + FORMAL_TRAINING["energy_weight"] * projection_energy_mse
                    ),
                }
            )
        return result

    history = []
    best_validation = math.inf
    best_state = None
    best_epoch = None
    start_epoch = 0
    previous_training_seconds = 0.0
    if args.resume and checkpoint_path.is_file():
        resumed = load_training_checkpoint(
            checkpoint_path,
            contract=contract,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            loss_balancer=loss_balancer,
        )
        start_epoch = int(resumed["next_epoch"])
        best_validation = float(resumed["best_validation"])
        best_state = resumed["best_state"]
        history = list(resumed["history"])
        if history:
            best_epoch = int(
                min(
                    history,
                    key=lambda row: row["validation_selection_score"],
                )["epoch"]
            )
        previous_training_seconds = float(resumed["training_seconds"])
        if start_epoch < 0 or start_epoch > args.epochs or len(history) != start_epoch:
            raise ValueError("training checkpoint has an invalid completed-epoch count")
    start = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        order = np.random.permutation(split["train"])
        epoch_terms = []
        for sequence_id in order:
            (
                time_s,
                condition,
                state,
                normalized_state,
                normalized_condition,
                normalized_time,
                internal_mass_flux,
                boundary_mass_flux,
            ) = tensors(str(sequence_id))
            prediction_norm = model(
                normalized_state[:, 0], normalized_condition, normalized_time, graph
            )
            data_loss = temperature_data_loss(prediction_norm, normalized_state, graph)
            edge_loss = data_loss.new_zeros(())
            projection_energy_loss = data_loss.new_zeros(())
            absolute_energy_diagnostic = data_loss.new_zeros(())
            if args.physics_mode == "energy_and_flux":
                assert residual_geometry is not None
                prediction = prediction_norm * state_std[None, None] + state_mean[None, None]
                (
                    edge_loss,
                    projection_energy_loss,
                    absolute_energy_diagnostic,
                    _,
                ) = chunked_physics_losses(
                    str(sequence_id),
                    prediction,
                    time_s,
                    condition,
                    state,
                    internal_mass_flux,
                    boundary_mass_flux,
                )
            if loss_balancer is None:
                loss = (
                    FORMAL_TRAINING["data_weight"] * data_loss
                    + FORMAL_TRAINING["edge_flux_weight"] * edge_loss
                    + FORMAL_TRAINING["energy_weight"]
                    * projection_energy_loss
                )
                named_weights = {
                    "temperature_data": FORMAL_TRAINING["data_weight"],
                    "reference_edge_energy_flux": FORMAL_TRAINING[
                        "edge_flux_weight"
                    ],
                    "projection_aware_transient_energy": FORMAL_TRAINING[
                        "energy_weight"
                    ],
                }
            else:
                loss, named_weights = balanced_fixed_flow_loss(
                    temperature_data=data_loss,
                    reference_edge_energy_flux=edge_loss,
                    projection_aware_transient_energy=projection_energy_loss,
                    balancer=loss_balancer,
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_terms.append(
                [
                    float(loss.detach().cpu()),
                    float(data_loss.detach().cpu()),
                    float(edge_loss.detach().cpu()),
                    float(projection_energy_loss.detach().cpu()),
                    float(absolute_energy_diagnostic.detach().cpu()),
                    float(
                        torch.as_tensor(
                            named_weights["temperature_data"]
                        )
                        .detach()
                        .cpu()
                    ),
                    float(
                        torch.as_tensor(
                            named_weights["reference_edge_energy_flux"]
                        )
                        .detach()
                        .cpu()
                    ),
                    float(
                        torch.as_tensor(
                            named_weights[
                                "projection_aware_transient_energy"
                            ]
                        )
                        .detach()
                        .cpu()
                    ),
                ]
            )
        scheduler.step()
        validation = evaluate(
            "validation", include_physics=args.physics_mode == "energy_and_flux"
        )
        validation_score = (
            loss_balancing_validation_score(validation)
            if loss_balancer is not None
            else validation_selection_score(validation, args.physics_mode)
        )
        if validation_score < best_validation:
            best_validation = validation_score
            best_epoch = epoch + 1
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        means = np.mean(epoch_terms, axis=0)
        history.append(
            {
                "epoch": epoch + 1,
                "total_loss": means[0],
                "temperature_data_loss": means[1],
                "reference_edge_flux_loss": means[2],
                "projection_aware_energy_loss": means[3],
                "absolute_energy_balance_diagnostic": means[4],
                "temperature_data_weight": means[5],
                "reference_edge_energy_flux_weight": means[6],
                "projection_aware_transient_energy_weight": means[7],
                "validation_solid_temperature_RMSE_K": validation["solid_temperature_RMSE_K"],
                "validation_normalized_temperature_data_MSE": validation[
                    "normalized_temperature_data_MSE"
                ],
                "validation_projection_aware_energy_normalized_RMSE": validation.get(
                    "projection_aware_energy_equation_normalized_RMSE", 0.0
                ),
                "validation_selection_score": validation_score,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if (epoch + 1) % args.checkpoint_every == 0 or epoch + 1 == args.epochs:
            save_training_checkpoint(
                checkpoint_path,
                contract=contract,
                next_epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_validation=best_validation,
                best_state=best_state,
                history=history,
                training_seconds=previous_training_seconds + time.perf_counter() - start,
                loss_balancer=loss_balancer,
            )
    training_seconds = previous_training_seconds + time.perf_counter() - start
    if best_state is None or best_epoch is None:
        raise RuntimeError("training did not produce a model state")
    model.load_state_dict(best_state)
    evaluation_roles = (
        ("train", "validation")
        if loss_balancer is not None and args.evaluation_stage == "selection"
        else ("train", "validation", "test")
    )
    metrics = {
        role: evaluate(
            role,
            include_physics=args.physics_mode == "energy_and_flux",
        )
        for role in evaluation_roles
    }

    prediction_files: dict[str, str] = {}
    prediction_file_records: dict[str, dict[str, object]] = {}
    model.eval()
    with torch.no_grad():
        for role in evaluation_roles:
            sequence_ids = split[role]
            role_time = []
            role_condition = []
            role_condition_normalized = []
            role_fixed_hydrodynamics = []
            role_internal_mass_flux = []
            role_boundary_mass_flux = []
            role_baseline = []
            role_target = []
            for sequence_id in sequence_ids:
                (
                    time_s,
                    condition,
                    state,
                    normalized_state,
                    normalized_condition,
                    normalized_time,
                    internal_mass_flux,
                    boundary_mass_flux,
                ) = tensors(sequence_id)
                prediction = model(
                    normalized_state[:, 0], normalized_condition, normalized_time, graph
                )
                role_time.append(time_s[0].cpu().numpy())
                role_condition.append(condition[0].cpu().numpy())
                role_condition_normalized.append(normalized_condition[0].cpu().numpy())
                role_fixed_hydrodynamics.append(state[0, 0, :, :4].cpu().numpy())
                role_internal_mass_flux.append(internal_mass_flux[0].cpu().numpy())
                role_boundary_mass_flux.append(boundary_mass_flux[0].cpu().numpy())
                role_baseline.append(prediction[0, ..., 4:5].cpu().numpy())
                role_target.append(normalized_state[0, ..., 4:5].cpu().numpy())
            prediction_path = args.output_dir / f"{role}_temporal_temperature_predictions.npz"
            np.savez_compressed(
                prediction_path,
                sequence_id=np.asarray(sequence_ids),
                time_s=np.stack(role_time),
                condition_physical=np.stack(role_condition),
                condition_normalized=np.stack(role_condition_normalized),
                fixed_hydrodynamics_physical=np.stack(role_fixed_hydrodynamics),
                fluid_internal_mass_flux_kg_s=np.stack(role_internal_mass_flux),
                fluid_boundary_mass_flux_kg_s=np.stack(role_boundary_mass_flux),
                baseline_temperature_normalized=np.stack(role_baseline),
                target_temperature_normalized=np.stack(role_target),
                node_type=graph.node_type.cpu().numpy(),
                node_volume_m3=graph.volume_m3.cpu().numpy(),
                node_centroid_m=graph.centroid_m.cpu().numpy(),
                structural_features=graph.structural_features().cpu().numpy(),
                temperature_mean_K_by_node_type=np.asarray(statistics["state_mean"])[:, 4],
                temperature_std_K_by_node_type=np.asarray(statistics["state_std"])[:, 4],
            )
            prediction_files[role] = prediction_path.name
            prediction_file_records[role] = file_record(prediction_path)
    model_state_path = args.output_dir / "model_state.pt"
    torch.save(best_state, model_state_path)
    with (args.output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    training_statistics_path = args.output_dir / "training_statistics.npz"
    np.savez_compressed(
        training_statistics_path,
        condition_mean=statistics["condition_mean"],
        condition_std=statistics["condition_std"],
        state_mean=statistics["state_mean"],
        state_std=statistics["state_std"],
        maximum_time_s=np.asarray(statistics["maximum_time_s"]),
    )
    loss_balancing_summary = None
    loss_balancing_candidate = None
    if loss_balancer is not None:
        loss_balancing_summary = {
            name: value
            for name, value in loss_balancer.state_dict().items()
            if name != "random_state"
        }
        loss_balancing_candidate = load_fixed_flow_candidate(
            args.loss_balancing_sources.resolve(),
            args.loss_balancing_candidate_id,
        )
    summary = {
        "status": "completed_p418_spatiotemporal_regional_operator",
        "run_role": args.run_role,
        "physics_mode": args.physics_mode,
        "dataset_index": str(index_path),
        "input_file_sha256": {
            "dataset_index": sha256_file(index_path),
            "split_file": sha256_file(args.splits.resolve()),
            "residual_geometry": (
                sha256_file(args.residual_geometry.resolve())
                if args.residual_geometry is not None
                else None
            ),
            "loss_balancing_sources": (
                sha256_file(args.loss_balancing_sources.resolve())
                if loss_balancer is not None
                else None
            ),
        },
        "split_name": args.split_name,
        "evaluation_stage": args.evaluation_stage,
        "test_evaluated": "test" in evaluation_roles,
        "selected_method_record": (
            str(args.selected_method_record.resolve())
            if args.selected_method_record is not None
            else None
        ),
        "selected_method_record_sha256": (
            sha256_file(args.selected_method_record.resolve())
            if args.selected_method_record is not None
            else None
        ),
        "seed": args.seed,
        "split_case_counts": {role: len(ids) for role, ids in split.items()},
        "split_case_ids": split,
        "temperature_metric_definition": "regional-volume-weighted RMSE, reported separately for fluid and solid",
        "registered_solid_temperature_range_K": list(SOLID_TEMPERATURE_RANGE_K),
        "registered_fluid_temperature_range_K": list(FLUID_TEMPERATURE_RANGE_K),
        "regional_node_count": graph.node_count,
        "condition_names": index["condition_names"],
        "architecture": {
            "hidden_dim": args.hidden_dim,
            "local_pre_iterations": args.local_pre_iterations,
            "physics_attention_blocks": args.physics_attention_blocks,
            "local_post_iterations": args.local_post_iterations,
            "physics_attention_heads": args.physics_attention_heads,
            "physics_slices": args.physics_slices,
            "temporal_layers": args.temporal_layers,
            "temporal_heads": args.temporal_heads,
            "spatial_time_chunk_size": args.spatial_time_chunk_size,
            "temporal_node_chunk_size": args.temporal_node_chunk_size,
            "spatial_temporal_mode": args.spatial_temporal_mode,
            "temperature_output_mode": args.temperature_output_mode,
            "temperature_output_bounds_K_by_node_type": {
                "fluid": TEMPERATURE_OUTPUT_BOUNDS_K[0].tolist(),
                "solid": TEMPERATURE_OUTPUT_BOUNDS_K[1].tolist(),
            },
            "temperature_output_bound_source_ids": list(
                TEMPERATURE_OUTPUT_BOUND_SOURCE_IDS
            ),
        },
        "loss_weights": {
            "temperature_data": (
                float(loss_balancing_candidate["temperature_data_weight"])
                if loss_balancing_candidate is not None
                else FORMAL_TRAINING["data_weight"]
            ),
            "reference_edge_energy_flux": (
                float(
                    loss_balancing_candidate[
                        "reference_edge_energy_flux_weight"
                    ]
                )
                if loss_balancing_candidate is not None
                else FORMAL_TRAINING["edge_flux_weight"]
            ),
            "projection_aware_transient_energy": (
                float(
                    loss_balancing_candidate[
                        "projection_aware_transient_energy_weight"
                    ]
                )
                if loss_balancing_candidate is not None
                else FORMAL_TRAINING["energy_weight"]
            ),
        },
        "loss_balancing": (
            {
                "candidate_id": args.loss_balancing_candidate_id,
                "method": loss_balancing_candidate["method"],
                "source_file": str(args.loss_balancing_sources.resolve()),
                "source_file_sha256": sha256_file(
                    args.loss_balancing_sources.resolve()
                ),
                "selected_checkpoint_state": loss_balancing_summary,
                "common_validation_score": (
                    "equal mean of dimensionless temperature, reference-edge-flux "
                    "and projection-aware-energy mean-square groups"
                ),
            }
            if loss_balancer is not None
            else {
                "candidate_id": None,
                "method": "legacy_fixed_5_1_1",
                "selected_checkpoint_state": None,
            }
        ),
        "physics_terms": [
            "temperature_data",
            "reference_edge_energy_flux",
            "projection_aware_transient_energy",
        ],
        "training_normalization_sequence_ids": split["train"],
        "physics_loss_definition": (
            "The formal physics term minimizes the difference between predicted and "
            "projected-OpenFOAM fluid/solid transient energy-equation residuals. This "
            "accounts for the fact that regional volume averaging does not commute with "
            "the nonlinear face-flux operator. Reference OpenFOAM edge heat flux is an "
            "additional spatial-gradient target. The absolute regional residual is "
            "reported only as a projection-error diagnostic. Fluid and solid residual "
            "fields are integrated with their finite-volume cell volumes."
        ),
        "metrics": metrics,
        "temporal_temperature_prediction_files": prediction_files,
        "temporal_temperature_prediction_file_records": prediction_file_records,
        "model_state_record": file_record(model_state_path),
        "training_statistics_record": file_record(training_statistics_path),
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "compute_device": str(device),
        "physics_computation_device": str(physics_device),
        "physics_time_chunk_size": args.physics_time_chunk_size,
        "physics_time_chunk_definition": (
            "Core time points per chunk with one neighbouring time retained on "
            "each internal side; this preserves the complete-history "
            "three-point derivative and volume/area-weighted mean losses."
        ),
        "torch_num_threads": torch.get_num_threads(),
        "training_seconds": training_seconds,
        "selection_split": "validation",
        "selection_metric": (
            "equal mean of validation temperature, reference edge heat-flux and "
            "projection-aware transient-energy mean-square groups"
            if loss_balancer is not None
            else "validation weighted normalized temperature, reference edge heat "
            "flux and projection-aware transient energy-operator objective"
            if args.physics_mode == "energy_and_flux"
            else "validation normalized regional temperature MSE"
        ),
        "selected_epoch": best_epoch,
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_validation,
        "training_resumed_from_epoch": start_epoch,
        "training_checkpoint": checkpoint_path.name,
        "checkpoint_every_epochs": args.checkpoint_every,
        "new_physical_parameters": [],
        "parameter_sources": list(P418_TRANSIENT_PARAMETER_IDS),
        "architecture_source_contract": (
            "parameters/hccb_p418_mgnt_temporal_pino_contract.yaml"
        ),
        "transient_case_contract": "parameters/hccb_p418_transient_step_plan.json",
        "scientific_scope": (
            "Regional temperature evolution after target hydrodynamics have adjusted. "
            "Velocity and pressure remain exact inputs; t=0 temperature is imposed exactly."
        ),
    }
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    (args.output_dir / "summary.json").write_text(summary_text, encoding="utf-8")
    if loss_balancer is not None:
        (args.output_dir / f"{args.evaluation_stage}_summary.json").write_text(
            summary_text, encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
