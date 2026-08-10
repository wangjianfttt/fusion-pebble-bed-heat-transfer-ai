#!/usr/bin/env python3
"""Train regional P418 state, mass-flow and energy-flow outputs together."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from hccb_p418_comparison_contract import (
    STEADY_METRIC_CONTRACT,
    file_record,
    integrated_heat_transfer_metrics,
    numerical_state_sha256,
    run_provenance,
    split_indices as comparison_split_indices,
    validate_split_and_statistics,
)
from hccb_p418_conservative_mixed_operator import (
    ConservativeRegionalOutput,
    HCCBP418ConservativeMixedOperator,
    load_regional_energy_flux_geometry,
    load_regional_mass_flux_geometry,
    regional_energy_balance,
    regional_mass_balance,
)
from hccb_p418_parametric_regional_operator import (
    collapse_mesh_to_level,
    load_p418_regional_mesh,
)
from train_hccb_p418_regional_operator import (
    FieldScales,
    build_model,
    load_scales,
    normalized_condition_values,
    source_learning_rate,
)


STEADY_CONSERVATIVE_LOSS_WEIGHTS = {
    "state_data": 5.0,
    "face_flux": 1.0,
    "physics_balance": 1.0,
}


def steady_checkpoint_contract(
    args: argparse.Namespace,
    *,
    split_case_ids: dict[str, list[str]],
    field_architecture: str,
    physics_constrained: bool,
    settings: dict[str, object],
    effective_batch_size: int,
    microbatch_size: int,
    loss_group_weights: dict[str, float],
) -> dict[str, object]:
    """Record the data and numerical settings that must match after a restart."""
    inputs = {
        "regional_topology": file_record(args.regional_topology),
        "model_geometry": file_record(args.model_geometry),
        "state_targets": file_record(args.state_targets),
        "mass_targets": file_record(args.mass_targets),
        "split_file": file_record(args.split_file),
        "training_statistics": file_record(args.training_statistics),
    }
    if args.energy_targets is not None:
        inputs["energy_targets"] = file_record(args.energy_targets)
    code_dir = Path(__file__).resolve().parent
    implementation_files = {
        path.name: file_record(path)
        for path in (
            Path(__file__),
            code_dir / "hccb_p418_comparison_contract.py",
            code_dir / "hccb_p418_conservative_mixed_operator.py",
            code_dir / "train_hccb_p418_regional_operator.py",
            code_dir / "hccb_p418_parametric_regional_operator.py",
            code_dir / "hccb_p418_coordinate_pinn.py",
        )
    }
    return {
        "inputs": inputs,
        "implementation_files": implementation_files,
        "split_name": args.split_name,
        "split_case_ids": split_case_ids,
        "regional_level": args.regional_level,
        "architecture": args.architecture,
        "field_architecture": field_architecture,
        "physics_constrained": physics_constrained,
        "epochs": args.epochs,
        "effective_batch_size": effective_batch_size,
        "microbatch_size": microbatch_size,
        "settings": settings,
        "loss_group_weights": loss_group_weights,
        "seed": args.seed,
        "training_order_seed": 0,
    }


def atomic_torch_save(payload: object, path: Path) -> None:
    """Write a PyTorch file completely before replacing the previous version."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_steady_training_checkpoint(
    path: Path,
    *,
    contract: dict[str, object],
    next_epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    training_rng: np.random.Generator,
    best_validation: float,
    best_epoch: int | None,
    history: list[dict[str, object]],
    training_seconds: float,
    optimization_seconds: float,
    validation_seconds: float,
    update_index: int,
) -> None:
    """Save every state needed to continue at the next complete epoch."""
    atomic_torch_save(
        {
            "contract": contract,
            "next_epoch": next_epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "training_rng_state": training_rng.bit_generator.state,
            "best_validation": best_validation,
            "best_epoch": best_epoch,
            "history": history,
            "training_seconds": training_seconds,
            "optimization_seconds": optimization_seconds,
            "validation_seconds": validation_seconds,
            "update_index": update_index,
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
            "cuda_random_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
        path,
    )


def load_steady_training_checkpoint(
    path: Path,
    *,
    contract: dict[str, object],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    training_rng: np.random.Generator,
    device: torch.device,
) -> dict[str, object]:
    """Restore a complete epoch after verifying data, split and settings."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("contract") != contract:
        raise ValueError(
            "steady checkpoint does not match the current data, split or model settings"
        )
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    saved_scheduler = payload.get("scheduler_state")
    if scheduler is None:
        if saved_scheduler is not None:
            raise ValueError("steady checkpoint contains an unexpected scheduler state")
    else:
        if saved_scheduler is None:
            raise ValueError("steady checkpoint lacks the required scheduler state")
        scheduler.load_state_dict(saved_scheduler)
    training_rng.bit_generator.state = payload["training_rng_state"]
    np.random.set_state(payload["numpy_random_state"])
    torch.random.set_rng_state(payload["torch_random_state"].cpu())
    cuda_state = payload.get("cuda_random_state_all")
    if device.type == "cuda" and cuda_state is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_state])
    return payload


def normalized_state(
    state: np.ndarray,
    condition: np.ndarray,
    node_type: np.ndarray,
    scales: FieldScales,
) -> np.ndarray:
    output = np.zeros_like(state, dtype=np.float32)
    fluid = node_type == 0
    solid = node_type == 1
    output[:, fluid, :3] = (
        state[:, fluid, :3] - scales.velocity_mean
    ) / scales.velocity_std
    output[:, fluid, 3] = (
        state[:, fluid, 3]
        - condition[:, None, 3]
        - scales.pressure_mean
    ) / scales.pressure_std
    output[:, fluid, 4] = (
        state[:, fluid, 4] - scales.fluid_temperature_mean
    ) / scales.fluid_temperature_std
    output[:, solid, 4] = (
        state[:, solid, 4] - scales.solid_temperature_mean
    ) / scales.solid_temperature_std
    return output


def normalized_conditions(condition: np.ndarray, scales: FieldScales) -> np.ndarray:
    return normalized_condition_values(condition, scales)


def regional_state_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    node_type: torch.Tensor,
    volume: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    fluid = node_type == 0
    solid = node_type == 1
    fluid_volume = volume[fluid].sum()
    solid_volume = volume[solid].sum()
    square = (prediction - target).square() * volume.view(1, -1, 1)
    channels = torch.stack(
        tuple(square[:, fluid, index].sum() / (fluid_volume * prediction.shape[0]) for index in range(5))
        + (square[:, solid, 4].sum() / (solid_volume * prediction.shape[0]),)
    )
    return channels.mean(), channels


def selected_loss_term_names(
    *, energy_enabled: bool, physics_constrained: bool
) -> tuple[str, ...]:
    """Return like-for-like supervised terms, optionally adding FV balances."""
    names = ["state", "internal_mass", "boundary_mass"]
    if physics_constrained:
        names.append("continuity")
    if energy_enabled:
        names.extend(("internal_energy", "boundary_energy"))
        if physics_constrained:
            names.append("energy_balance")
    return tuple(names)


def grouped_conservative_loss(
    loss_values: dict[str, torch.Tensor],
    *,
    energy_enabled: bool,
    physics_constrained: bool,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    """Combine normalized losses without changing shared data coefficients."""
    required_weights = {"state_data", "face_flux", "physics_balance"}
    if set(weights) != required_weights:
        raise ValueError(
            f"loss weights must contain exactly {sorted(required_weights)}"
        )
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in weights.values()):
        raise ValueError("loss weights must be finite and positive")

    face_names = ["internal_mass", "boundary_mass"]
    if energy_enabled:
        face_names.extend(("internal_energy", "boundary_energy"))
    missing = [name for name in ("state", *face_names) if name not in loss_values]
    if missing:
        raise ValueError(f"missing supervised loss terms {missing}")

    groups = {
        "state_data": loss_values["state"],
        "face_flux": torch.stack([loss_values[name] for name in face_names]).mean(),
    }
    active_groups = ["state_data", "face_flux"]
    effective_term_weights = {
        "state": float(weights["state_data"]),
        **{
            name: float(weights["face_flux"]) / len(face_names)
            for name in face_names
        },
    }
    if physics_constrained:
        physics_names = ["continuity"]
        if energy_enabled:
            physics_names.append("energy_balance")
        missing = [name for name in physics_names if name not in loss_values]
        if missing:
            raise ValueError(f"missing physical-balance loss terms {missing}")
        groups["physics_balance"] = torch.stack(
            [loss_values[name] for name in physics_names]
        ).mean()
        active_groups.append("physics_balance")
        effective_term_weights.update(
            {
                name: float(weights["physics_balance"]) / len(physics_names)
                for name in physics_names
            }
        )

    total = sum(float(weights[name]) * groups[name] for name in active_groups)
    return total, groups, effective_term_weights


def aggregate_batch_metrics(
    batches: list[tuple[int, dict[str, float | list[float]]]]
) -> dict[str, float | list[float]]:
    """Combine pre-update batch metrics over one complete training epoch."""
    total_cases = sum(count for count, _ in batches)
    if total_cases <= 0:
        raise ValueError("cannot aggregate an empty training epoch")
    keys = set(batches[0][1])
    if any(set(metrics) != keys for _, metrics in batches):
        raise ValueError("training batches report different metric fields")
    output: dict[str, float | list[float]] = {}
    for key in sorted(keys):
        values = [metrics[key] for _, metrics in batches]
        if key == "state_channel_rmse":
            arrays = [np.asarray(value, dtype=np.float64) for value in values]
            output[key] = np.sqrt(
                sum(count * array**2 for (count, _), array in zip(batches, arrays))
                / total_cases
            ).tolist()
        elif key == "total_loss":
            output[key] = float(
                sum(count * float(value) for (count, _), value in zip(batches, values))
                / total_cases
            )
        else:
            output[key] = float(
                np.sqrt(
                    sum(count * float(value) ** 2 for (count, _), value in zip(batches, values))
                    / total_cases
                )
            )
    return output


def weighted_microbatches(
    indices: np.ndarray, microbatch_size: int
) -> list[tuple[np.ndarray, float]]:
    """Partition one optimizer batch without changing its mean-loss gradient."""
    if microbatch_size <= 0:
        raise ValueError("microbatch size must be positive")
    if len(indices) == 0:
        raise ValueError("cannot partition an empty optimizer batch")
    return [
        (
            indices[start : start + microbatch_size],
            len(indices[start : start + microbatch_size]) / len(indices),
        )
        for start in range(0, len(indices), microbatch_size)
    ]


def incident_flux_rms(
    internal: np.ndarray,
    boundary: np.ndarray,
    internal_owner: np.ndarray,
    internal_neighbour: np.ndarray,
    boundary_owner: np.ndarray,
    node_count: int,
    source: np.ndarray | None = None,
) -> float:
    if internal.shape[0] != boundary.shape[0]:
        raise ValueError("internal and boundary flow case counts differ")
    if source is not None and source.shape[0] != internal.shape[0]:
        raise ValueError("source and flow case counts differ")
    values: list[np.ndarray] = []
    for case_index, (case_internal, case_boundary) in enumerate(
        zip(internal, boundary)
    ):
        incident = np.zeros(node_count, dtype=np.float64)
        np.add.at(incident, internal_owner, np.abs(case_internal))
        np.add.at(incident, internal_neighbour, np.abs(case_internal))
        np.add.at(incident, boundary_owner, np.abs(case_boundary))
        if source is not None:
            incident += np.abs(source[case_index])
        values.append(incident)
    scale = float(np.sqrt(np.mean(np.square(values))))
    if not scale > 0.0:
        raise ValueError("incident flow scale is not positive")
    return scale


def physical_state(
    normalized: np.ndarray,
    condition: np.ndarray,
    node_type: np.ndarray,
    scales: FieldScales,
) -> np.ndarray:
    output = np.zeros_like(normalized, dtype=np.float64)
    fluid = node_type == 0
    solid = node_type == 1
    output[fluid, :3] = normalized[fluid, :3] * scales.velocity_std + scales.velocity_mean
    output[fluid, 3] = normalized[fluid, 3] * scales.pressure_std + scales.pressure_mean + condition[3]
    output[fluid, 4] = normalized[fluid, 4] * scales.fluid_temperature_std + scales.fluid_temperature_mean
    output[solid, 4] = normalized[solid, 4] * scales.solid_temperature_std + scales.solid_temperature_mean
    return output


def engineering_metrics(
    state: np.ndarray,
    *,
    boundary_owner: np.ndarray,
    boundary_patch: np.ndarray,
    boundary_area: np.ndarray,
    inlet_patch: int,
    outlet_patch: int,
    node_type: np.ndarray,
) -> dict[str, float]:
    inlet = boundary_patch == inlet_patch
    outlet = boundary_patch == outlet_patch
    inlet_pressure = float(np.sum(state[boundary_owner[inlet], 3] * boundary_area[inlet]) / np.sum(boundary_area[inlet]))
    outlet_pressure = float(np.sum(state[boundary_owner[outlet], 3] * boundary_area[outlet]) / np.sum(boundary_area[outlet]))
    outlet_temperature = float(np.sum(state[boundary_owner[outlet], 4] * boundary_area[outlet]) / np.sum(boundary_area[outlet]))
    return {
        "pressure_drop_Pa": inlet_pressure - outlet_pressure,
        "outlet_temperature_K": outlet_temperature,
        "solid_maximum_temperature_K": float(np.max(state[node_type == 1, 4])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regional-topology", type=Path, required=True)
    parser.add_argument("--model-geometry", type=Path, required=True)
    parser.add_argument("--state-targets", type=Path, required=True)
    parser.add_argument("--mass-targets", type=Path, required=True)
    parser.add_argument("--energy-targets", type=Path)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--training-statistics", type=Path, required=True)
    parser.add_argument("--split-name", default="pilot_smoke")
    parser.add_argument("--regional-level", type=int, default=5)
    parser.add_argument(
        "--architecture",
        choices=("pinn_data_only", "pinn", "graph", "transolver"),
        default="transolver",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--effective-batch-size", type=int)
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--state-data-weight",
        type=float,
        default=STEADY_CONSERVATIVE_LOSS_WEIGHTS["state_data"],
    )
    parser.add_argument(
        "--face-flux-weight",
        type=float,
        default=STEADY_CONSERVATIVE_LOSS_WEIGHTS["face_flux"],
    )
    parser.add_argument(
        "--physics-balance-weight",
        type=float,
        default=STEADY_CONSERVATIVE_LOSS_WEIGHTS["physics_balance"],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.epochs <= 0 or args.threads <= 0:
        raise ValueError("epochs and threads must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive")
    loss_group_weights = {
        "state_data": float(args.state_data_weight),
        "face_flux": float(args.face_flux_weight),
        "physics_balance": float(args.physics_balance_weight),
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in loss_group_weights.values()):
        raise ValueError("loss weights must be finite and positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    field_architecture = "pinn" if args.architecture == "pinn_data_only" else args.architecture
    physics_constrained = args.architecture != "pinn_data_only"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.state_targets.resolve(), allow_pickle=False) as loaded:
        condition_ids = loaded["condition_id"].astype(str)
        condition_physical = loaded["condition_physical"].astype(np.float64)
        state_physical = loaded["state_physical"].astype(np.float64)
        node_type_np = loaded["node_type"].astype(np.int64)
        node_volume_np = loaded["node_volume_m3"].astype(np.float64)
    with np.load(args.mass_targets.resolve(), allow_pickle=False) as loaded:
        mass_ids = loaded["condition_id"].astype(str)
        internal_target_np = loaded["internal_mass_flow_kg_s"].astype(np.float64)
        boundary_target_np = loaded["boundary_mass_flow_kg_s"].astype(np.float64)
        internal_owner_np = loaded["internal_owner"].astype(np.int64)
        internal_neighbour_np = loaded["internal_neighbour"].astype(np.int64)
        boundary_owner_np = loaded["boundary_owner"].astype(np.int64)
        boundary_patch_np = loaded["boundary_patch"].astype(np.int64)
        boundary_area_np = loaded["boundary_face_area_m2"].astype(np.float64)
    if not np.array_equal(condition_ids, mass_ids):
        raise ValueError("state and mass target case orders differ")
    energy_enabled = args.energy_targets is not None
    if energy_enabled:
        with np.load(args.energy_targets.resolve(), allow_pickle=False) as loaded:
            energy_ids = loaded["condition_id"].astype(str)
            energy_node_type_np = loaded["node_type"].astype(np.int64)
            energy_internal_target_np = loaded["internal_energy_flow_W"].astype(np.float64)
            energy_boundary_target_np = loaded["boundary_energy_flow_W"].astype(np.float64)
            energy_source_np = loaded["node_source_power_W"].astype(np.float64)
            energy_internal_owner_np = loaded["internal_owner"].astype(np.int64)
            energy_internal_neighbour_np = loaded["internal_neighbour"].astype(np.int64)
            energy_internal_kind_np = loaded["internal_kind"].astype(np.int64)
            energy_internal_kind_name_np = loaded["internal_kind_name"].astype(str)
            energy_boundary_owner_np = loaded["boundary_owner"].astype(np.int64)
            energy_boundary_kind_np = loaded["boundary_kind"].astype(np.int64)
            energy_boundary_kind_name_np = loaded["boundary_kind_name"].astype(str)
            energy_internal_kind_count = int(np.max(loaded["internal_kind"])) + 1
            energy_boundary_kind_count = int(np.max(loaded["boundary_kind"])) + 1
        if not np.array_equal(condition_ids, energy_ids):
            raise ValueError("state and energy target case orders differ")
        if not np.array_equal(node_type_np, energy_node_type_np):
            raise ValueError("state and energy target node orders differ")
    else:
        energy_internal_target_np = None
        energy_boundary_target_np = None
        energy_source_np = None
        energy_internal_owner_np = None
        energy_internal_neighbour_np = None
        energy_internal_kind_np = None
        energy_internal_kind_name_np = None
        energy_boundary_owner_np = None
        energy_boundary_kind_np = None
        energy_boundary_kind_name_np = None
        energy_internal_kind_count = None
        energy_boundary_kind_count = None
    split_case_ids, _ = validate_split_and_statistics(
        split_file=args.split_file,
        training_statistics=args.training_statistics,
        split_name=args.split_name,
        condition_ids=condition_ids,
    )
    split_indices = comparison_split_indices(split_case_ids, condition_ids)
    scales = load_scales(args.training_statistics.resolve(), args.split_name)
    train_index = split_indices["train"]
    condition_normalized = normalized_conditions(condition_physical, scales)
    state_normalized = normalized_state(
        state_physical, condition_physical, node_type_np, scales
    )
    internal_scale = float(np.sqrt(np.mean(np.square(internal_target_np[train_index]))))
    boundary_scale = float(np.sqrt(np.mean(np.square(boundary_target_np[train_index]))))
    balance_scale = incident_flux_rms(
        internal_target_np[train_index],
        boundary_target_np[train_index],
        internal_owner_np,
        internal_neighbour_np,
        boundary_owner_np,
        int(np.count_nonzero(node_type_np == 0)),
    )
    if energy_enabled:
        energy_internal_scale = float(
            np.sqrt(np.mean(np.square(energy_internal_target_np[train_index])))
        )
        energy_boundary_scale = float(
            np.sqrt(np.mean(np.square(energy_boundary_target_np[train_index])))
        )
        energy_balance_scale = incident_flux_rms(
            energy_internal_target_np[train_index],
            energy_boundary_target_np[train_index],
            energy_internal_owner_np,
            energy_internal_neighbour_np,
            energy_boundary_owner_np,
            len(node_type_np),
            energy_source_np[train_index],
        )
    else:
        energy_internal_scale = None
        energy_boundary_scale = None
        energy_balance_scale = None

    mesh = collapse_mesh_to_level(
        load_p418_regional_mesh(
            args.regional_topology.resolve(), args.model_geometry.resolve()
        ),
        args.regional_level,
    ).to(device)
    if not torch.equal(mesh.levels[0].node_type.cpu(), torch.as_tensor(node_type_np)):
        raise ValueError("regional target node order differs from the model mesh")
    field_operator, settings = build_model(
        field_architecture, int(mesh.fine_boundary_role.shape[1])
    )
    model = HCCBP418ConservativeMixedOperator(
        field_operator=field_operator,
        patch_count=int(np.max(boundary_patch_np)) + 1,
        internal_mass_scale_kg_s=internal_scale,
        boundary_mass_scale_kg_s=boundary_scale,
        internal_energy_scale_W=energy_internal_scale,
        boundary_energy_scale_W=energy_boundary_scale,
        internal_energy_kind_count=energy_internal_kind_count,
        boundary_energy_kind_count=energy_boundary_kind_count,
    ).to(device)
    initial_model_state_sha256 = numerical_state_sha256(model.state_dict())
    model_parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    effective_batch_size = (
        args.effective_batch_size
        if args.effective_batch_size is not None
        else int(settings["effective_batch_size"])
    )
    if effective_batch_size <= 0:
        raise ValueError("effective batch size must be positive")
    microbatch_size = (
        args.microbatch_size
        if args.microbatch_size is not None
        else effective_batch_size
    )
    if microbatch_size <= 0 or microbatch_size > effective_batch_size:
        raise ValueError("microbatch size must be positive and no larger than the effective batch")
    updates_per_epoch = math.ceil(len(train_index) / effective_batch_size)
    total_updates = args.epochs * updates_per_epoch
    flux_geometry = load_regional_mass_flux_geometry(
        args.mass_targets.resolve(),
        patch_count=int(np.max(boundary_patch_np)) + 1,
        device=device,
        dtype=torch.float32,
    )
    energy_geometry = (
        load_regional_energy_flux_geometry(
            args.energy_targets.resolve(), device=device, dtype=torch.float32
        )
        if energy_enabled
        else None
    )
    condition_t = torch.as_tensor(condition_normalized, device=device)
    state_t = torch.as_tensor(state_normalized, device=device)
    internal_t = torch.as_tensor(internal_target_np, device=device, dtype=torch.float32)
    boundary_t = torch.as_tensor(boundary_target_np, device=device, dtype=torch.float32)
    energy_internal_t = (
        torch.as_tensor(energy_internal_target_np, device=device, dtype=torch.float32)
        if energy_enabled
        else None
    )
    energy_boundary_t = (
        torch.as_tensor(energy_boundary_target_np, device=device, dtype=torch.float32)
        if energy_enabled
        else None
    )
    energy_source_t = (
        torch.as_tensor(energy_source_np, device=device, dtype=torch.float32)
        if energy_enabled
        else None
    )
    node_type = torch.as_tensor(node_type_np, device=device)
    node_volume = torch.as_tensor(node_volume_np, device=device, dtype=torch.float32)
    if field_architecture == "pinn":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(settings["learning_rate"])
        )
        scheduler = None
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(settings["learning_rate"]),
            weight_decay=float(settings["weight_decay"]),
        )
        scheduler = (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(settings["learning_rate"]),
                total_steps=total_updates,
            )
            if args.architecture == "transolver"
            else None
        )
    optimizer_name = type(optimizer).__name__

    def loss_for(
        indices: np.ndarray, backward_weight: float | None
    ) -> tuple[torch.Tensor, dict[str, float | list[float]]]:
        selected = torch.as_tensor(indices, device=device, dtype=torch.long)
        prediction = model(
            condition_t[selected], mesh, flux_geometry, energy_geometry
        )
        state_loss, state_channels = regional_state_loss(
            prediction.regional_state, state_t[selected], node_type, node_volume
        )
        internal_loss = torch.mean(
            ((prediction.internal_mass_flow_kg_s - internal_t[selected]) / internal_scale).square()
        )
        boundary_loss = torch.mean(
            ((prediction.boundary_mass_flow_kg_s - boundary_t[selected]) / boundary_scale).square()
        )
        balance = regional_mass_balance(
            prediction, flux_geometry, int(np.count_nonzero(node_type_np == 0))
        )
        continuity_loss = torch.mean((balance / balance_scale).square())
        loss_values = {
            "state": state_loss,
            "internal_mass": internal_loss,
            "boundary_mass": boundary_loss,
            "continuity": continuity_loss,
        }
        metrics = {
            "state_normalized_rmse": float(torch.sqrt(state_loss).detach()),
            "internal_flux_normalized_rmse": float(torch.sqrt(internal_loss).detach()),
            "boundary_flux_normalized_rmse": float(torch.sqrt(boundary_loss).detach()),
            "continuity_normalized_rmse": float(torch.sqrt(continuity_loss).detach()),
            "state_channel_rmse": torch.sqrt(state_channels.detach()).cpu().tolist(),
        }
        if energy_enabled:
            energy_internal_loss = torch.mean(
                (
                    (prediction.internal_energy_flow_W - energy_internal_t[selected])
                    / energy_internal_scale
                ).square()
            )
            energy_boundary_loss = torch.mean(
                (
                    (prediction.boundary_energy_flow_W - energy_boundary_t[selected])
                    / energy_boundary_scale
                ).square()
            )
            predicted_energy_balance = regional_energy_balance(
                prediction, energy_geometry, energy_source_t[selected]
            )
            target_energy_output = ConservativeRegionalOutput(
                regional_state=prediction.regional_state,
                internal_mass_flow_kg_s=prediction.internal_mass_flow_kg_s,
                boundary_mass_flow_kg_s=prediction.boundary_mass_flow_kg_s,
                internal_energy_flow_W=energy_internal_t[selected],
                boundary_energy_flow_W=energy_boundary_t[selected],
            )
            target_energy_balance = regional_energy_balance(
                target_energy_output, energy_geometry, energy_source_t[selected]
            )
            energy_balance_loss = torch.mean(
                (
                    (predicted_energy_balance - target_energy_balance)
                    / energy_balance_scale
                ).square()
            )
            loss_values.update(
                {
                    "internal_energy": energy_internal_loss,
                    "boundary_energy": energy_boundary_loss,
                    "energy_balance": energy_balance_loss,
                }
            )
            metrics.update(
                {
                    "internal_energy_normalized_rmse": float(
                        torch.sqrt(energy_internal_loss).detach()
                    ),
                    "boundary_energy_normalized_rmse": float(
                        torch.sqrt(energy_boundary_loss).detach()
                    ),
                    "energy_balance_normalized_rmse": float(
                        torch.sqrt(energy_balance_loss).detach()
                    ),
                }
            )
        total, loss_groups, _ = grouped_conservative_loss(
            loss_values,
            energy_enabled=energy_enabled,
            physics_constrained=physics_constrained,
            weights=loss_group_weights,
        )
        metrics["state_data_group_normalized_rmse"] = float(
            torch.sqrt(loss_groups["state_data"]).detach()
        )
        metrics["face_flux_group_normalized_rmse"] = float(
            torch.sqrt(loss_groups["face_flux"]).detach()
        )
        if physics_constrained:
            metrics["physics_balance_group_normalized_rmse"] = float(
                torch.sqrt(loss_groups["physics_balance"]).detach()
            )
        if backward_weight is not None:
            (total * backward_weight).backward()
        metrics["total_loss"] = float(total.detach())
        return total, metrics

    def evaluate_loss(
        indices: np.ndarray,
    ) -> tuple[float, dict[str, float | list[float]]]:
        parts: list[tuple[int, dict[str, float | list[float]]]] = []
        weighted_loss = 0.0
        for start in range(0, len(indices), microbatch_size):
            part = indices[start : start + microbatch_size]
            loss, metrics = loss_for(part, None)
            weighted_loss += len(part) * float(loss.detach())
            parts.append((len(part), metrics))
        return weighted_loss / len(indices), aggregate_batch_metrics(parts)

    def predict_in_microbatches(
        indices: np.ndarray,
    ) -> tuple[ConservativeRegionalOutput, np.ndarray, np.ndarray | None]:
        states: list[torch.Tensor] = []
        internal_mass: list[torch.Tensor] = []
        boundary_mass: list[torch.Tensor] = []
        internal_energy: list[torch.Tensor] = []
        boundary_energy: list[torch.Tensor] = []
        mass_balances: list[torch.Tensor] = []
        energy_balances: list[torch.Tensor] = []
        for start in range(0, len(indices), microbatch_size):
            part = indices[start : start + microbatch_size]
            selected = torch.as_tensor(part, device=device, dtype=torch.long)
            prediction = model(
                condition_t[selected], mesh, flux_geometry, energy_geometry
            )
            states.append(prediction.regional_state.cpu())
            internal_mass.append(prediction.internal_mass_flow_kg_s.cpu())
            boundary_mass.append(prediction.boundary_mass_flow_kg_s.cpu())
            mass_balances.append(
                regional_mass_balance(
                    prediction,
                    flux_geometry,
                    int(np.count_nonzero(node_type_np == 0)),
                ).cpu()
            )
            if energy_enabled:
                internal_energy.append(prediction.internal_energy_flow_W.cpu())
                boundary_energy.append(prediction.boundary_energy_flow_W.cpu())
                energy_balances.append(
                    regional_energy_balance(
                        prediction, energy_geometry, energy_source_t[selected]
                    ).cpu()
                )
        output = ConservativeRegionalOutput(
            regional_state=torch.cat(states, dim=0),
            internal_mass_flow_kg_s=torch.cat(internal_mass, dim=0),
            boundary_mass_flow_kg_s=torch.cat(boundary_mass, dim=0),
            internal_energy_flow_W=(
                torch.cat(internal_energy, dim=0) if energy_enabled else None
            ),
            boundary_energy_flow_W=(
                torch.cat(boundary_energy, dim=0) if energy_enabled else None
            ),
        )
        return (
            output,
            torch.cat(mass_balances, dim=0).numpy(),
            torch.cat(energy_balances, dim=0).numpy() if energy_enabled else None,
        )

    training_contract = steady_checkpoint_contract(
        args,
        split_case_ids=split_case_ids,
        field_architecture=field_architecture,
        physics_constrained=physics_constrained,
        settings=settings,
        effective_batch_size=effective_batch_size,
        microbatch_size=microbatch_size,
        loss_group_weights=loss_group_weights,
    )
    checkpoint_path = output_dir / "training_checkpoint.pt"
    best_path = output_dir / "best.pt"
    history_path = output_dir / "training_history.jsonl"
    history: list[dict[str, object]] = []
    best_validation = math.inf
    best_epoch: int | None = None
    start_epoch = 0
    previous_training_seconds = 0.0
    optimization_seconds = 0.0
    validation_seconds = 0.0
    training_rng = np.random.default_rng(0)
    update_index = 0
    if args.resume and checkpoint_path.is_file():
        resumed = load_steady_training_checkpoint(
            checkpoint_path,
            contract=training_contract,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_rng=training_rng,
            device=device,
        )
        start_epoch = int(resumed["next_epoch"])
        history = list(resumed["history"])
        best_validation = float(resumed["best_validation"])
        best_epoch = (
            int(resumed["best_epoch"])
            if resumed["best_epoch"] is not None
            else None
        )
        previous_training_seconds = float(resumed["training_seconds"])
        optimization_seconds = float(resumed["optimization_seconds"])
        validation_seconds = float(resumed["validation_seconds"])
        update_index = int(resumed["update_index"])
        if start_epoch < 0 or start_epoch > args.epochs:
            raise ValueError("steady checkpoint has an invalid completed-epoch count")
        if len(history) != start_epoch:
            raise ValueError("steady checkpoint history does not match its completed epochs")
        if update_index != start_epoch * updates_per_epoch:
            raise ValueError("steady checkpoint update count does not match its completed epochs")
        if best_epoch is None or not best_path.is_file():
            raise ValueError("steady checkpoint lacks its validation-selected model")
        saved_best = torch.load(best_path, map_location="cpu", weights_only=False)
        if int(saved_best.get("epoch", -1)) != best_epoch:
            raise ValueError("steady best model does not match the restart checkpoint")
    elif not args.resume:
        for stale in (checkpoint_path, best_path, history_path):
            stale.unlink(missing_ok=True)
    with history_path.open("w", encoding="utf-8") as stream:
        for row in history:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.perf_counter()
        model.train()
        shuffled = train_index.copy()
        training_rng.shuffle(shuffled)
        batch_metrics: list[tuple[int, dict[str, float | list[float]]]] = []
        learning_rate = float(optimizer.param_groups[0]["lr"])
        for start in range(0, len(shuffled), effective_batch_size):
            update_started = time.perf_counter()
            batch = shuffled[start : start + effective_batch_size]
            optimizer.zero_grad(set_to_none=True)
            if field_architecture == "graph":
                learning_rate = source_learning_rate(update_index, total_updates)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
            else:
                learning_rate = float(optimizer.param_groups[0]["lr"])
            for microbatch, backward_weight in weighted_microbatches(
                batch, microbatch_size
            ):
                _, current_metrics = loss_for(
                    microbatch, backward_weight
                )
                batch_metrics.append((len(microbatch), current_metrics))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            update_index += 1
            optimization_seconds += time.perf_counter() - update_started
        train_metrics = aggregate_batch_metrics(batch_metrics)
        model.eval()
        validation_started = time.perf_counter()
        with torch.no_grad():
            validation_loss, validation_metrics = evaluate_loss(
                split_indices["validation"]
            )
        validation_seconds += time.perf_counter() - validation_started
        row = {
            "epoch": epoch + 1,
            "learning_rate": learning_rate,
            "parameter_updates": updates_per_epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "epoch_wall_time_s": time.perf_counter() - epoch_started,
        }
        history.append(row)
        with (output_dir / "training_history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch + 1
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "epoch": best_epoch,
                    "validation_total_loss": best_validation,
                    "settings": settings,
                    "contract": training_contract,
                },
                best_path,
            )
        if (epoch + 1) % args.checkpoint_every == 0 or epoch + 1 == args.epochs:
            save_steady_training_checkpoint(
                checkpoint_path,
                contract=training_contract,
                next_epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                training_rng=training_rng,
                best_validation=best_validation,
                best_epoch=best_epoch,
                history=history,
                training_seconds=(
                    previous_training_seconds + time.perf_counter() - started
                ),
                optimization_seconds=optimization_seconds,
                validation_seconds=validation_seconds,
                update_index=update_index,
            )

    training_seconds = previous_training_seconds + time.perf_counter() - started
    if best_epoch is None or not best_path.is_file():
        raise RuntimeError("steady training did not produce a validation-selected model")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    evaluations: dict[str, object] = {}
    prediction_files: dict[str, str] = {}
    prediction_file_records: dict[str, dict[str, object]] = {}
    inlet_patch = 0
    outlet_patch = 1
    final_evaluation_started = time.perf_counter()
    with torch.no_grad():
        for split_name in ("train", "validation", "test"):
            indices = split_indices[split_name]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_started = time.perf_counter()
            prediction, balance, predicted_energy_balance_np = predict_in_microbatches(
                indices
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds = time.perf_counter() - inference_started
            prediction_path = output_dir / f"{split_name}_regional_predictions.npz"
            np.savez_compressed(
                prediction_path,
                condition_id=condition_ids[indices],
                condition_normalized=condition_normalized[indices],
                baseline_state_normalized=prediction.regional_state.cpu().numpy(),
                target_state_normalized=state_normalized[indices],
                internal_mass_flow_kg_s=prediction.internal_mass_flow_kg_s.cpu().numpy(),
                boundary_mass_flow_kg_s=prediction.boundary_mass_flow_kg_s.cpu().numpy(),
                node_type=node_type_np,
                node_volume_m3=node_volume_np,
            )
            prediction_files[split_name] = prediction_path.name
            prediction_file_records[split_name] = file_record(prediction_path)
            _, metrics = evaluate_loss(indices)
            cases: list[dict[str, object]] = []
            for local_position, case_index in enumerate(indices):
                inlet_mass = abs(float(np.sum(boundary_target_np[case_index][boundary_patch_np == inlet_patch])))
                predicted_state = physical_state(
                    prediction.regional_state[local_position].cpu().numpy(),
                    condition_physical[case_index], node_type_np, scales
                )
                reference_state = state_physical[case_index]
                predicted_engineering = engineering_metrics(
                    predicted_state,
                    boundary_owner=boundary_owner_np,
                    boundary_patch=boundary_patch_np,
                    boundary_area=boundary_area_np,
                    inlet_patch=inlet_patch,
                    outlet_patch=outlet_patch,
                    node_type=node_type_np,
                )
                reference_engineering = engineering_metrics(
                    reference_state,
                    boundary_owner=boundary_owner_np,
                    boundary_patch=boundary_patch_np,
                    boundary_area=boundary_area_np,
                    inlet_patch=inlet_patch,
                    outlet_patch=outlet_patch,
                    node_type=node_type_np,
                )
                predicted_heat_transfer = integrated_heat_transfer_metrics(
                    internal_energy_flow_w=prediction.internal_energy_flow_W[
                        local_position
                    ].cpu().numpy(),
                    boundary_energy_flow_w=prediction.boundary_energy_flow_W[
                        local_position
                    ].cpu().numpy(),
                    internal_kind=energy_internal_kind_np,
                    internal_kind_name=energy_internal_kind_name_np,
                    boundary_kind=energy_boundary_kind_np,
                    boundary_kind_name=energy_boundary_kind_name_np,
                )
                reference_heat_transfer = integrated_heat_transfer_metrics(
                    internal_energy_flow_w=energy_internal_target_np[case_index],
                    boundary_energy_flow_w=energy_boundary_target_np[case_index],
                    internal_kind=energy_internal_kind_np,
                    internal_kind_name=energy_internal_kind_name_np,
                    boundary_kind=energy_boundary_kind_np,
                    boundary_kind_name=energy_boundary_kind_name_np,
                )
                cases.append(
                    {
                        "condition_id": str(condition_ids[case_index]),
                        "generated_power_W": float(
                            np.sum(energy_source_np[case_index])
                        ),
                        "local_mass_l1_over_two_inlet": float(np.sum(np.abs(balance[local_position])) / (2.0 * inlet_mass)),
                        "global_mass_imbalance_over_inlet": float(abs(np.sum(balance[local_position])) / inlet_mass),
                        "engineering_absolute_errors": {
                            **{
                                name: abs(
                                    predicted_engineering[name]
                                    - reference_engineering[name]
                                )
                                for name in predicted_engineering
                            },
                            **{
                                name: abs(
                                    predicted_heat_transfer[name]
                                    - reference_heat_transfer[name]
                                )
                                for name in predicted_heat_transfer
                            },
                        },
                        "predicted_engineering": {
                            **{
                                name: float(value)
                                for name, value in predicted_engineering.items()
                            },
                            **{
                                name: float(value)
                                for name, value in predicted_heat_transfer.items()
                            },
                        },
                        "reference_engineering": {
                            **{
                                name: float(value)
                                for name, value in reference_engineering.items()
                            },
                            **{
                                name: float(value)
                                for name, value in reference_heat_transfer.items()
                            },
                        },
                        **(
                            {
                                "local_energy_l1_over_two_generated_power": float(
                                    np.sum(np.abs(predicted_energy_balance_np[local_position]))
                                    / (2.0 * np.sum(energy_source_np[case_index]))
                                ),
                                "global_energy_imbalance_over_generated_power": float(
                                    abs(np.sum(predicted_energy_balance_np[local_position]))
                                    / np.sum(energy_source_np[case_index])
                                ),
                            }
                            if energy_enabled
                            else {}
                        ),
                    }
                )
            evaluations[split_name] = {
                "metrics": metrics,
                "cases": cases,
                "inference_seconds": inference_seconds,
                "inference_seconds_per_case": inference_seconds / len(indices),
            }
    final_evaluation_seconds = time.perf_counter() - final_evaluation_started

    code_dir = Path(__file__).resolve().parent
    common_inputs = {
        "state_targets": args.state_targets,
        "mass_targets": args.mass_targets,
        "split_file": args.split_file,
        "training_statistics": args.training_statistics,
    }
    if args.energy_targets is not None:
        common_inputs["energy_targets"] = args.energy_targets
    provenance = run_provenance(
        architecture=args.architecture,
        comparison_epochs=args.epochs,
        split_name=args.split_name,
        split_case_ids=split_case_ids,
        common_inputs=common_inputs,
        implementation_files=(
            Path(__file__),
            code_dir / "hccb_p418_comparison_contract.py",
            code_dir / "hccb_p418_conservative_mixed_operator.py",
            code_dir / "train_hccb_p418_regional_operator.py",
            code_dir / "hccb_p418_parametric_regional_operator.py",
            code_dir / "hccb_p418_coordinate_pinn.py",
        ),
    )
    active_loss_names = selected_loss_term_names(
        energy_enabled=energy_enabled,
        physics_constrained=physics_constrained,
    )
    placeholder_losses = {
        name: torch.tensor(0.0, device=device) for name in active_loss_names
    }
    _, active_loss_groups, effective_loss_term_weights = grouped_conservative_loss(
        placeholder_losses,
        energy_enabled=energy_enabled,
        physics_constrained=physics_constrained,
        weights=loss_group_weights,
    )
    summary = {
        "status": "conservative_mixed_operator_training_complete",
        "architecture": args.architecture,
        "split_name": args.split_name,
        "split_case_ids": split_case_ids,
        "epochs": args.epochs,
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_total_loss": float(best_validation),
        "checkpoint_selection_rule": "minimum validation total loss; independent test cases are not used",
        "training_seconds": training_seconds,
        "training_resumed_from_epoch": start_epoch,
        "training_checkpoint": checkpoint_path.name,
        "checkpoint_every_epochs": args.checkpoint_every,
        "optimization_seconds": optimization_seconds,
        "optimization_seconds_per_update": optimization_seconds / total_updates,
        "validation_seconds": validation_seconds,
        "final_evaluation_seconds": final_evaluation_seconds,
        "model_parameter_count": model_parameter_count,
        "initial_model_state_sha256": initial_model_state_sha256,
        "effective_batch_size": effective_batch_size,
        "microbatch_size": microbatch_size,
        "gradient_accumulation": microbatch_size < effective_batch_size,
        "updates_per_epoch": updates_per_epoch,
        "total_parameter_updates": total_updates,
        "training_seed": args.seed,
        "optimizer_name": optimizer_name,
        "device": str(device),
        "peak_gpu_memory_GB": (
            float(torch.cuda.max_memory_allocated(device) / 1.0e9)
            if device.type == "cuda"
            else None
        ),
        "torch_threads": args.threads,
        "field_architecture": field_architecture,
        "physics_constraints_in_training": physics_constrained,
        "loss_terms": list(active_loss_names),
        "active_loss_groups": list(active_loss_groups),
        "loss_group_weights": loss_group_weights,
        "effective_loss_term_weights": effective_loss_term_weights,
        "loss_definition": (
            "5*state_data + 1*mean(face-flow data terms)"
            + (" + 1*mean(local mass/energy balance terms)" if physics_constrained else "")
        ),
        "loss_weight_source": (
            "third_party/physics_informed/configs/operator/"
            "Re500-05s-1000-PINO.yaml (xy_loss:ic_loss:f_loss = 5:1:1); "
            "numerical training weights, not pebble-bed physical parameters"
        ),
        "normalization": {
            "internal_mass_scale_kg_s": internal_scale,
            "boundary_mass_scale_kg_s": boundary_scale,
            "regional_incident_mass_scale_kg_s": balance_scale,
            "internal_energy_scale_W": energy_internal_scale,
            "boundary_energy_scale_W": energy_boundary_scale,
            "regional_incident_energy_scale_W": energy_balance_scale,
            "scales_use_training_cases_only": True,
        },
        "settings_from_archived_source": settings,
        "metric_contract": STEADY_METRIC_CONTRACT,
        "run_provenance": provenance,
        "evaluations": evaluations,
        "regional_prediction_files": prediction_files,
        "regional_prediction_file_records": prediction_file_records,
        "selected_model_record": file_record(best_path),
        "evidence_boundary": (
            "P418 multi-condition comparison with complete condition-level holdouts"
            if len(condition_ids) >= 20
            else "small-case software and physics pipeline; not enough conditions for model-accuracy claims"
        ),
        "new_physical_parameters": [],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
