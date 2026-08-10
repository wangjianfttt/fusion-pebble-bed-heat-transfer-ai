#!/usr/bin/env python3
"""Train the full-state P418 graph--Transformer on complete flow--heat steps."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from hccb_p418_fully_coupled_dataset import (
    load_index,
    load_sequence,
    selected_split,
    sequence_records,
    training_statistics,
)
from hccb_p418_loss_balancing import (
    LOSS_GROUP_NAMES,
    FixedLossBalancer,
    ReLoBRaLoLossBalancer,
    build_loss_balancer,
    common_validation_score,
    weighted_group_loss,
)
from hccb_p418_fully_coupled_spatiotemporal_operator import (
    FULLY_COUPLED_ARCHITECTURE_REVISION,
    HCCBP418FullyCoupledRegionalOperator,
    build_p418_fully_coupled_flux_graph,
)
from hccb_p418_fully_coupled_training import (
    PHYSICS_TERM_NAMES,
    combine_fully_coupled_loss_groups,
    projection_aware_physics_terms,
    supervised_fully_coupled_terms,
    training_equation_scales,
)
from hccb_p418_fully_coupled_transient_physics import (
    P418FullyCoupledEquationScales,
    P418FullyCoupledTransientResidual,
    assemble_p418_fully_coupled_transient_residual,
    dimensionless_fully_coupled_equation_terms,
)
from hccb_p418_regional_cht_adapter import load_p418_subface_geometry
from hccb_p418_spatiotemporal_regional_operator import (
    FORMAL_ARCHITECTURE,
    P418ThermalStepRegionalGraph,
)


FORMAL_NUMERICAL_SETTINGS = {
    "epochs": 500,
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-5,
}
LOSS_BALANCING_SOURCES = (
    Path(__file__).resolve().parents[1]
    / "parameters"
    / "hccb_p418_loss_balancing_sources.json"
)
FULLY_COUPLED_MODEL_PATH = Path(__file__).with_name(
    "hccb_p418_fully_coupled_spatiotemporal_operator.py"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_statistics_by_node(
    statistics: dict[str, object],
    node_type: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.as_tensor(statistics["state_mean"], device=device)
    std = torch.as_tensor(statistics["state_std"], device=device)
    return mean[node_type], std[node_type]


def detached_residual(
    residual: P418FullyCoupledTransientResidual,
) -> P418FullyCoupledTransientResidual:
    return P418FullyCoupledTransientResidual(
        **{field.name: getattr(residual, field.name).detach() for field in fields(residual)}
    )


def scale_record(scales: P418FullyCoupledEquationScales) -> dict[str, float]:
    return {
        field.name: float(getattr(scales, field.name).detach().cpu())
        for field in fields(scales)
    }


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def checkpoint_contract(
    *,
    args: argparse.Namespace,
    dataset_index: Path,
    split_path: Path,
    residual_geometry: Path,
    split: dict[str, list[str]],
) -> dict[str, object]:
    return {
        "dataset_index": str(dataset_index),
        "dataset_sha256": sha256(dataset_index),
        "split_file": str(split_path),
        "split_sha256": sha256(split_path),
        "residual_geometry": str(residual_geometry),
        "residual_geometry_sha256": sha256(residual_geometry),
        "split_name": args.split_name,
        "split_sequence_ids": split,
        "run_role": args.run_role,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "state_weight": args.state_weight,
        "face_flux_weight": args.face_flux_weight,
        "physics_weight": args.physics_weight,
        "loss_balance_candidate_id": args.loss_balance_candidate_id,
        "loss_balance_method": args.loss_balance_method,
        "relobralo_temperature": args.relobralo_temperature,
        "relobralo_alpha": args.relobralo_alpha,
        "relobralo_rho": args.relobralo_rho,
        "loss_balancing_sources_sha256": sha256(LOSS_BALANCING_SOURCES),
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
        "seed": args.seed,
        "architecture_revision": FULLY_COUPLED_ARCHITECTURE_REVISION,
        "model_implementation_sha256": sha256(FULLY_COUPLED_MODEL_PATH),
        "loss_definition": "full_state_face_flux_projection_aware_eight_equations_v1",
    }


def save_checkpoint(
    path: Path,
    *,
    contract: dict[str, object],
    next_epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_balancer: FixedLossBalancer | ReLoBRaLoLossBalancer,
    best_validation: float,
    best_epoch: int | None,
    best_state: dict[str, torch.Tensor] | None,
    best_loss_balancer_state: dict[str, object] | None,
    history: list[dict[str, float]],
    training_seconds: float,
) -> None:
    payload = {
        "contract": contract,
        "next_epoch": next_epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "loss_balancer_state": loss_balancer.state_dict(),
        "best_validation": best_validation,
        "best_epoch": best_epoch,
        "best_state": best_state,
        "best_loss_balancer_state": best_loss_balancer_state,
        "history": history,
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


def load_checkpoint(
    path: Path,
    *,
    contract: dict[str, object],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_balancer: FixedLossBalancer | ReLoBRaLoLossBalancer,
    device: torch.device,
) -> dict[str, object]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("contract") != contract:
        raise ValueError("checkpoint data, split, model or loss settings differ")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    loss_balancer.load_state_dict(payload["loss_balancer_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.random.set_rng_state(payload["torch_random_state"].cpu())
    cuda_state = payload.get("cuda_random_state_all")
    if device.type == "cuda" and cuda_state is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_state])
    return payload


def formal_configuration_check(args: argparse.Namespace) -> None:
    expected = {
        "epochs": FORMAL_NUMERICAL_SETTINGS["epochs"],
        "learning_rate": FORMAL_NUMERICAL_SETTINGS["learning_rate"],
        "weight_decay": FORMAL_NUMERICAL_SETTINGS["weight_decay"],
        "hidden_dim": FORMAL_ARCHITECTURE["hidden_dim"],
        "local_pre_iterations": FORMAL_ARCHITECTURE["local_pre_iterations"],
        "physics_attention_blocks": FORMAL_ARCHITECTURE["physics_attention_blocks"],
        "local_post_iterations": FORMAL_ARCHITECTURE["local_post_iterations"],
        "physics_attention_heads": FORMAL_ARCHITECTURE["physics_attention_heads"],
        "physics_slices": FORMAL_ARCHITECTURE["physics_slices"],
        "temporal_layers": FORMAL_ARCHITECTURE["temporal_layers"],
        "temporal_heads": FORMAL_ARCHITECTURE["temporal_heads"],
    }
    changed = {
        name: (getattr(args, name), value)
        for name, value in expected.items()
        if getattr(args, name) != value
    }
    if changed:
        raise ValueError(f"formal settings differ from the recorded model contract: {changed}")
    source = json.loads(LOSS_BALANCING_SOURCES.read_text(encoding="utf-8"))
    candidates = source.get("formal_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("loss-balancing source file has no formal candidates")
    requested = {
        "method": args.loss_balance_method,
        "state_weight": args.state_weight,
        "face_flux_weight": args.face_flux_weight,
        "physics_weight": args.physics_weight,
    }
    if args.loss_balance_method == "relobralo":
        requested.update(
            {
                "temperature": args.relobralo_temperature,
                "alpha": args.relobralo_alpha,
                "expected_rho": args.relobralo_rho,
            }
        )
    matching = [
        row
        for row in candidates
        if row.get("candidate_id") == args.loss_balance_candidate_id
        and all(row.get(name) == value for name, value in requested.items())
    ]
    if len(matching) != 1:
        raise ValueError(
            "formal loss-balancing settings must match exactly one recorded candidate"
        )
    if args.evaluation_stage == "final":
        if args.selected_method_record is None:
            raise ValueError(
                "formal final evaluation requires the validation-only method-selection record"
            )
        record_path = args.selected_method_record.resolve()
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "p418_loss_balancing_selected_on_validation_only":
            raise ValueError("loss-balancing selection record has an unexpected status")
        if record.get("independent_test_read") is not False:
            raise ValueError("loss-balancing selection must not read independent test curves")
        if record.get("source_file_sha256") != sha256(LOSS_BALANCING_SOURCES):
            raise ValueError("loss-balancing source settings changed after method selection")
        if record.get("selected_candidate_id") != args.loss_balance_candidate_id:
            raise ValueError("selected loss-balancing candidate differs from this run")
        selection_summary_path = Path(record["selected_summary_path"]).resolve()
        if sha256(selection_summary_path) != record.get("selected_summary_sha256"):
            raise ValueError("validation-only selection summary changed before final evaluation")
        selection_summary = json.loads(
            selection_summary_path.read_text(encoding="utf-8")
        )
        if (
            selection_summary.get("evaluation_stage") != "selection"
            or selection_summary.get("test_evaluated") is not False
            or "test" in selection_summary.get("metrics", {})
        ):
            raise ValueError("selected method summary is not validation-only")
        if (
            selection_summary.get("loss_balancing", {}).get("candidate_id")
            != args.loss_balance_candidate_id
        ):
            raise ValueError("selected summary and final candidate differ")
        expected_inputs = {
            "dataset_index": sha256(args.dataset_index.resolve()),
            "split_file": sha256(args.splits.resolve()),
            "residual_geometry": sha256(args.residual_geometry.resolve()),
            "loss_balancing_sources": sha256(LOSS_BALANCING_SOURCES),
        }
        if selection_summary.get("input_file_sha256") != expected_inputs:
            raise ValueError("data, split, geometry or method source changed after selection")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--residual-geometry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-role", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--epochs", type=int, default=FORMAL_NUMERICAL_SETTINGS["epochs"])
    parser.add_argument(
        "--learning-rate", type=float, default=FORMAL_NUMERICAL_SETTINGS["learning_rate"]
    )
    parser.add_argument(
        "--weight-decay", type=float, default=FORMAL_NUMERICAL_SETTINGS["weight_decay"]
    )
    parser.add_argument("--state-weight", type=float, required=True)
    parser.add_argument("--face-flux-weight", type=float, required=True)
    parser.add_argument("--physics-weight", type=float, required=True)
    parser.add_argument(
        "--loss-balance-method",
        choices=("fixed", "relobralo"),
        default="fixed",
    )
    parser.add_argument("--loss-balance-candidate-id")
    parser.add_argument("--relobralo-temperature", type=float)
    parser.add_argument("--relobralo-alpha", type=float)
    parser.add_argument("--relobralo-rho", type=float)
    parser.add_argument(
        "--evaluation-stage",
        choices=("selection", "final"),
        default="final",
    )
    parser.add_argument("--selected-method-record", type=Path)
    parser.add_argument("--hidden-dim", type=int, default=FORMAL_ARCHITECTURE["hidden_dim"])
    parser.add_argument(
        "--local-pre-iterations",
        type=int,
        default=FORMAL_ARCHITECTURE["local_pre_iterations"],
    )
    parser.add_argument(
        "--physics-attention-blocks",
        type=int,
        default=FORMAL_ARCHITECTURE["physics_attention_blocks"],
    )
    parser.add_argument(
        "--local-post-iterations",
        type=int,
        default=FORMAL_ARCHITECTURE["local_post_iterations"],
    )
    parser.add_argument(
        "--physics-attention-heads",
        type=int,
        default=FORMAL_ARCHITECTURE["physics_attention_heads"],
    )
    parser.add_argument(
        "--physics-slices", type=int, default=FORMAL_ARCHITECTURE["physics_slices"]
    )
    parser.add_argument(
        "--temporal-layers", type=int, default=FORMAL_ARCHITECTURE["temporal_layers"]
    )
    parser.add_argument(
        "--temporal-heads", type=int, default=FORMAL_ARCHITECTURE["temporal_heads"]
    )
    parser.add_argument("--spatial-time-chunk-size", type=int, default=1)
    parser.add_argument("--temporal-node-chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    args = parser.parse_args()

    if args.run_role == "formal":
        formal_configuration_check(args)
    if args.epochs <= 0 or args.checkpoint_every <= 0:
        raise ValueError("epochs and checkpoint interval must be positive")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (args.state_weight, args.face_flux_weight, args.physics_weight)
    ):
        raise ValueError("all three loss-group weights must be finite and positive")
    if args.torch_threads is not None:
        if args.torch_threads <= 0:
            raise ValueError("torch thread count must be positive")
        torch.set_num_threads(args.torch_threads)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    dataset_index = args.dataset_index.resolve()
    split_path = args.splits.resolve()
    residual_geometry_path = args.residual_geometry.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root = dataset_index.parent
    index = load_index(dataset_index)
    records = sequence_records(index)
    if args.run_role == "formal" and (
        len(records) != 12 or not all(bool(row.get("complete")) for row in records.values())
    ):
        raise ValueError("formal training requires all 12 complete fully coupled curves")
    split = selected_split(set(records), split_path, args.split_name)
    graph_path = root / str(index["regional_geometry_file"])
    graph = P418ThermalStepRegionalGraph.from_npz(graph_path, device=device)
    geometry = load_p418_subface_geometry(
        residual_geometry_path,
        fluid_patch_names=index["boundary_patch_names"]["fluid"],
        solid_patch_names=index["boundary_patch_names"]["solid"],
        device=device,
        dtype=torch.float32,
    )
    with np.load(graph_path, allow_pickle=False) as regional_geometry:
        if not np.array_equal(
            regional_geometry["fluid_global_region"], geometry.fluid_global_region
        ) or not np.array_equal(
            regional_geometry["solid_global_region"], geometry.solid_global_region
        ):
            raise ValueError("regional graph and conservative geometry use different nodes")
    flux_graph = build_p418_fully_coupled_flux_graph(geometry=geometry, graph=graph)
    statistics = training_statistics(
        root,
        records,
        split["train"],
        graph.node_type.detach().cpu().numpy(),
    )
    condition_mean = torch.as_tensor(statistics["condition_mean"], device=device)
    condition_std = torch.as_tensor(statistics["condition_std"], device=device)
    state_mean, state_std = state_statistics_by_node(statistics, graph.node_type, device)
    internal_mean = torch.as_tensor(
        statistics["internal_mass_flux_mean_kg_s"], device=device
    )
    internal_std = torch.as_tensor(
        statistics["internal_mass_flux_std_kg_s"], device=device
    )
    boundary_mean = torch.as_tensor(
        statistics["boundary_mass_flux_mean_kg_s"], device=device
    )
    boundary_std = torch.as_tensor(
        statistics["boundary_mass_flux_std_kg_s"], device=device
    )

    curve_cache: dict[str, dict[str, torch.Tensor]] = {}

    def curve(sequence_id: str) -> dict[str, torch.Tensor]:
        if sequence_id not in curve_cache:
            times_np, condition_np, state_np, internal_np, boundary_np = load_sequence(
                root, records[sequence_id]
            )
            times = torch.as_tensor(times_np, device=device)
            condition = torch.as_tensor(condition_np, device=device).unsqueeze(0)
            state = torch.as_tensor(state_np, device=device).unsqueeze(0)
            internal = torch.as_tensor(internal_np, device=device).unsqueeze(0)
            boundary = torch.as_tensor(boundary_np, device=device).unsqueeze(0)
            if internal.shape[-1] != flux_graph.internal_face_count:
                raise ValueError(f"internal face count differs in {sequence_id}")
            if boundary.shape[-1] != flux_graph.boundary_face_count:
                raise ValueError(f"boundary face count differs in {sequence_id}")
            curve_cache[sequence_id] = {
                "time": times,
                "condition": condition,
                "condition_normalized": (condition - condition_mean) / condition_std,
                "time_normalized": times / float(statistics["maximum_time_s"]),
                "state": state,
                "state_normalized": (state - state_mean[None, None])
                / state_std[None, None],
                "internal": internal,
                "internal_normalized": (internal - internal_mean) / internal_std,
                "boundary": boundary,
                "boundary_normalized": (boundary - boundary_mean) / boundary_std,
            }
        return curve_cache[sequence_id]

    reference_cache: dict[str, P418FullyCoupledTransientResidual] = {}

    def reference_residual(sequence_id: str) -> P418FullyCoupledTransientResidual:
        if sequence_id not in reference_cache:
            values = curve(sequence_id)
            with torch.no_grad():
                reference_cache[sequence_id] = detached_residual(
                    assemble_p418_fully_coupled_transient_residual(
                        geometry=geometry,
                        step_condition=values["condition"],
                        state_physical=values["state"],
                        time_s=values["time"],
                        fluid_internal_mass_flux_kg_s=values["internal"],
                        fluid_boundary_mass_flux_kg_s=values["boundary"],
                    )
                )
        return reference_cache[sequence_id]

    equation_scales = training_equation_scales(
        [reference_residual(sequence_id) for sequence_id in split["train"]],
        [curve(sequence_id)["state"] for sequence_id in split["train"]],
    )
    model = HCCBP418FullyCoupledRegionalOperator(
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
        internal_face_feature_dim=flux_graph.internal_features.shape[1],
        boundary_face_feature_dim=flux_graph.boundary_features.shape[1],
        boundary_role_count=graph.boundary_role_count,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0.0
    )
    loss_balancer = build_loss_balancer(
        method=args.loss_balance_method,
        state_weight=args.state_weight,
        face_flux_weight=args.face_flux_weight,
        physics_weight=args.physics_weight,
        relobralo_temperature=args.relobralo_temperature,
        relobralo_alpha=args.relobralo_alpha,
        relobralo_rho=args.relobralo_rho,
        seed=args.seed,
    )

    def prediction(sequence_id: str):
        values = curve(sequence_id)
        normalized = model(
            values["state_normalized"][:, 0],
            values["internal_normalized"][:, 0],
            values["boundary_normalized"][:, 0],
            values["condition_normalized"],
            values["time_normalized"],
            graph,
            flux_graph,
        )
        return normalized, (
            normalized.state * state_std[None, None] + state_mean[None, None],
            normalized.internal_mass_flux * internal_std + internal_mean,
            normalized.boundary_mass_flux * boundary_std + boundary_mean,
        )

    def loss_terms(sequence_id: str):
        values = curve(sequence_id)
        normalized, physical = prediction(sequence_id)
        predicted_state, predicted_internal, predicted_boundary = physical
        predicted_residual = assemble_p418_fully_coupled_transient_residual(
            geometry=geometry,
            step_condition=values["condition"],
            state_physical=predicted_state,
            time_s=values["time"],
            fluid_internal_mass_flux_kg_s=predicted_internal,
            fluid_boundary_mass_flux_kg_s=predicted_boundary,
        )
        supervised = supervised_fully_coupled_terms(
            predicted_state=predicted_state,
            reference_state=values["state"],
            predicted_internal_mass_flux=predicted_internal,
            reference_internal_mass_flux=values["internal"],
            predicted_boundary_mass_flux=predicted_boundary,
            reference_boundary_mass_flux=values["boundary"],
            node_type=graph.node_type,
            state_scale_by_node=state_std,
            internal_mass_flux_scale_kg_s=internal_std,
            boundary_mass_flux_scale_kg_s=boundary_std,
        )
        physics = projection_aware_physics_terms(
            prediction=predicted_residual,
            reference=reference_residual(sequence_id),
            scales=equation_scales,
            fluid_volume_m3=geometry.fluid_mesh.cell_volume,
            solid_volume_m3=geometry.solid_mesh.cell_volume,
        )
        _, groups = combine_fully_coupled_loss_groups(
            supervised_terms=supervised,
            physics_terms=physics,
            state_weight=1.0,
            face_flux_weight=1.0,
            physics_weight=1.0,
        )
        absolute = dimensionless_fully_coupled_equation_terms(
            residual=predicted_residual,
            scales=equation_scales,
            fluid_volume_m3=geometry.fluid_mesh.cell_volume,
            solid_volume_m3=geometry.solid_mesh.cell_volume,
        )
        return supervised, physics, groups, absolute, normalized, physical

    def evaluate(role: str) -> dict[str, float]:
        model.eval()
        accumulated: dict[str, float] = {}
        count = 0
        with torch.no_grad():
            for sequence_id in split[role]:
                supervised, physics, groups, absolute, _, physical = loss_terms(sequence_id)
                sample = next(iter(groups.values()))
                current_weights = loss_balancer.weights(
                    device=sample.device, dtype=sample.dtype
                )
                total = weighted_group_loss(groups, current_weights)
                selection_score = common_validation_score(groups)
                predicted_state, predicted_internal, predicted_boundary = physical
                values = curve(sequence_id)
                error = predicted_state - values["state"]
                fluid = graph.node_type == 0
                solid = graph.node_type == 1
                items = {
                    "combined_loss": total,
                    "selection_score": selection_score,
                    **{f"supervised_{name}": value for name, value in supervised.items()},
                    **{f"physics_difference_{name}": value for name, value in physics.items()},
                    **{f"loss_group_{name}": value for name, value in groups.items()},
                    **{f"absolute_equation_{name}": value for name, value in absolute.items()},
                    "fluid_velocity_RMSE_m_s": torch.sqrt(
                        error[:, :, fluid, :3].square().mean()
                    ),
                    "fluid_pressure_RMSE_Pa": torch.sqrt(
                        error[:, :, fluid, 3].square().mean()
                    ),
                    "fluid_temperature_RMSE_K": torch.sqrt(
                        error[:, :, fluid, 4].square().mean()
                    ),
                    "solid_temperature_RMSE_K": torch.sqrt(
                        error[:, :, solid, 4].square().mean()
                    ),
                    "internal_mass_flux_RMSE_kg_s": torch.sqrt(
                        (predicted_internal - values["internal"]).square().mean()
                    ),
                    "boundary_mass_flux_RMSE_kg_s": torch.sqrt(
                        (predicted_boundary - values["boundary"]).square().mean()
                    ),
                }
                for name, value in items.items():
                    accumulated[name] = accumulated.get(name, 0.0) + float(
                        value.detach().cpu()
                    )
                count += 1
        if count == 0:
            raise ValueError(f"{role} split is empty")
        return {name: value / count for name, value in accumulated.items()}

    contract = checkpoint_contract(
        args=args,
        dataset_index=dataset_index,
        split_path=split_path,
        residual_geometry=residual_geometry_path,
        split=split,
    )
    checkpoint_path = output_dir / "training_checkpoint.pt"
    history: list[dict[str, float]] = []
    best_validation = math.inf
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_loss_balancer_state: dict[str, object] | None = None
    start_epoch = 0
    previous_seconds = 0.0
    if args.resume and checkpoint_path.is_file():
        resumed = load_checkpoint(
            checkpoint_path,
            contract=contract,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_balancer=loss_balancer,
            device=device,
        )
        start_epoch = int(resumed["next_epoch"])
        history = list(resumed["history"])
        best_validation = float(resumed["best_validation"])
        best_epoch = resumed["best_epoch"]
        best_state = resumed["best_state"]
        best_loss_balancer_state = resumed["best_loss_balancer_state"]
        previous_seconds = float(resumed["training_seconds"])
        if len(history) != start_epoch or start_epoch > args.epochs:
            raise ValueError("checkpoint completed-epoch count is invalid")

    start = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_values = []
        for sequence_id in np.random.permutation(split["train"]):
            _, _, groups, _, _, _ = loss_terms(str(sequence_id))
            effective_weights = loss_balancer.update(groups)
            total = weighted_group_loss(groups, effective_weights)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if not all(
                parameter.grad is None or torch.all(torch.isfinite(parameter.grad))
                for parameter in model.parameters()
            ):
                raise FloatingPointError("non-finite full-state model gradient")
            optimizer.step()
            epoch_values.append(
                [
                    float(total.detach().cpu()),
                    float(groups["state_data"].detach().cpu()),
                    float(groups["face_flux_data"].detach().cpu()),
                    float(groups["physics"].detach().cpu()),
                    *[
                        float(effective_weights[name].detach().cpu())
                        for name in LOSS_GROUP_NAMES
                    ],
                ]
            )
        scheduler.step()
        validation = evaluate("validation")
        validation_score = validation["selection_score"]
        if validation_score < best_validation:
            best_validation = validation_score
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_loss_balancer_state = copy.deepcopy(loss_balancer.state_dict())
        means = np.mean(epoch_values, axis=0)
        history.append(
            {
                "epoch": epoch + 1,
                "combined_loss": float(means[0]),
                "state_data_loss": float(means[1]),
                "face_flux_data_loss": float(means[2]),
                "physics_loss": float(means[3]),
                "state_data_weight": float(means[4]),
                "face_flux_data_weight": float(means[5]),
                "physics_weight": float(means[6]),
                "validation_selection_score": validation_score,
                "validation_training_weighted_loss": validation["combined_loss"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if (epoch + 1) % args.checkpoint_every == 0 or epoch + 1 == args.epochs:
            save_checkpoint(
                checkpoint_path,
                contract=contract,
                next_epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_balancer=loss_balancer,
                best_validation=best_validation,
                best_epoch=best_epoch,
                best_state=best_state,
                best_loss_balancer_state=best_loss_balancer_state,
                history=history,
                training_seconds=previous_seconds + time.perf_counter() - start,
            )
    training_seconds = previous_seconds + time.perf_counter() - start
    if (
        best_state is None
        or best_epoch is None
        or best_loss_balancer_state is None
    ):
        raise RuntimeError("training did not produce a selected model")
    model.load_state_dict(best_state)
    loss_balancer.load_state_dict(best_loss_balancer_state)

    evaluation_roles = (
        ("train", "validation")
        if args.evaluation_stage == "selection"
        else ("train", "validation", "test")
    )
    metrics = {role: evaluate(role) for role in evaluation_roles}
    prediction_files: dict[str, str] = {}
    model.eval()
    with torch.no_grad():
        for role in evaluation_roles:
            sequence_ids = split[role]
            paths = []
            for sequence_id in sequence_ids:
                _, physical = prediction(sequence_id)
                predicted_state, predicted_internal, predicted_boundary = physical
                values = curve(sequence_id)
                path = output_dir / f"{role}_{sequence_id}_prediction.npz"
                np.savez_compressed(
                    path,
                    sequence_id=np.asarray(sequence_id),
                    time_s=values["time"].cpu().numpy(),
                    condition_physical=values["condition"][0].cpu().numpy(),
                    state_prediction=predicted_state[0].cpu().numpy(),
                    state_target=values["state"][0].cpu().numpy(),
                    internal_mass_flux_prediction=predicted_internal[0].cpu().numpy(),
                    internal_mass_flux_target=values["internal"][0].cpu().numpy(),
                    boundary_mass_flux_prediction=predicted_boundary[0].cpu().numpy(),
                    boundary_mass_flux_target=values["boundary"][0].cpu().numpy(),
                )
                paths.append(path.name)
            prediction_files[role] = paths

    model_path = output_dir / "model_state.pt"
    torch.save(best_state, model_path)
    with (output_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    np.savez_compressed(
        output_dir / "training_statistics.npz",
        condition_mean=statistics["condition_mean"],
        condition_std=statistics["condition_std"],
        state_mean=statistics["state_mean"],
        state_std=statistics["state_std"],
        internal_mass_flux_mean_kg_s=np.asarray(
            statistics["internal_mass_flux_mean_kg_s"]
        ),
        internal_mass_flux_std_kg_s=np.asarray(
            statistics["internal_mass_flux_std_kg_s"]
        ),
        boundary_mass_flux_mean_kg_s=np.asarray(
            statistics["boundary_mass_flux_mean_kg_s"]
        ),
        boundary_mass_flux_std_kg_s=np.asarray(
            statistics["boundary_mass_flux_std_kg_s"]
        ),
        maximum_time_s=np.asarray(statistics["maximum_time_s"]),
        training_sequence_ids=np.asarray(split["train"]),
    )
    loss_balancer_summary = {
        name: value
        for name, value in loss_balancer.state_dict().items()
        if name != "random_state"
    }
    summary = {
        "status": "completed_p418_fully_coupled_spatiotemporal_operator",
        "run_role": args.run_role,
        "dataset_index": str(dataset_index),
        "input_file_sha256": {
            "dataset_index": sha256(dataset_index),
            "split_file": sha256(split_path),
            "residual_geometry": sha256(residual_geometry_path),
            "loss_balancing_sources": sha256(LOSS_BALANCING_SOURCES),
        },
        "split_name": args.split_name,
        "split_sequence_ids": split,
        "selection_split": "validation",
        "evaluation_stage": args.evaluation_stage,
        "test_evaluated": args.evaluation_stage == "final",
        "test_used_after_model_selection_only": (
            args.evaluation_stage == "final"
            and args.selected_method_record is not None
        ),
        "selected_method_record": (
            str(args.selected_method_record.resolve())
            if args.selected_method_record is not None
            else None
        ),
        "selected_method_record_sha256": (
            sha256(args.selected_method_record.resolve())
            if args.selected_method_record is not None
            else None
        ),
        "seed": args.seed,
        "architecture": {
            "revision": FULLY_COUPLED_ARCHITECTURE_REVISION,
            "model_implementation_sha256": sha256(FULLY_COUPLED_MODEL_PATH),
            "hidden_dim": args.hidden_dim,
            "local_pre_iterations": args.local_pre_iterations,
            "physics_attention_blocks": args.physics_attention_blocks,
            "local_post_iterations": args.local_post_iterations,
            "physics_attention_heads": args.physics_attention_heads,
            "physics_slices": args.physics_slices,
            "temporal_layers": args.temporal_layers,
            "temporal_heads": args.temporal_heads,
        },
        "loss_balancing": {
            "candidate_id": args.loss_balance_candidate_id,
            "method": args.loss_balance_method,
            "declared_initial_or_fixed_weights": {
                "state_data": args.state_weight,
                "face_flux_data": args.face_flux_weight,
                "physics": args.physics_weight,
            },
            "relobralo_temperature": args.relobralo_temperature,
            "relobralo_alpha": args.relobralo_alpha,
            "relobralo_expected_rho": args.relobralo_rho,
            "selected_checkpoint_state": loss_balancer_summary,
            "source_file": str(LOSS_BALANCING_SOURCES),
            "source_file_sha256": sha256(LOSS_BALANCING_SOURCES),
            "common_validation_score": (
                "equal mean of dimensionless state, face-flux and physics groups"
            ),
        },
        "physics_terms": list(PHYSICS_TERM_NAMES),
        "equation_scales_from_training_curves": scale_record(equation_scales),
        "training_normalization_sequence_ids": statistics["training_sequence_ids"],
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_validation,
        "metrics": metrics,
        "prediction_files": prediction_files,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_state_sha256": sha256(model_path),
        "training_seconds": training_seconds,
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "formal_openfoam_data": args.run_role == "formal",
        "new_physical_parameters": [],
    }
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    stage_summary_path = output_dir / f"{args.evaluation_stage}_summary.json"
    stage_summary_path.write_text(summary_text, encoding="utf-8")
    (output_dir / "summary.json").write_text(summary_text, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
