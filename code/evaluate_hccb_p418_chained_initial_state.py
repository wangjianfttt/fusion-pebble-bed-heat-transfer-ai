#!/usr/bin/env python3
"""Evaluate a transient graph--Transformer with a steady-PINN initial field."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from hccb_p418_comparison_contract import file_record, sha256_file
from hccb_p418_spatiotemporal_regional_operator import (
    HCCBP418SpatiotemporalRegionalOperator,
    P418ThermalStepRegionalGraph,
    temperature_output_bounds_by_node_type,
)
from hccb_p418_chain_roles import (
    DETERMINISTIC_CHAIN_STATUS,
    endpoint_novelty_class,
    steady_condition_roles,
    summarize_endpoint_groups,
)
from train_hccb_p418_conservative_mixed_operator import physical_state
from train_hccb_p418_regional_operator import load_scales


def verify_registered_file(
    path: Path, record: dict[str, object] | None, label: str
) -> None:
    """Require a completed output to match the size and SHA-256 saved by its producer."""
    if not isinstance(record, dict):
        raise ValueError(f"{label} has no saved file record")
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    expected_size = int(record.get("size_bytes", -1))
    expected_sha = str(record.get("sha256", ""))
    if path.stat().st_size != expected_size:
        raise ValueError(f"{label} size differs from the completed training result")
    if sha256_file(path) != expected_sha:
        raise ValueError(f"{label} SHA-256 differs from the completed training result")


def compose_chained_initial_state(
    source_steady_state: np.ndarray,
    target_steady_state: np.ndarray,
    node_type: np.ndarray,
) -> np.ndarray:
    """Use source temperature and target hydrodynamics, matching the step model."""
    if source_steady_state.shape != target_steady_state.shape:
        raise ValueError("source and target steady states have different shapes")
    if source_steady_state.ndim != 2 or source_steady_state.shape[1] != 5:
        raise ValueError("steady states must have shape [node,5]")
    if node_type.shape != source_steady_state.shape[:1]:
        raise ValueError("node types do not match the steady states")
    output = target_steady_state.copy()
    output[:, 4] = source_steady_state[:, 4]
    output[node_type == 1, :4] = 0.0
    if not np.all(np.isfinite(output)):
        raise ValueError("chained initial state contains non-finite values")
    return output


def volume_weighted_temperature_rmse(
    prediction: np.ndarray,
    target: np.ndarray,
    node_type: np.ndarray,
    node_volume_m3: np.ndarray,
    material: int,
) -> float:
    """Return a time-and-volume weighted temperature RMSE in kelvin."""
    selected = node_type == material
    if not np.any(selected):
        raise ValueError(f"material {material} has no regional nodes")
    error = prediction[:, selected] - target[:, selected]
    weights = node_volume_m3[selected]
    mse = np.sum(np.square(error) * weights[None, :]) / (
        error.shape[0] * np.sum(weights)
    )
    return float(math.sqrt(mse))


def chained_prediction_artifact(
    upstream: dict[str, np.ndarray],
    sequence_ids: list[str],
    chained_histories: list[np.ndarray],
    target_internal_mass_fluxes: list[np.ndarray],
    target_boundary_mass_fluxes: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Replace deterministic temperature and target mass flux for chained evaluation."""
    if "sequence_id" not in upstream or "baseline_temperature_normalized" not in upstream:
        raise ValueError("upstream deterministic artifact lacks sequence or temperature data")
    if [str(value) for value in upstream["sequence_id"]] != sequence_ids:
        raise ValueError("upstream prediction order differs from chained evaluation")
    chained = np.stack(chained_histories).astype(np.float32)
    if chained.shape != upstream["baseline_temperature_normalized"].shape:
        raise ValueError("chained temperature histories differ from upstream shape")
    internal = np.stack(target_internal_mass_fluxes).astype(np.float32)
    boundary = np.stack(target_boundary_mass_fluxes).astype(np.float32)
    for name, values in (
        ("fluid_internal_mass_flux_kg_s", internal),
        ("fluid_boundary_mass_flux_kg_s", boundary),
    ):
        if name not in upstream or values.shape != upstream[name].shape:
            raise ValueError(f"chained {name} differs from upstream shape")
    output = {name: value.copy() for name, value in upstream.items()}
    output["exact_initial_baseline_temperature_normalized"] = output[
        "baseline_temperature_normalized"
    ].copy()
    output["exact_target_fluid_internal_mass_flux_kg_s"] = output[
        "fluid_internal_mass_flux_kg_s"
    ].copy()
    output["exact_target_fluid_boundary_mass_flux_kg_s"] = output[
        "fluid_boundary_mass_flux_kg_s"
    ].copy()
    output["baseline_temperature_normalized"] = chained
    output["fluid_internal_mass_flux_kg_s"] = internal
    output["fluid_boundary_mass_flux_kg_s"] = boundary
    return output


def load_steady_prediction_map(
    *,
    summary_path: Path,
    state_targets_path: Path,
    training_statistics_path: Path,
    split_name: str,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, tuple[np.ndarray, np.ndarray]],
    np.ndarray,
    np.ndarray,
    dict[str, str],
]:
    """Load all steady-PINN predictions and convert them to physical units."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("architecture") != "pinn":
        raise ValueError("chained evaluation requires the physics-constrained steady PINN")
    if summary.get("split_name") != split_name:
        raise ValueError("steady PINN summary uses a different condition split")
    role_by_condition = steady_condition_roles(summary)
    files = summary.get("regional_prediction_files")
    if not isinstance(files, dict) or set(files) != {"train", "validation", "test"}:
        raise ValueError("steady PINN summary does not provide all prediction files")
    records = summary.get("regional_prediction_file_records")
    if not isinstance(records, dict) or set(records) != set(files):
        raise ValueError("steady PINN summary does not record all prediction files")

    with np.load(state_targets_path, allow_pickle=False) as loaded:
        condition_ids = loaded["condition_id"].astype(str)
        conditions = loaded["condition_physical"].astype(np.float64)
        node_type = loaded["node_type"].astype(np.int64)
        node_volume = loaded["node_volume_m3"].astype(np.float64)
    condition_by_id = {
        identifier: conditions[index]
        for index, identifier in enumerate(condition_ids)
    }
    scales = load_scales(training_statistics_path, split_name)
    predictions: dict[str, np.ndarray] = {}
    mass_flux_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for role in ("train", "validation", "test"):
        path = summary_path.parent / str(files[role])
        verify_registered_file(path, records.get(role), f"steady {role} prediction")
        with np.load(path, allow_pickle=False) as loaded:
            identifiers = loaded["condition_id"].astype(str)
            normalized = loaded["baseline_state_normalized"].astype(np.float64)
            internal_mass = loaded["internal_mass_flow_kg_s"].astype(np.float64)
            boundary_mass = loaded["boundary_mass_flow_kg_s"].astype(np.float64)
            if len(internal_mass) != len(identifiers) or len(boundary_mass) != len(
                identifiers
            ):
                raise ValueError("steady mass-flow prediction count differs from conditions")
            if not np.array_equal(loaded["node_type"].astype(np.int64), node_type):
                raise ValueError("steady prediction node order differs from state targets")
            if not np.allclose(
                loaded["node_volume_m3"].astype(np.float64), node_volume
            ):
                raise ValueError("steady prediction volumes differ from state targets")
        for identifier, values in zip(identifiers, normalized):
            if identifier in predictions:
                raise ValueError(f"steady prediction {identifier} is repeated")
            predictions[identifier] = physical_state(
                values, condition_by_id[identifier], node_type, scales
            )
        for identifier, internal, boundary in zip(
            identifiers, internal_mass, boundary_mass
        ):
            if identifier in mass_flux_predictions:
                raise ValueError(f"steady mass-flow prediction {identifier} is repeated")
            mass_flux_predictions[identifier] = (internal, boundary)
    if set(predictions) != set(condition_ids):
        raise ValueError("steady PINN predictions do not cover all P418 conditions")
    if set(role_by_condition) != set(condition_ids):
        raise ValueError("steady condition roles do not cover all P418 conditions")
    if set(mass_flux_predictions) != set(condition_ids):
        raise ValueError("steady mass-flow predictions do not cover all P418 conditions")
    return predictions, mass_flux_predictions, node_type, node_volume, role_by_condition


def registered_steady_endpoint_inference_time(
    *,
    steady_summary: dict[str, object],
    role_by_condition: dict[str, str],
    condition_ids: list[str],
) -> dict[str, object]:
    """Sum measured steady-PINN inference time over unique required endpoints."""
    evaluations = steady_summary.get("evaluations")
    if not isinstance(evaluations, dict):
        raise ValueError("steady PINN summary lacks measured evaluation timing")
    unique_ids = sorted(set(condition_ids))
    role_counts = {role: 0 for role in ("train", "validation", "test")}
    role_seconds_per_case: dict[str, float] = {}
    total_seconds = 0.0
    for condition_id in unique_ids:
        if condition_id not in role_by_condition:
            raise ValueError(f"steady role is missing for endpoint {condition_id}")
        role = role_by_condition[condition_id]
        record = evaluations.get(role)
        if not isinstance(record, dict):
            raise ValueError(f"steady PINN evaluation timing is missing for {role}")
        seconds = float(record.get("inference_seconds_per_case", math.nan))
        if not math.isfinite(seconds) or seconds <= 0.0:
            raise ValueError(f"invalid steady PINN inference time for {role}: {seconds}")
        role_counts[role] += 1
        role_seconds_per_case[role] = seconds
        total_seconds += seconds
    return {
        "unique_endpoint_count": len(unique_ids),
        "unique_endpoint_condition_ids": unique_ids,
        "endpoint_count_by_steady_role": role_counts,
        "measured_inference_seconds_per_case_by_steady_role": role_seconds_per_case,
        "measured_unique_endpoint_inference_seconds": total_seconds,
    }


def deterministic_chain_model_cost(
    *,
    steady_summary: dict[str, object],
    transient_summary: dict[str, object],
) -> dict[str, float | int]:
    """Return measured training cost and parameter count for the two-stage chain."""
    values: dict[str, float | int] = {}
    for prefix, summary in (
        ("steady_PINN", steady_summary),
        ("graph_transformer", transient_summary),
    ):
        parameters = int(summary.get("model_parameter_count", 0))
        training = float(summary.get("training_seconds", math.nan))
        if parameters <= 0:
            raise ValueError(f"{prefix} parameter count is missing")
        if not math.isfinite(training) or training <= 0.0:
            raise ValueError(f"{prefix} measured training time is missing")
        values[f"{prefix}_model_parameter_count"] = parameters
        values[f"{prefix}_training_seconds"] = training
    values["deterministic_chain_model_parameter_count"] = int(
        values["steady_PINN_model_parameter_count"]
    ) + int(values["graph_transformer_model_parameter_count"])
    values["deterministic_chain_training_seconds"] = float(
        values["steady_PINN_training_seconds"]
    ) + float(values["graph_transformer_training_seconds"])
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transient-summary", type=Path, required=True)
    parser.add_argument("--transient-dataset-index", type=Path, required=True)
    parser.add_argument("--steady-summary", type=Path, required=True)
    parser.add_argument("--steady-state-targets", type=Path, required=True)
    parser.add_argument("--steady-training-statistics", type=Path, required=True)
    parser.add_argument("--steady-split-name", default="interleaved_all_ranges")
    parser.add_argument("--role", choices=("validation", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    transient_summary_path = args.transient_summary.resolve()
    transient_summary = json.loads(
        transient_summary_path.read_text(encoding="utf-8")
    )
    if transient_summary.get("status") != "completed_p418_spatiotemporal_regional_operator":
        raise ValueError("transient graph--Transformer is not complete")
    if transient_summary.get("physics_mode") != "energy_and_flux":
        raise ValueError("chained evaluation requires the physics-constrained graph--Transformer branch")
    dataset_path = args.transient_dataset_index.resolve()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    records = {str(row["sequence_id"]): row for row in dataset["sequences"]}
    sequence_ids = [
        str(value)
        for value in transient_summary["split_case_ids"][args.role]
    ]
    if any(identifier not in records for identifier in sequence_ids):
        raise ValueError("transient summary contains sequences absent from the dataset")

    steady_summary_path = args.steady_summary.resolve()
    steady_summary = json.loads(steady_summary_path.read_text(encoding="utf-8"))
    (
        steady,
        steady_mass_flux,
        steady_node_type,
        steady_volume,
        steady_roles,
    ) = load_steady_prediction_map(
        summary_path=steady_summary_path,
        state_targets_path=args.steady_state_targets.resolve(),
        training_statistics_path=args.steady_training_statistics.resolve(),
        split_name=args.steady_split_name,
    )
    graph = P418ThermalStepRegionalGraph.from_npz(
        dataset_path.parent / str(dataset["regional_geometry_file"]),
        device=args.device,
    )
    graph_node_type = graph.node_type.cpu().numpy()
    graph_volume = graph.volume_m3.cpu().numpy()
    if not np.array_equal(graph_node_type, steady_node_type):
        raise ValueError("steady and transient regional node orders differ")
    if not np.allclose(graph_volume, steady_volume):
        raise ValueError("steady and transient regional node volumes differ")

    training_statistics_path = transient_summary_path.parent / "training_statistics.npz"
    verify_registered_file(
        training_statistics_path,
        transient_summary.get("training_statistics_record"),
        "transient training statistics",
    )
    with np.load(training_statistics_path, allow_pickle=False) as loaded:
        condition_mean = loaded["condition_mean"].astype(np.float32)
        condition_std = loaded["condition_std"].astype(np.float32)
        state_mean_by_type = loaded["state_mean"].astype(np.float32)
        state_std_by_type = loaded["state_std"].astype(np.float32)
        maximum_time_s = float(loaded["maximum_time_s"])

    architecture = transient_summary["architecture"]
    temperature_output_mode = str(
        architecture.get("temperature_output_mode", "additive_normalized")
    )
    temperature_output_arguments: dict[str, object] = {
        "temperature_output_mode": temperature_output_mode
    }
    if temperature_output_mode in {
        "literature_bounded_logit",
        "literature_bounded_residual",
    }:
        temperature_output_arguments.update(
            {
                "temperature_mean_k_by_node_type": torch.as_tensor(
                    state_mean_by_type[:, 4], device=args.device
                ),
                "temperature_std_k_by_node_type": torch.as_tensor(
                    state_std_by_type[:, 4], device=args.device
                ),
                "temperature_bounds_k_by_node_type": torch.as_tensor(
                    temperature_output_bounds_by_node_type(
                        architecture["temperature_output_bounds_K_by_node_type"]
                    ),
                    device=args.device,
                    dtype=torch.float32,
                ),
            }
        )
    model = HCCBP418SpatiotemporalRegionalOperator(
        condition_dim=len(dataset["condition_names"]),
        hidden_dim=int(architecture["hidden_dim"]),
        local_pre_iterations=int(architecture["local_pre_iterations"]),
        physics_attention_blocks=int(architecture["physics_attention_blocks"]),
        local_post_iterations=int(architecture["local_post_iterations"]),
        physics_attention_heads=int(architecture["physics_attention_heads"]),
        physics_slices=int(architecture["physics_slices"]),
        temporal_layers=int(architecture["temporal_layers"]),
        temporal_heads=int(architecture["temporal_heads"]),
        spatial_time_chunk_size=int(architecture["spatial_time_chunk_size"]),
        temporal_node_chunk_size=architecture["temporal_node_chunk_size"],
        spatial_temporal_mode=str(architecture["spatial_temporal_mode"]),
        boundary_role_count=graph.boundary_role_count,
        **temperature_output_arguments,
    ).to(args.device)
    state_path = transient_summary_path.parent / "model_state.pt"
    verify_registered_file(
        state_path,
        transient_summary.get("model_state_record"),
        "transient graph--Transformer state",
    )
    model.load_state_dict(torch.load(state_path, map_location=args.device, weights_only=True))
    model.eval()
    state_mean = state_mean_by_type[graph_node_type]
    state_std = state_std_by_type[graph_node_type]

    rows: list[dict[str, object]] = []
    exact_solid: list[float] = []
    chained_solid: list[float] = []
    chained_normalized_histories: list[np.ndarray] = []
    chained_internal_mass_fluxes: list[np.ndarray] = []
    chained_boundary_mass_fluxes: list[np.ndarray] = []
    exact_graph_inference_seconds = 0.0
    chained_graph_inference_seconds = 0.0
    endpoint_condition_ids: list[str] = []
    with torch.no_grad():
        for sequence_id in sequence_ids:
            record = records[sequence_id]
            with np.load(
                dataset_path.parent / str(record["sequence_file"]),
                allow_pickle=False,
            ) as loaded:
                time_s = loaded["time_s"].astype(np.float32)
                condition = loaded["condition_physical"].astype(np.float32)
                target_state = loaded["state_physical"].astype(np.float32)
            source_id = str(record["source_condition_id"])
            target_id = str(record["target_condition_id"])
            endpoint_condition_ids.extend((source_id, target_id))
            source_role = steady_roles[source_id]
            target_role = steady_roles[target_id]
            chained = compose_chained_initial_state(
                steady[source_id], steady[target_id], graph_node_type
            ).astype(np.float32)
            exact = target_state[0]
            exact_normalized = (exact - state_mean) / state_std
            chained_normalized = (chained - state_mean) / state_std
            normalized_condition = (condition - condition_mean) / condition_std
            normalized_time = time_s / maximum_time_s

            def predict(initial: np.ndarray) -> tuple[np.ndarray, float]:
                if str(args.device).startswith("cuda"):
                    torch.cuda.synchronize()
                started = time.perf_counter()
                predicted_normalized = model(
                    torch.as_tensor(initial, device=args.device).unsqueeze(0),
                    torch.as_tensor(normalized_condition, device=args.device).unsqueeze(0),
                    torch.as_tensor(normalized_time, device=args.device).unsqueeze(0),
                    graph,
                )
                if str(args.device).startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                predicted_normalized = predicted_normalized[0].cpu().numpy()
                return (
                    predicted_normalized[..., 4] * state_std[:, 4] + state_mean[:, 4],
                    elapsed,
                )

            exact_temperature, exact_seconds = predict(exact_normalized)
            chained_temperature, chained_seconds = predict(chained_normalized)
            exact_graph_inference_seconds += exact_seconds
            chained_graph_inference_seconds += chained_seconds
            chained_temperature_normalized = (
                (chained_temperature - state_mean[:, 4][None, :])
                / state_std[:, 4][None, :]
            )[..., None]
            target_temperature = target_state[..., 4]
            exact_fluid_rmse = volume_weighted_temperature_rmse(
                exact_temperature, target_temperature, graph_node_type, graph_volume, 0
            )
            exact_solid_rmse = volume_weighted_temperature_rmse(
                exact_temperature, target_temperature, graph_node_type, graph_volume, 1
            )
            chained_fluid_rmse = volume_weighted_temperature_rmse(
                chained_temperature, target_temperature, graph_node_type, graph_volume, 0
            )
            chained_solid_rmse = volume_weighted_temperature_rmse(
                chained_temperature, target_temperature, graph_node_type, graph_volume, 1
            )
            source_temperature_rmse = volume_weighted_temperature_rmse(
                chained[None, :, 4], exact[None, :, 4], graph_node_type, graph_volume, 1
            )
            exact_solid.append(exact_solid_rmse)
            chained_solid.append(chained_solid_rmse)
            chained_normalized_histories.append(
                chained_temperature_normalized.astype(np.float32)
            )
            chained_internal_mass_fluxes.append(
                steady_mass_flux[target_id][0].astype(np.float32)
            )
            chained_boundary_mass_fluxes.append(
                steady_mass_flux[target_id][1].astype(np.float32)
            )
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "source_condition_id": source_id,
                    "target_condition_id": target_id,
                    "source_steady_role": source_role,
                    "target_steady_role": target_role,
                    "endpoint_novelty_class": endpoint_novelty_class(
                        source_role, target_role
                    ),
                    "source_solid_initial_temperature_RMSE_K": source_temperature_rmse,
                    "exact_initial_fluid_temperature_RMSE_K": exact_fluid_rmse,
                    "exact_initial_solid_temperature_RMSE_K": exact_solid_rmse,
                    "steady_PINN_initial_fluid_temperature_RMSE_K": chained_fluid_rmse,
                    "steady_PINN_initial_solid_temperature_RMSE_K": chained_solid_rmse,
                    "solid_temperature_error_amplification": (
                        chained_solid_rmse
                        / max(exact_solid_rmse, np.finfo(np.float64).eps)
                    ),
                }
            )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "chained_initial_state_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    upstream_files = transient_summary.get("temporal_temperature_prediction_files")
    if not isinstance(upstream_files, dict) or args.role not in upstream_files:
        raise ValueError("transient summary does not provide the requested prediction file")
    upstream_path = transient_summary_path.parent / str(upstream_files[args.role])
    upstream_records = transient_summary.get(
        "temporal_temperature_prediction_file_records"
    )
    if not isinstance(upstream_records, dict):
        raise ValueError("transient summary does not record prediction files")
    verify_registered_file(
        upstream_path,
        upstream_records.get(args.role),
        f"transient {args.role} prediction",
    )
    with np.load(upstream_path, allow_pickle=False) as loaded:
        upstream = {name: loaded[name].copy() for name in loaded.files}
    chained_prediction_path = output / f"{args.role}_chained_temperature_predictions.npz"
    chained_artifact = chained_prediction_artifact(
        upstream,
        sequence_ids,
        chained_normalized_histories,
        chained_internal_mass_fluxes,
        chained_boundary_mass_fluxes,
    )
    for name in (
        "source_condition_id",
        "target_condition_id",
        "source_steady_role",
        "target_steady_role",
        "endpoint_novelty_class",
    ):
        chained_artifact[name] = np.asarray([str(row[name]) for row in rows])
    np.savez_compressed(chained_prediction_path, **chained_artifact)
    steady_endpoint_timing = registered_steady_endpoint_inference_time(
        steady_summary=steady_summary,
        role_by_condition=steady_roles,
        condition_ids=endpoint_condition_ids,
    )
    cold_start_seconds = (
        float(steady_endpoint_timing["measured_unique_endpoint_inference_seconds"])
        + chained_graph_inference_seconds
    )
    model_cost = deterministic_chain_model_cost(
        steady_summary=steady_summary,
        transient_summary=transient_summary,
    )
    summary = {
        "status": DETERMINISTIC_CHAIN_STATUS,
        "transient_split_name": transient_summary["split_name"],
        "role": args.role,
        "curve_count": len(rows),
        "exact_initial_mean_solid_temperature_RMSE_K": float(np.mean(exact_solid)),
        "steady_PINN_initial_mean_solid_temperature_RMSE_K": float(
            np.mean(chained_solid)
        ),
        "mean_error_amplification": float(
            np.mean(chained_solid) / max(np.mean(exact_solid), np.finfo(np.float64).eps)
        ),
        "endpoint_novelty_groups": summarize_endpoint_groups(rows),
        "metric_table": csv_path.name,
        "prediction_files": {args.role: chained_prediction_path.name},
        "prediction_file_records": {
            args.role: file_record(chained_prediction_path)
        },
        "upstream_exact_initial_prediction_file": str(upstream_path),
        "timing": {
            "exact_initial_graph_transformer_inference_seconds": exact_graph_inference_seconds,
            "graph_transformer_inference_seconds": chained_graph_inference_seconds,
            "graph_transformer_inference_seconds_per_curve": (
                chained_graph_inference_seconds / len(rows)
            ),
            "registered_steady_PINN_endpoint_timing": steady_endpoint_timing,
            "warm_start_deterministic_chain_inference_seconds": chained_graph_inference_seconds,
            "warm_start_deterministic_chain_inference_seconds_per_curve": (
                chained_graph_inference_seconds / len(rows)
            ),
            "cold_start_deterministic_chain_inference_seconds": cold_start_seconds,
            "cold_start_deterministic_chain_inference_seconds_per_curve": (
                cold_start_seconds / len(rows)
            ),
            "definition": (
                "Warm start assumes steady endpoint fields are already available. Cold start "
                "adds the measured steady-PINN inference time for every unique source or target "
                "endpoint required by this held-out trajectory set; shared endpoints are counted once."
            ),
        },
        "model_cost": model_cost,
        "new_physical_parameters": [],
        "scientific_scope": (
            "The exact-initial result is state-assisted. The chained result uses the steady "
            "physics-constrained PINN to provide source temperature, target hydrodynamics and "
            "target internal/boundary mass fluxes "
            "before the regional graph--Transformer predicts the held-out thermal step. Both remain computed "
            "results on one fixed P418 packing and published endpoint conditions."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
