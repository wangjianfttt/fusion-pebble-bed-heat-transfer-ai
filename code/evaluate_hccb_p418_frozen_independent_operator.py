#!/usr/bin/env python3
"""Run a frozen P418 transient operator on independent high-Re histories."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from hccb_p418_comparison_contract import file_record, sha256_file
from hccb_p418_fully_coupled_dataset import (
    load_index as load_fully_coupled_index,
    load_sequence as load_fully_coupled_sequence,
    sequence_records as fully_coupled_records,
)
from hccb_p418_fully_coupled_spatiotemporal_operator import (
    FULLY_COUPLED_ARCHITECTURE_REVISION,
    HCCBP418FullyCoupledRegionalOperator,
    build_p418_fully_coupled_flux_graph,
)
from hccb_p418_fully_coupled_training import (
    PHYSICS_TERM_NAMES,
    projection_aware_physics_terms,
)
from hccb_p418_fully_coupled_transient_physics import (
    P418FullyCoupledEquationScales,
    assemble_p418_fully_coupled_transient_residual,
    dimensionless_fully_coupled_equation_terms,
)
from hccb_p418_regional_cht_adapter import load_p418_subface_geometry
from hccb_p418_spatiotemporal_regional_operator import (
    HCCBP418SpatiotemporalRegionalOperator,
    P418ThermalStepRegionalGraph,
    temperature_output_bounds_by_node_type,
)
from hccb_p418_transient_hotspot_metrics import solid_transient_hotspot_metrics
from train_hccb_p418_spatiotemporal_regional_operator import (
    FLUID_TEMPERATURE_RANGE_K,
    SOLID_TEMPERATURE_RANGE_K,
    load_sequence as load_fixed_sequence,
    sequence_records as fixed_records,
)


FIXED_STATUS = "completed_p418_spatiotemporal_regional_operator"
FULL_STATUS = "completed_p418_fully_coupled_spatiotemporal_operator"
ABSOLUTE_EQUATION_TERM_NAMES = (
    "continuity",
    "momentum",
    "fluid_energy",
    "solid_energy",
    "interface_flux",
    "interface_temperature",
    "internal_mass_flux",
    "boundary_mass_flux",
)


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def load_statistics(path: Path) -> dict[str, np.ndarray | float]:
    with np.load(path, allow_pickle=False) as data:
        values: dict[str, np.ndarray | float] = {}
        for name in data.files:
            array = data[name].copy()
            values[name] = float(array) if array.ndim == 0 else array
    return values


def state_scales(
    statistics: dict[str, np.ndarray | float], node_type: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(statistics["state_mean"], dtype=np.float32)[node_type]
    std = np.asarray(statistics["state_std"], dtype=np.float32)[node_type]
    return mean, std


def graph_arrays(path: Path) -> dict[str, np.ndarray]:
    names = (
        "node_centroid_m",
        "node_volume_m3",
        "node_type",
        "edge_source",
        "edge_target",
        "edge_kind",
        "edge_area_m2",
        "edge_area_vector_m2",
        "node_boundary_fraction",
    )
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in names if name in data.files}


def check_training_and_test_graphs(
    training_index: dict[str, object],
    training_root: Path,
    test_index: dict[str, object],
    test_root: Path,
) -> None:
    training = graph_arrays(
        training_root / str(training_index["regional_geometry_file"])
    )
    test = graph_arrays(test_root / str(test_index["regional_geometry_file"]))
    if set(training) != set(test):
        raise ValueError("training and independent-test graph fields differ")
    for name in training:
        if not np.array_equal(training[name], test[name]):
            raise ValueError(
                f"training and independent-test regional graph differ in {name}"
            )


def architecture_arguments(summary: dict[str, object]) -> dict[str, object]:
    architecture = dict(summary["architecture"])
    return {
        "hidden_dim": int(architecture["hidden_dim"]),
        "local_pre_iterations": int(architecture["local_pre_iterations"]),
        "physics_attention_blocks": int(
            architecture["physics_attention_blocks"]
        ),
        "local_post_iterations": int(architecture["local_post_iterations"]),
        "physics_attention_heads": int(
            architecture["physics_attention_heads"]
        ),
        "physics_slices": int(architecture["physics_slices"]),
        "temporal_layers": int(architecture["temporal_layers"]),
        "temporal_heads": int(architecture["temporal_heads"]),
        "spatial_time_chunk_size": int(
            architecture.get("spatial_time_chunk_size", 1)
        ),
        "temporal_node_chunk_size": int(
            architecture.get("temporal_node_chunk_size", 4096)
        ),
    }


def fixed_temperature_output_arguments(
    summary: dict[str, object],
    statistics: dict[str, np.ndarray | float],
    device: torch.device,
) -> dict[str, object]:
    architecture = dict(summary["architecture"])
    mode = str(architecture.get("temperature_output_mode", "additive_normalized"))
    arguments: dict[str, object] = {"temperature_output_mode": mode}
    if mode in {
        "literature_bounded_logit",
        "literature_bounded_residual",
    }:
        bounds = temperature_output_bounds_by_node_type(
            architecture["temperature_output_bounds_K_by_node_type"]
        )
        arguments.update(
            {
                "temperature_mean_k_by_node_type": torch.as_tensor(
                    np.asarray(statistics["state_mean"], dtype=np.float32)[:, 4],
                    device=device,
                ),
                "temperature_std_k_by_node_type": torch.as_tensor(
                    np.asarray(statistics["state_std"], dtype=np.float32)[:, 4],
                    device=device,
                ),
                "temperature_bounds_k_by_node_type": torch.as_tensor(
                    bounds, device=device
                ),
            }
        )
    return arguments


def load_model_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if not isinstance(state, dict) or not state:
        raise ValueError("frozen model state is empty or invalid")
    return state


def equation_scales_from_training_summary(
    summary: dict[str, object],
    device: torch.device,
) -> P418FullyCoupledEquationScales:
    record = summary.get("equation_scales_from_training_curves")
    if not isinstance(record, dict):
        raise ValueError(
            "fully coupled frozen evaluation needs equation scales from training curves"
        )
    names = tuple(P418FullyCoupledEquationScales.__dataclass_fields__)
    if set(record) != set(names):
        raise ValueError("training-summary equation scales are incomplete")
    values: dict[str, torch.Tensor] = {}
    for name in names:
        value = float(record[name])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"training-summary equation scale {name} is invalid")
        values[name] = torch.tensor(value, device=device, dtype=torch.float32)
    return P418FullyCoupledEquationScales(**values)


def normalized_equation_metrics(
    *,
    prediction_residual,
    reference_residual,
    scales: P418FullyCoupledEquationScales,
    fluid_volume_m3: torch.Tensor,
    solid_volume_m3: torch.Tensor,
) -> dict[str, float]:
    difference = projection_aware_physics_terms(
        prediction=prediction_residual,
        reference=reference_residual,
        scales=scales,
        fluid_volume_m3=fluid_volume_m3,
        solid_volume_m3=solid_volume_m3,
    )
    prediction = dimensionless_fully_coupled_equation_terms(
        residual=prediction_residual,
        scales=scales,
        fluid_volume_m3=fluid_volume_m3,
        solid_volume_m3=solid_volume_m3,
    )
    reference = dimensionless_fully_coupled_equation_terms(
        residual=reference_residual,
        scales=scales,
        fluid_volume_m3=fluid_volume_m3,
        solid_volume_m3=solid_volume_m3,
    )
    return {
        **{
            f"physics_difference_{name}_normalized_RMSE": float(
                torch.sqrt(difference[name]).detach().cpu()
            )
            for name in PHYSICS_TERM_NAMES
        },
        **{
            f"predicted_equation_{name}_normalized_RMS": float(
                torch.sqrt(prediction[name]).detach().cpu()
            )
            for name in ABSOLUTE_EQUATION_TERM_NAMES
        },
        **{
            f"reference_equation_{name}_normalized_RMS": float(
                torch.sqrt(reference[name]).detach().cpu()
            )
            for name in ABSOLUTE_EQUATION_TERM_NAMES
        },
    }


def volume_weighted_temperature_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    node_type: np.ndarray,
    node_volume: np.ndarray,
) -> dict[str, float]:
    error = prediction - target
    result: dict[str, float] = {}
    for material, label in ((0, "fluid"), (1, "solid")):
        selected = node_type == material
        weight = node_volume[selected] / node_volume[selected].sum()
        mse = np.mean(
            np.sum(np.square(error[:, selected]) * weight[None, :], axis=-1)
        )
        result[f"{label}_temperature_volume_weighted_RMSE_K"] = float(
            math.sqrt(mse)
        )
        result[f"{label}_temperature_maximum_absolute_error_K"] = float(
            np.max(np.abs(error[:, selected]))
        )
    return result


def mean_square_summary(rows: list[dict[str, float]], names: tuple[str, ...]) -> dict[str, float]:
    return {
        name: float(math.sqrt(np.mean([row[name] ** 2 for row in rows])))
        for name in names
    }


def verify_independent_role(
    summary: dict[str, object],
    test_index: dict[str, object],
) -> list[str]:
    records = test_index.get("sequences", [])
    sequence_ids = [str(row["sequence_id"]) for row in records]
    if len(sequence_ids) != 6 or len(sequence_ids) != len(set(sequence_ids)):
        raise ValueError("the high-Re independent set must contain six unique curves")
    used = set()
    for key in ("split_case_ids", "split_sequence_ids"):
        split = summary.get(key, {})
        if isinstance(split, dict):
            for values in split.values():
                used.update(map(str, values))
    overlap = sorted(used.intersection(sequence_ids))
    if overlap:
        raise ValueError(
            f"independent high-Re curves overlap model-development curves: {overlap}"
        )
    return sequence_ids


def run_fixed(
    *,
    summary: dict[str, object],
    test_index: dict[str, object],
    test_root: Path,
    graph: P418ThermalStepRegionalGraph,
    statistics: dict[str, np.ndarray | float],
    model_state: dict[str, torch.Tensor],
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, object], dict[str, str]]:
    records = fixed_records(test_index)
    sequence_ids = [str(row["sequence_id"]) for row in test_index["sequences"]]
    architecture = architecture_arguments(summary)
    architecture["spatial_temporal_mode"] = str(
        summary["architecture"].get(
            "spatial_temporal_mode", "repeated_query_spatial"
        )
    )
    model = HCCBP418SpatiotemporalRegionalOperator(
        condition_dim=len(test_index["condition_names"]),
        boundary_role_count=graph.boundary_role_count,
        **fixed_temperature_output_arguments(summary, statistics, device),
        **architecture,
    ).to(device)
    model.load_state_dict(model_state, strict=True)
    model.eval()

    condition_mean = np.asarray(statistics["condition_mean"], dtype=np.float32)
    condition_std = np.asarray(statistics["condition_std"], dtype=np.float32)
    state_mean, state_std = state_scales(
        statistics, graph.node_type.detach().cpu().numpy()
    )
    maximum_time = float(statistics["maximum_time_s"])
    node_type = graph.node_type.detach().cpu().numpy()
    node_volume = graph.volume_m3.detach().cpu().numpy()
    node_centroid = graph.centroid_m.detach().cpu().numpy()

    times = []
    conditions = []
    fixed_hydrodynamics = []
    internal_fluxes = []
    boundary_fluxes = []
    predictions_normalized = []
    targets_normalized = []
    prediction_temperature = []
    target_temperature = []
    rows: list[dict[str, float | str]] = []
    inference_total = 0.0

    with torch.no_grad():
        for sequence_id in sequence_ids:
            time_s, condition, state, internal_flux, boundary_flux = load_fixed_sequence(
                test_root, records[sequence_id]
            )
            normalized_state = (state - state_mean[None]) / state_std[None]
            normalized_condition = (condition - condition_mean) / condition_std
            normalized_time = time_s / maximum_time
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            prediction_normalized = model(
                torch.as_tensor(
                    normalized_state[None, 0], device=device
                ),
                torch.as_tensor(
                    normalized_condition[None], device=device
                ),
                torch.as_tensor(normalized_time[None], device=device),
                graph,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            inference_total += elapsed
            prediction_normalized_np = prediction_normalized[0].cpu().numpy()
            prediction = (
                prediction_normalized_np * state_std[None] + state_mean[None]
            )
            metrics = volume_weighted_temperature_metrics(
                prediction[..., 4], state[..., 4], node_type, node_volume
            )
            fluid_prediction = prediction[..., 4][..., node_type == 0]
            solid_prediction = prediction[..., 4][..., node_type == 1]
            fluid_low_k, fluid_high_k = FLUID_TEMPERATURE_RANGE_K
            low_k, high_k = SOLID_TEMPERATURE_RANGE_K
            metrics.update(
                {
                    "predicted_fluid_temperature_minimum_K": float(
                        fluid_prediction.min()
                    ),
                    "predicted_fluid_temperature_maximum_K": float(
                        fluid_prediction.max()
                    ),
                    "predicted_fluid_temperature_outside_registered_range_fraction": float(
                        np.mean(
                            (fluid_prediction < fluid_low_k)
                            | (fluid_prediction > fluid_high_k)
                        )
                    ),
                    "predicted_solid_temperature_minimum_K": float(
                        solid_prediction.min()
                    ),
                    "predicted_solid_temperature_maximum_K": float(
                        solid_prediction.max()
                    ),
                    "predicted_solid_temperature_outside_registered_range_fraction": float(
                        np.mean(
                            (solid_prediction < low_k)
                            | (solid_prediction > high_k)
                        )
                    ),
                }
            )
            metrics["sequence_id"] = sequence_id
            metrics["inference_seconds"] = elapsed
            rows.append(metrics)
            times.append(time_s)
            conditions.append(condition)
            fixed_hydrodynamics.append(state[0, :, :4])
            internal_fluxes.append(internal_flux)
            boundary_fluxes.append(boundary_flux)
            predictions_normalized.append(prediction_normalized_np[..., 4:5])
            targets_normalized.append(normalized_state[..., 4:5])
            prediction_temperature.append(prediction[..., 4])
            target_temperature.append(state[..., 4])

    prediction_temperature_array = np.stack(prediction_temperature)
    target_temperature_array = np.stack(target_temperature)
    hotspot = solid_transient_hotspot_metrics(
        prediction_temperature_array,
        target_temperature_array,
        node_type,
        node_centroid,
    )
    metric_names = (
        "fluid_temperature_volume_weighted_RMSE_K",
        "solid_temperature_volume_weighted_RMSE_K",
    )
    aggregate = {
        **mean_square_summary(rows, metric_names),
        "fluid_temperature_maximum_absolute_error_K": max(
            float(row["fluid_temperature_maximum_absolute_error_K"])
            for row in rows
        ),
        "solid_temperature_maximum_absolute_error_K": max(
            float(row["solid_temperature_maximum_absolute_error_K"])
            for row in rows
        ),
        **hotspot,
        "predicted_fluid_temperature_minimum_K": min(
            float(row["predicted_fluid_temperature_minimum_K"]) for row in rows
        ),
        "predicted_fluid_temperature_maximum_K": max(
            float(row["predicted_fluid_temperature_maximum_K"]) for row in rows
        ),
        "predicted_fluid_temperature_outside_registered_range_fraction": float(
            np.mean(
                [
                    float(
                        row[
                            "predicted_fluid_temperature_outside_registered_range_fraction"
                        ]
                    )
                    for row in rows
                ]
            )
        ),
        "registered_fluid_temperature_range_K": list(
            FLUID_TEMPERATURE_RANGE_K
        ),
        "predicted_solid_temperature_minimum_K": min(
            float(row["predicted_solid_temperature_minimum_K"]) for row in rows
        ),
        "predicted_solid_temperature_maximum_K": max(
            float(row["predicted_solid_temperature_maximum_K"]) for row in rows
        ),
        "predicted_solid_temperature_outside_registered_range_fraction": float(
            np.mean(
                [
                    float(
                        row[
                            "predicted_solid_temperature_outside_registered_range_fraction"
                        ]
                    )
                    for row in rows
                ]
            )
        ),
        "registered_solid_temperature_range_K": list(
            SOLID_TEMPERATURE_RANGE_K
        ),
        "total_inference_seconds": inference_total,
        "mean_inference_seconds_per_curve": inference_total / len(rows),
    }
    prediction_path = output_dir / "test_temporal_temperature_predictions.npz"
    np.savez_compressed(
        prediction_path,
        sequence_id=np.asarray(sequence_ids),
        time_s=np.stack(times),
        condition_physical=np.stack(conditions),
        fixed_hydrodynamics_physical=np.stack(fixed_hydrodynamics),
        fluid_internal_mass_flux_kg_s=np.stack(internal_fluxes),
        fluid_boundary_mass_flux_kg_s=np.stack(boundary_fluxes),
        baseline_temperature_normalized=np.stack(predictions_normalized),
        target_temperature_normalized=np.stack(targets_normalized),
        node_type=node_type,
        node_volume_m3=node_volume,
        node_centroid_m=node_centroid,
        temperature_mean_K_by_node_type=np.asarray(statistics["state_mean"])[:, 4],
        temperature_std_K_by_node_type=np.asarray(statistics["state_std"])[:, 4],
    )
    return (
        {"aggregate_metrics": aggregate, "per_curve_metrics": rows},
        {"test": prediction_path.name},
    )


def full_state_metrics(
    prediction_state: np.ndarray,
    target_state: np.ndarray,
    prediction_internal: np.ndarray,
    target_internal: np.ndarray,
    prediction_boundary: np.ndarray,
    target_boundary: np.ndarray,
    node_type: np.ndarray,
    node_volume: np.ndarray,
) -> dict[str, float]:
    error = prediction_state - target_state
    fluid = node_type == 0
    solid = node_type == 1
    fluid_weight = node_volume[fluid] / node_volume[fluid].sum()
    solid_weight = node_volume[solid] / node_volume[solid].sum()

    def weighted_rmse(values: np.ndarray, weight: np.ndarray) -> float:
        return float(
            math.sqrt(
                np.mean(np.sum(np.square(values) * weight[None, :, None], axis=1))
            )
        )

    def weighted_scalar_rmse(values: np.ndarray, weight: np.ndarray) -> float:
        return float(
            math.sqrt(
                np.mean(np.sum(np.square(values) * weight[None, :], axis=1))
            )
        )

    return {
        "fluid_velocity_volume_weighted_RMSE_m_s": weighted_rmse(
            error[:, fluid, :3], fluid_weight
        ),
        "fluid_pressure_volume_weighted_RMSE_Pa": weighted_scalar_rmse(
            error[:, fluid, 3], fluid_weight
        ),
        "fluid_temperature_volume_weighted_RMSE_K": weighted_scalar_rmse(
            error[:, fluid, 4], fluid_weight
        ),
        "solid_temperature_volume_weighted_RMSE_K": weighted_scalar_rmse(
            error[:, solid, 4], solid_weight
        ),
        "internal_mass_flux_RMSE_kg_s": float(
            np.sqrt(np.mean(np.square(prediction_internal - target_internal)))
        ),
        "boundary_mass_flux_RMSE_kg_s": float(
            np.sqrt(np.mean(np.square(prediction_boundary - target_boundary)))
        ),
        "state_maximum_absolute_error": float(np.max(np.abs(error))),
    }


def run_fully_coupled(
    *,
    summary: dict[str, object],
    test_index: dict[str, object],
    test_root: Path,
    graph: P418ThermalStepRegionalGraph,
    statistics: dict[str, np.ndarray | float],
    model_state: dict[str, torch.Tensor],
    residual_geometry: Path,
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, object], dict[str, list[str]]]:
    records = fully_coupled_records(test_index)
    sequence_ids = [str(row["sequence_id"]) for row in test_index["sequences"]]
    geometry = load_p418_subface_geometry(
        residual_geometry,
        fluid_patch_names=test_index["boundary_patch_names"]["fluid"],
        solid_patch_names=test_index["boundary_patch_names"]["solid"],
        device=device,
        dtype=torch.float32,
    )
    flux_graph = build_p418_fully_coupled_flux_graph(
        geometry=geometry, graph=graph
    )
    architecture = architecture_arguments(summary)
    model = HCCBP418FullyCoupledRegionalOperator(
        condition_dim=len(test_index["condition_names"]),
        boundary_role_count=graph.boundary_role_count,
        internal_face_feature_dim=flux_graph.internal_features.shape[1],
        boundary_face_feature_dim=flux_graph.boundary_features.shape[1],
        **architecture,
    ).to(device)
    model.load_state_dict(model_state, strict=True)
    model.eval()

    condition_mean = np.asarray(statistics["condition_mean"], dtype=np.float32)
    condition_std = np.asarray(statistics["condition_std"], dtype=np.float32)
    node_type = graph.node_type.detach().cpu().numpy()
    node_volume = graph.volume_m3.detach().cpu().numpy()
    node_centroid = graph.centroid_m.detach().cpu().numpy()
    state_mean, state_std = state_scales(statistics, node_type)
    maximum_time = float(statistics["maximum_time_s"])
    internal_mean = float(statistics["internal_mass_flux_mean_kg_s"])
    internal_std = float(statistics["internal_mass_flux_std_kg_s"])
    boundary_mean = float(statistics["boundary_mass_flux_mean_kg_s"])
    boundary_std = float(statistics["boundary_mass_flux_std_kg_s"])
    equation_scales = equation_scales_from_training_summary(summary, device)

    rows: list[dict[str, float | str]] = []
    prediction_paths: list[str] = []
    hotspot_prediction = []
    hotspot_target = []
    inference_total = 0.0
    with torch.no_grad():
        for sequence_id in sequence_ids:
            time_s, condition, state, internal, boundary = load_fully_coupled_sequence(
                test_root, records[sequence_id]
            )
            state_normalized = (state - state_mean[None]) / state_std[None]
            condition_normalized = (condition - condition_mean) / condition_std
            internal_normalized = (internal - internal_mean) / internal_std
            boundary_normalized = (boundary - boundary_mean) / boundary_std
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            normalized = model(
                torch.as_tensor(state_normalized[None, 0], device=device),
                torch.as_tensor(internal_normalized[None, 0], device=device),
                torch.as_tensor(boundary_normalized[None, 0], device=device),
                torch.as_tensor(condition_normalized[None], device=device),
                torch.as_tensor((time_s / maximum_time)[None], device=device),
                graph,
                flux_graph,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            inference_total += elapsed
            predicted_state = (
                normalized.state[0].cpu().numpy() * state_std[None]
                + state_mean[None]
            )
            predicted_internal = (
                normalized.internal_mass_flux[0].cpu().numpy() * internal_std
                + internal_mean
            )
            predicted_boundary = (
                normalized.boundary_mass_flux[0].cpu().numpy() * boundary_std
                + boundary_mean
            )
            metrics = full_state_metrics(
                predicted_state,
                state,
                predicted_internal,
                internal,
                predicted_boundary,
                boundary,
                node_type,
                node_volume,
            )
            condition_tensor = torch.as_tensor(
                condition[None], device=device, dtype=torch.float32
            )
            time_tensor = torch.as_tensor(
                time_s, device=device, dtype=torch.float32
            )
            predicted_residual = assemble_p418_fully_coupled_transient_residual(
                geometry=geometry,
                step_condition=condition_tensor,
                state_physical=torch.as_tensor(
                    predicted_state[None], device=device, dtype=torch.float32
                ),
                time_s=time_tensor,
                fluid_internal_mass_flux_kg_s=torch.as_tensor(
                    predicted_internal[None], device=device, dtype=torch.float32
                ),
                fluid_boundary_mass_flux_kg_s=torch.as_tensor(
                    predicted_boundary[None], device=device, dtype=torch.float32
                ),
            )
            reference_residual = assemble_p418_fully_coupled_transient_residual(
                geometry=geometry,
                step_condition=condition_tensor,
                state_physical=torch.as_tensor(
                    state[None], device=device, dtype=torch.float32
                ),
                time_s=time_tensor,
                fluid_internal_mass_flux_kg_s=torch.as_tensor(
                    internal[None], device=device, dtype=torch.float32
                ),
                fluid_boundary_mass_flux_kg_s=torch.as_tensor(
                    boundary[None], device=device, dtype=torch.float32
                ),
            )
            metrics.update(
                normalized_equation_metrics(
                    prediction_residual=predicted_residual,
                    reference_residual=reference_residual,
                    scales=equation_scales,
                    fluid_volume_m3=geometry.fluid_mesh.cell_volume,
                    solid_volume_m3=geometry.solid_mesh.cell_volume,
                )
            )
            metrics["sequence_id"] = sequence_id
            metrics["inference_seconds"] = elapsed
            rows.append(metrics)
            hotspot_prediction.append(predicted_state[..., 4])
            hotspot_target.append(state[..., 4])
            prediction_path = output_dir / f"test_{sequence_id}_prediction.npz"
            np.savez_compressed(
                prediction_path,
                sequence_id=np.asarray(sequence_id),
                time_s=time_s,
                condition_physical=condition,
                state_prediction=predicted_state,
                state_target=state,
                internal_mass_flux_prediction=predicted_internal,
                internal_mass_flux_target=internal,
                boundary_mass_flux_prediction=predicted_boundary,
                boundary_mass_flux_target=boundary,
            )
            prediction_paths.append(prediction_path.name)

    names = (
        "fluid_velocity_volume_weighted_RMSE_m_s",
        "fluid_pressure_volume_weighted_RMSE_Pa",
        "fluid_temperature_volume_weighted_RMSE_K",
        "solid_temperature_volume_weighted_RMSE_K",
        "internal_mass_flux_RMSE_kg_s",
        "boundary_mass_flux_RMSE_kg_s",
        *tuple(
            f"physics_difference_{name}_normalized_RMSE"
            for name in PHYSICS_TERM_NAMES
        ),
        *tuple(
            f"predicted_equation_{name}_normalized_RMS"
            for name in ABSOLUTE_EQUATION_TERM_NAMES
        ),
        *tuple(
            f"reference_equation_{name}_normalized_RMS"
            for name in ABSOLUTE_EQUATION_TERM_NAMES
        ),
    )
    aggregate = {
        **mean_square_summary(rows, names),
        "state_maximum_absolute_error": max(
            float(row["state_maximum_absolute_error"]) for row in rows
        ),
        **solid_transient_hotspot_metrics(
            np.stack(hotspot_prediction),
            np.stack(hotspot_target),
            node_type,
            node_centroid,
        ),
        "total_inference_seconds": inference_total,
        "mean_inference_seconds_per_curve": inference_total / len(rows),
        "equation_scale_source": "training_curves_only",
    }
    return (
        {"aggregate_metrics": aggregate, "per_curve_metrics": rows},
        {"test": prediction_paths},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fixed", "fully_coupled"), required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--training-dataset-index", type=Path, required=True)
    parser.add_argument("--test-dataset-index", type=Path, required=True)
    parser.add_argument("--model-state", type=Path)
    parser.add_argument("--training-statistics", type=Path)
    parser.add_argument("--residual-geometry", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    device = select_device(args.device)
    summary_path = args.training_summary.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_status = FIXED_STATUS if args.mode == "fixed" else FULL_STATUS
    if summary.get("status") != expected_status:
        raise ValueError(
            f"{args.mode} frozen model summary has status {summary.get('status')!r}"
        )
    if args.mode == "fully_coupled":
        recorded_revision = summary.get("architecture", {}).get("revision")
        if recorded_revision != FULLY_COUPLED_ARCHITECTURE_REVISION:
            raise ValueError(
                "fully coupled model architecture revision differs from the current "
                "oriented face-flux implementation"
            )
    model_state_path = (
        args.model_state.resolve()
        if args.model_state is not None
        else summary_path.parent / "model_state.pt"
    )
    statistics_path = (
        args.training_statistics.resolve()
        if args.training_statistics is not None
        else summary_path.parent / "training_statistics.npz"
    )
    if not model_state_path.is_file() or not statistics_path.is_file():
        raise FileNotFoundError("frozen model state or training statistics are missing")
    state_record = summary.get("model_state_record")
    if isinstance(state_record, dict) and state_record.get("sha256"):
        if sha256_file(model_state_path) != state_record["sha256"]:
            raise ValueError("frozen model-state hash differs from its training summary")
    if summary.get("model_state_sha256"):
        if sha256_file(model_state_path) != summary["model_state_sha256"]:
            raise ValueError("frozen full-model state hash differs from its summary")

    training_index_path = args.training_dataset_index.resolve()
    test_index_path = args.test_dataset_index.resolve()
    if args.mode == "fully_coupled":
        training_index = load_fully_coupled_index(training_index_path)
        test_index = load_fully_coupled_index(test_index_path)
    else:
        training_index = json.loads(training_index_path.read_text(encoding="utf-8"))
        test_index = json.loads(test_index_path.read_text(encoding="utf-8"))
    if training_index.get("condition_names") != test_index.get("condition_names"):
        raise ValueError("training and independent-test condition vectors differ")
    if training_index.get("state_names") != test_index.get("state_names"):
        raise ValueError("training and independent-test state vectors differ")
    check_training_and_test_graphs(
        training_index,
        training_index_path.parent,
        test_index,
        test_index_path.parent,
    )
    sequence_ids = verify_independent_role(summary, test_index)
    graph_path = (
        test_index_path.parent / str(test_index["regional_geometry_file"])
    )
    graph = P418ThermalStepRegionalGraph.from_npz(graph_path, device=device)
    statistics = load_statistics(statistics_path)
    model_state = load_model_state(model_state_path, device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "fixed":
        result, prediction_files = run_fixed(
            summary=summary,
            test_index=test_index,
            test_root=test_index_path.parent,
            graph=graph,
            statistics=statistics,
            model_state=model_state,
            device=device,
            output_dir=output_dir,
        )
    else:
        if args.residual_geometry is None:
            raise ValueError("fully coupled frozen inference needs residual geometry")
        result, prediction_files = run_fully_coupled(
            summary=summary,
            test_index=test_index,
            test_root=test_index_path.parent,
            graph=graph,
            statistics=statistics,
            model_state=model_state,
            residual_geometry=args.residual_geometry.resolve(),
            device=device,
            output_dir=output_dir,
        )

    output = {
        "status": "completed_p418_frozen_high_re_independent_evaluation",
        "mode": args.mode,
        "training_or_model_selection_performed": False,
        "independent_reference_curve_count": len(sequence_ids),
        "independent_sequence_ids": sequence_ids,
        "training_summary": str(summary_path),
        "training_summary_record": file_record(summary_path),
        "training_dataset_index": str(training_index_path),
        "training_dataset_index_record": file_record(training_index_path),
        "test_dataset_index": str(test_index_path),
        "test_dataset_index_record": file_record(test_index_path),
        "model_state_record": file_record(model_state_path),
        "training_statistics_record": file_record(statistics_path),
        "regional_geometry_record": file_record(graph_path),
        "prediction_files": prediction_files,
        "compute_device": str(device),
        **result,
        "new_physical_parameters": [],
    }
    summary_output = output_dir / "summary.json"
    summary_output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
