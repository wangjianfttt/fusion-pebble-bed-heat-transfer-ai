#!/usr/bin/env python3
"""Linear/quadratic response-surface control for fixed-mesh P418 fields."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hccb_p418_comparison_contract import (
    STEADY_METRIC_CONTRACT,
    integrated_heat_transfer_metrics,
    run_provenance,
    split_indices as comparison_split_indices,
    validate_split_and_statistics,
)
from train_hccb_p418_conservative_mixed_operator import (
    engineering_metrics,
    incident_flux_rms,
    normalized_conditions,
    normalized_state,
    physical_state,
)
from train_hccb_p418_regional_operator import load_scales


def design_matrix(condition: np.ndarray, order: int) -> np.ndarray:
    """Standard linear or full quadratic basis of the three varied P418 inputs."""
    varied = np.asarray(condition[:, :3], dtype=np.float64)
    columns = [np.ones(len(varied)), *[varied[:, index] for index in range(3)]]
    if order == 2:
        columns.extend(varied[:, index] ** 2 for index in range(3))
        columns.extend(
            varied[:, left] * varied[:, right]
            for left, right in ((0, 1), (0, 2), (1, 2))
        )
    elif order != 1:
        raise ValueError("response-surface order must be one or two")
    return np.column_stack(columns)


def state_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    node_type: np.ndarray,
    volume: np.ndarray,
) -> dict[str, object]:
    fluid = node_type == 0
    solid = node_type == 1
    channel_mse = [
        float(
            np.sum((prediction[:, fluid, channel] - target[:, fluid, channel]) ** 2 * volume[fluid])
            / (len(prediction) * np.sum(volume[fluid]))
        )
        for channel in range(5)
    ]
    channel_mse.append(
        float(
            np.sum((prediction[:, solid, 4] - target[:, solid, 4]) ** 2 * volume[solid])
            / (len(prediction) * np.sum(volume[solid]))
        )
    )
    return {
        "state_normalized_rmse": float(np.sqrt(np.mean(channel_mse))),
        "state_channel_rmse": np.sqrt(channel_mse).tolist(),
    }


def oriented_balance(
    internal: np.ndarray,
    boundary: np.ndarray,
    internal_owner: np.ndarray,
    internal_neighbour: np.ndarray,
    boundary_owner: np.ndarray,
    node_count: int,
    source: np.ndarray | None = None,
) -> np.ndarray:
    balance = np.zeros((len(internal), node_count), dtype=np.float64)
    if source is not None:
        balance -= source
    for case_index in range(len(internal)):
        np.add.at(balance[case_index], internal_owner, internal[case_index])
        np.add.at(balance[case_index], internal_neighbour, -internal[case_index])
        np.add.at(balance[case_index], boundary_owner, boundary[case_index])
    return balance


def normalized_flow_rmse(prediction: np.ndarray, target: np.ndarray, scale: float) -> float:
    return float(np.sqrt(np.mean(np.square((prediction - target) / scale))))


def fit_response(
    train_design: np.ndarray,
    target: np.ndarray,
    train_index: np.ndarray,
) -> np.ndarray:
    flat = target.reshape(len(target), -1)
    coefficient, *_ = np.linalg.lstsq(train_design, flat[train_index], rcond=None)
    return coefficient


def predict_response(
    design: np.ndarray,
    coefficient: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    return (design @ coefficient).reshape((len(design),) + shape)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-targets", type=Path, required=True)
    parser.add_argument("--mass-targets", type=Path, required=True)
    parser.add_argument("--energy-targets", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--training-statistics", type=Path, required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--comparison-epochs", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    with np.load(args.state_targets.resolve(), allow_pickle=False) as loaded:
        condition_ids = loaded["condition_id"].astype(str)
        condition_physical = loaded["condition_physical"].astype(np.float64)
        state_physical = loaded["state_physical"].astype(np.float64)
        node_type = loaded["node_type"].astype(np.int64)
        node_volume = loaded["node_volume_m3"].astype(np.float64)
    with np.load(args.mass_targets.resolve(), allow_pickle=False) as loaded:
        mass_ids = loaded["condition_id"].astype(str)
        internal_mass_target = loaded["internal_mass_flow_kg_s"].astype(np.float64)
        boundary_mass_target = loaded["boundary_mass_flow_kg_s"].astype(np.float64)
        mass_internal_owner = loaded["internal_owner"].astype(np.int64)
        mass_internal_neighbour = loaded["internal_neighbour"].astype(np.int64)
        boundary_owner = loaded["boundary_owner"].astype(np.int64)
        boundary_patch = loaded["boundary_patch"].astype(np.int64)
        boundary_area = loaded["boundary_face_area_m2"].astype(np.float64)
    with np.load(args.energy_targets.resolve(), allow_pickle=False) as loaded:
        energy_ids = loaded["condition_id"].astype(str)
        internal_energy_target = loaded["internal_energy_flow_W"].astype(np.float64)
        boundary_energy_target = loaded["boundary_energy_flow_W"].astype(np.float64)
        energy_source = loaded["node_source_power_W"].astype(np.float64)
        energy_internal_owner = loaded["internal_owner"].astype(np.int64)
        energy_internal_neighbour = loaded["internal_neighbour"].astype(np.int64)
        energy_internal_kind = loaded["internal_kind"].astype(np.int64)
        energy_internal_kind_name = loaded["internal_kind_name"].astype(str)
        energy_boundary_owner = loaded["boundary_owner"].astype(np.int64)
        energy_boundary_kind = loaded["boundary_kind"].astype(np.int64)
        energy_boundary_kind_name = loaded["boundary_kind_name"].astype(str)
    if not np.array_equal(condition_ids, mass_ids):
        raise ValueError("state and mass target case orders differ")
    if not np.array_equal(condition_ids, energy_ids):
        raise ValueError("state and energy target case orders differ")
    split_case_ids, _ = validate_split_and_statistics(
        split_file=args.split_file,
        training_statistics=args.training_statistics,
        split_name=args.split_name,
        condition_ids=condition_ids,
    )
    split_indices = comparison_split_indices(split_case_ids, condition_ids)
    scales = load_scales(args.training_statistics.resolve(), args.split_name)
    condition_normalized = normalized_conditions(condition_physical, scales)
    target_normalized = normalized_state(
        state_physical, condition_physical, node_type, scales
    ).astype(np.float64)
    train_index = split_indices["train"]
    internal_mass_scale = float(np.sqrt(np.mean(np.square(internal_mass_target[train_index]))))
    boundary_mass_scale = float(np.sqrt(np.mean(np.square(boundary_mass_target[train_index]))))
    mass_balance_scale = incident_flux_rms(
        internal_mass_target[train_index],
        boundary_mass_target[train_index],
        mass_internal_owner,
        mass_internal_neighbour,
        boundary_owner,
        int(np.count_nonzero(node_type == 0)),
    )
    internal_energy_scale = float(np.sqrt(np.mean(np.square(internal_energy_target[train_index]))))
    boundary_energy_scale = float(np.sqrt(np.mean(np.square(boundary_energy_target[train_index]))))
    energy_balance_scale = incident_flux_rms(
        internal_energy_target[train_index],
        boundary_energy_target[train_index],
        energy_internal_owner,
        energy_internal_neighbour,
        energy_boundary_owner,
        len(node_type),
        energy_source[train_index],
    )
    target_energy_balance = oriented_balance(
        internal_energy_target,
        boundary_energy_target,
        energy_internal_owner,
        energy_internal_neighbour,
        energy_boundary_owner,
        len(node_type),
        energy_source,
    )
    candidates: dict[int, dict[str, object]] = {}
    for order in (1, 2):
        train_design = design_matrix(condition_normalized[train_index], order)
        coefficients = {
            "state": fit_response(train_design, target_normalized, train_index),
            "internal_mass": fit_response(train_design, internal_mass_target, train_index),
            "boundary_mass": fit_response(train_design, boundary_mass_target, train_index),
            "internal_energy": fit_response(train_design, internal_energy_target, train_index),
            "boundary_energy": fit_response(train_design, boundary_energy_target, train_index),
        }
        validation_index = split_indices["validation"]
        validation_design = design_matrix(
            condition_normalized[validation_index], order
        )
        validation_state = predict_response(
            validation_design, coefficients["state"], (len(node_type), 5)
        )
        validation_state[:, node_type == 1, :4] = 0.0
        validation_internal_mass = predict_response(
            validation_design,
            coefficients["internal_mass"],
            internal_mass_target.shape[1:],
        )
        validation_boundary_mass = predict_response(
            validation_design,
            coefficients["boundary_mass"],
            boundary_mass_target.shape[1:],
        )
        validation_internal_energy = predict_response(
            validation_design,
            coefficients["internal_energy"],
            internal_energy_target.shape[1:],
        )
        validation_boundary_energy = predict_response(
            validation_design,
            coefficients["boundary_energy"],
            boundary_energy_target.shape[1:],
        )
        metrics = state_metrics(
            validation_state,
            target_normalized[validation_index],
            node_type,
            node_volume,
        )
        validation_mass_balance = oriented_balance(
            validation_internal_mass,
            validation_boundary_mass,
            mass_internal_owner,
            mass_internal_neighbour,
            boundary_owner,
            int(np.count_nonzero(node_type == 0)),
        )
        validation_energy_balance = oriented_balance(
            validation_internal_energy,
            validation_boundary_energy,
            energy_internal_owner,
            energy_internal_neighbour,
            energy_boundary_owner,
            len(node_type),
            energy_source[validation_index],
        )
        metrics.update(
            {
                "internal_flux_normalized_rmse": normalized_flow_rmse(
                    validation_internal_mass,
                    internal_mass_target[validation_index],
                    internal_mass_scale,
                ),
                "boundary_flux_normalized_rmse": normalized_flow_rmse(
                    validation_boundary_mass,
                    boundary_mass_target[validation_index],
                    boundary_mass_scale,
                ),
                "continuity_normalized_rmse": float(
                    np.sqrt(np.mean(np.square(validation_mass_balance / mass_balance_scale)))
                ),
                "internal_energy_normalized_rmse": normalized_flow_rmse(
                    validation_internal_energy,
                    internal_energy_target[validation_index],
                    internal_energy_scale,
                ),
                "boundary_energy_normalized_rmse": normalized_flow_rmse(
                    validation_boundary_energy,
                    boundary_energy_target[validation_index],
                    boundary_energy_scale,
                ),
                "energy_balance_normalized_rmse": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                (validation_energy_balance - target_energy_balance[validation_index])
                                / energy_balance_scale
                            )
                        )
                    )
                ),
            }
        )
        score_names = (
            "state_normalized_rmse",
            "internal_flux_normalized_rmse",
            "boundary_flux_normalized_rmse",
            "continuity_normalized_rmse",
            "internal_energy_normalized_rmse",
            "boundary_energy_normalized_rmse",
            "energy_balance_normalized_rmse",
        )
        validation_score = float(
            np.sqrt(np.mean([float(metrics[name]) ** 2 for name in score_names]))
        )
        candidates[order] = {
            "coefficients": coefficients,
            "validation_state_normalized_rmse": metrics["state_normalized_rmse"],
            "validation_combined_normalized_rmse": validation_score,
        }
    selected_order = min(
        candidates,
        key=lambda order: float(candidates[order]["validation_combined_normalized_rmse"]),
    )
    coefficients = candidates[selected_order]["coefficients"]
    evaluations: dict[str, object] = {}
    prediction_files: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        indices = split_indices[split_name]
        inference_started = time.perf_counter()
        design = design_matrix(condition_normalized[indices], selected_order)
        prediction = predict_response(design, coefficients["state"], (len(node_type), 5))
        prediction[:, node_type == 1, :4] = 0.0
        predicted_internal_mass = predict_response(
            design, coefficients["internal_mass"], internal_mass_target.shape[1:]
        )
        predicted_boundary_mass = predict_response(
            design, coefficients["boundary_mass"], boundary_mass_target.shape[1:]
        )
        predicted_internal_energy = predict_response(
            design, coefficients["internal_energy"], internal_energy_target.shape[1:]
        )
        predicted_boundary_energy = predict_response(
            design, coefficients["boundary_energy"], boundary_energy_target.shape[1:]
        )
        inference_seconds = time.perf_counter() - inference_started
        metrics = state_metrics(
            prediction, target_normalized[indices], node_type, node_volume
        )
        predicted_mass_balance = oriented_balance(
            predicted_internal_mass,
            predicted_boundary_mass,
            mass_internal_owner,
            mass_internal_neighbour,
            boundary_owner,
            int(np.count_nonzero(node_type == 0)),
        )
        predicted_energy_balance = oriented_balance(
            predicted_internal_energy,
            predicted_boundary_energy,
            energy_internal_owner,
            energy_internal_neighbour,
            energy_boundary_owner,
            len(node_type),
            energy_source[indices],
        )
        metrics.update(
            {
                "internal_flux_normalized_rmse": normalized_flow_rmse(
                    predicted_internal_mass, internal_mass_target[indices], internal_mass_scale
                ),
                "boundary_flux_normalized_rmse": normalized_flow_rmse(
                    predicted_boundary_mass, boundary_mass_target[indices], boundary_mass_scale
                ),
                "continuity_normalized_rmse": float(
                    np.sqrt(np.mean(np.square(predicted_mass_balance / mass_balance_scale)))
                ),
                "internal_energy_normalized_rmse": normalized_flow_rmse(
                    predicted_internal_energy, internal_energy_target[indices], internal_energy_scale
                ),
                "boundary_energy_normalized_rmse": normalized_flow_rmse(
                    predicted_boundary_energy, boundary_energy_target[indices], boundary_energy_scale
                ),
                "energy_balance_normalized_rmse": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                (predicted_energy_balance - target_energy_balance[indices])
                                / energy_balance_scale
                            )
                        )
                    )
                ),
            }
        )
        cases: list[dict[str, object]] = []
        for local_position, case_index in enumerate(indices):
            predicted_physical = physical_state(
                prediction[local_position],
                condition_physical[case_index],
                node_type,
                scales,
            )
            predicted_engineering = engineering_metrics(
                predicted_physical,
                boundary_owner=boundary_owner,
                boundary_patch=boundary_patch,
                boundary_area=boundary_area,
                inlet_patch=0,
                outlet_patch=1,
                node_type=node_type,
            )
            reference_engineering = engineering_metrics(
                state_physical[case_index],
                boundary_owner=boundary_owner,
                boundary_patch=boundary_patch,
                boundary_area=boundary_area,
                inlet_patch=0,
                outlet_patch=1,
                node_type=node_type,
            )
            predicted_heat_transfer = integrated_heat_transfer_metrics(
                internal_energy_flow_w=predicted_internal_energy[local_position],
                boundary_energy_flow_w=predicted_boundary_energy[local_position],
                internal_kind=energy_internal_kind,
                internal_kind_name=energy_internal_kind_name,
                boundary_kind=energy_boundary_kind,
                boundary_kind_name=energy_boundary_kind_name,
            )
            reference_heat_transfer = integrated_heat_transfer_metrics(
                internal_energy_flow_w=internal_energy_target[case_index],
                boundary_energy_flow_w=boundary_energy_target[case_index],
                internal_kind=energy_internal_kind,
                internal_kind_name=energy_internal_kind_name,
                boundary_kind=energy_boundary_kind,
                boundary_kind_name=energy_boundary_kind_name,
            )
            cases.append(
                {
                    "condition_id": str(condition_ids[case_index]),
                    "generated_power_W": float(np.sum(energy_source[case_index])),
                    "local_mass_l1_over_two_inlet": float(
                        np.sum(np.abs(predicted_mass_balance[local_position]))
                        / (
                            2.0
                            * abs(
                                np.sum(
                                    boundary_mass_target[case_index][boundary_patch == 0]
                                )
                            )
                        )
                    ),
                    "global_mass_imbalance_over_inlet": float(
                        abs(np.sum(predicted_mass_balance[local_position]))
                        / abs(
                            np.sum(boundary_mass_target[case_index][boundary_patch == 0])
                        )
                    ),
                    "local_energy_l1_over_two_generated_power": float(
                        np.sum(np.abs(predicted_energy_balance[local_position]))
                        / (2.0 * np.sum(energy_source[case_index]))
                    ),
                    "global_energy_imbalance_over_generated_power": float(
                        abs(np.sum(predicted_energy_balance[local_position]))
                        / np.sum(energy_source[case_index])
                    ),
                    "engineering_absolute_errors": {
                        **{
                            name: abs(predicted_engineering[name] - reference_engineering[name])
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
                }
            )
        path = output / f"{split_name}_regional_predictions.npz"
        np.savez_compressed(
            path,
            condition_id=condition_ids[indices],
            condition_normalized=condition_normalized[indices],
            baseline_state_normalized=prediction.astype(np.float32),
            target_state_normalized=target_normalized[indices].astype(np.float32),
            node_type=node_type,
            node_volume_m3=node_volume,
        )
        prediction_files[split_name] = path.name
        evaluations[split_name] = {
            "metrics": metrics,
            "cases": cases,
            "inference_seconds": inference_seconds,
            "inference_seconds_per_case": inference_seconds / len(indices),
        }

    coefficient_count = int(
        sum(np.asarray(value).size for value in coefficients.values())
    )

    code_dir = Path(__file__).resolve().parent
    provenance = run_provenance(
        architecture="response_surface",
        comparison_epochs=args.comparison_epochs,
        split_name=args.split_name,
        split_case_ids=split_case_ids,
        common_inputs={
            "state_targets": args.state_targets,
            "mass_targets": args.mass_targets,
            "energy_targets": args.energy_targets,
            "split_file": args.split_file,
            "training_statistics": args.training_statistics,
        },
        implementation_files=(
            Path(__file__),
            code_dir / "hccb_p418_comparison_contract.py",
            code_dir / "train_hccb_p418_conservative_mixed_operator.py",
        ),
    )
    summary = {
        "status": "regional_response_surface_complete",
        "architecture": "response_surface",
        "split_name": args.split_name,
        "split_case_ids": split_case_ids,
        "comparison_requested_epochs": args.comparison_epochs,
        "epochs": 0,
        "best_epoch": 0,
        "training_seconds": time.time() - started,
        "selected_order": selected_order,
        "candidate_validation_state_normalized_rmse": {
            str(order): float(value["validation_state_normalized_rmse"])
            for order, value in candidates.items()
        },
        "candidate_validation_combined_normalized_rmse": {
            str(order): float(value["validation_combined_normalized_rmse"])
            for order, value in candidates.items()
        },
        "model_parameter_count": coefficient_count,
        "basis": (
            "intercept and three normalized varied inputs"
            if selected_order == 1
            else "intercept, three normalized varied inputs, their squares and pair products"
        ),
        "metric_contract": STEADY_METRIC_CONTRACT,
        "run_provenance": provenance,
        "evaluations": evaluations,
        "regional_prediction_files": prediction_files,
        "normalization": {
            "internal_mass_scale_kg_s": internal_mass_scale,
            "boundary_mass_scale_kg_s": boundary_mass_scale,
            "regional_incident_mass_scale_kg_s": mass_balance_scale,
            "internal_energy_scale_W": internal_energy_scale,
            "boundary_energy_scale_W": boundary_energy_scale,
            "regional_incident_energy_scale_W": energy_balance_scale,
            "scales_use_training_cases_only": True,
        },
        "physical_scope": "fixed P418 mesh with response surfaces for state, mass flow and energy flow",
        "new_physical_parameters": [],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
