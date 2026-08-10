#!/usr/bin/env python3
"""Evaluate P418 temperature models with one projection-aware energy operator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from hccb_p418_regional_cht_adapter import load_p418_subface_geometry
from hccb_p418_transient_regional_physics import (
    assemble_p418_transient_regional_residual,
)
from hccb_source_backed_thermophysical import load_hccb_thermophysical_parameters


ROLES = ("train", "validation", "test")
PRIMARY_ENERGY_METRIC = "projection_aware_volume_weighted_energy_equation_normalized_RMSE"


def prediction_files(
    summary: dict, roles: tuple[str, ...] = ROLES
) -> dict[str, str]:
    files = summary.get("prediction_files")
    if files is None:
        files = summary.get("temporal_temperature_prediction_files")
    if not isinstance(files, dict) or not set(roles).issubset(files):
        raise ValueError(f"model summary does not list predictions for {list(roles)}")
    return {role: str(files[role]) for role in roles}


def temperature_fields(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if "temperature_prediction_K" in data:
        prediction = data["temperature_prediction_K"].astype(np.float32)
        target = data["temperature_target_K"].astype(np.float32)
    else:
        for name in (
            "corrected_temperature_normalized",
            "refined_temperature_normalized",
            "baseline_temperature_normalized",
        ):
            if name in data:
                prediction_normalized = data[name].astype(np.float32)
                break
        else:
            raise ValueError("prediction artifact has no supported temperature field")
        target_normalized = data["target_temperature_normalized"].astype(np.float32)
        node_type = data["node_type"].astype(np.int64)
        mean = data["temperature_mean_K_by_node_type"].astype(np.float32)
        std = data["temperature_std_K_by_node_type"].astype(np.float32)
        scale = std[node_type][None, None, :]
        offset = mean[node_type][None, None, :]
        prediction = prediction_normalized[..., 0] * scale + offset
        target = target_normalized[..., 0] * scale + offset
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("temperature fields must have shape [curve,time,node]")
    return prediction, target


def registered_temperature_range_diagnostics(
    data: dict[str, np.ndarray],
) -> dict[str, float | bool | int | list[float]]:
    """Check whether temperature histories remain inside registered relations.

    The helium relations require positive absolute temperature.  The solid
    sensible-energy relation is explicitly limited to the literature interval
    stored in the source-backed thermophysical manifest.  Predictions outside
    that interval must not be clipped or extrapolated merely to obtain an
    energy-residual number.
    """
    prediction, reference = temperature_fields(data)
    node_type = np.asarray(data["node_type"], dtype=np.int64)
    if node_type.shape != (prediction.shape[-1],):
        raise ValueError("node types differ from temperature nodes")
    fluid = node_type == 0
    solid = node_type == 1
    if not fluid.any() or not solid.any() or np.any(~np.isin(node_type, (0, 1))):
        raise ValueError("temperature artifact must contain fluid and solid nodes")
    low, high = load_hccb_thermophysical_parameters().solid_cp_temperature_range_k

    output: dict[str, float | bool | int | list[float]] = {
        "solid_registered_temperature_range_K": [float(low), float(high)],
    }
    validity = {}
    for label, values in (("prediction", prediction), ("reference", reference)):
        fluid_values = values[..., fluid]
        solid_values = values[..., solid]
        finite = bool(np.isfinite(values).all())
        fluid_nonpositive = int(np.count_nonzero(fluid_values <= 0.0))
        solid_outside = int(
            np.count_nonzero((solid_values < low) | (solid_values > high))
        )
        output.update(
            {
                f"{label}_fluid_temperature_min_K": float(np.nanmin(fluid_values)),
                f"{label}_fluid_temperature_max_K": float(np.nanmax(fluid_values)),
                f"{label}_solid_temperature_min_K": float(np.nanmin(solid_values)),
                f"{label}_solid_temperature_max_K": float(np.nanmax(solid_values)),
                f"{label}_nonfinite_value_count": int(
                    values.size - np.count_nonzero(np.isfinite(values))
                ),
                f"{label}_fluid_nonpositive_value_count": fluid_nonpositive,
                f"{label}_solid_out_of_range_value_count": solid_outside,
                f"{label}_solid_out_of_range_fraction": float(
                    solid_outside / solid_values.size
                ),
            }
        )
        validity[label] = finite and fluid_nonpositive == 0 and solid_outside == 0
        output[f"{label}_within_registered_thermophysical_range"] = validity[label]
    return output


def physical_state(temperature_k: np.ndarray, hydrodynamics: np.ndarray) -> np.ndarray:
    if temperature_k.ndim != 2:
        raise ValueError("one temperature history must have shape [time,node]")
    if hydrodynamics.shape != (temperature_k.shape[1], 4):
        raise ValueError("fixed hydrodynamics must have shape [node,4]")
    fixed = np.broadcast_to(
        hydrodynamics[None], (temperature_k.shape[0], temperature_k.shape[1], 4)
    )
    return np.concatenate((fixed, temperature_k[..., None]), axis=-1).astype(np.float32)


def residual_components(
    residual,
    condition: torch.Tensor,
    fluid_volume_m3: torch.Tensor,
    solid_volume_m3: torch.Tensor,
) -> dict[str, float]:
    return residual_field_components(
        residual.fluid_energy_w_m3,
        residual.solid_energy_w_m3,
        condition,
        fluid_volume_m3,
        solid_volume_m3,
    )


def residual_field_components(
    fluid_energy_w_m3: torch.Tensor,
    solid_energy_w_m3: torch.Tensor,
    condition: torch.Tensor,
    fluid_volume_m3: torch.Tensor,
    solid_volume_m3: torch.Tensor,
) -> dict[str, float]:
    """Summarize dimensional fluid and solid equation fields.

    The inputs may be absolute regional residuals or the difference between a
    prediction and the projected OpenFOAM reference.  This distinction is
    necessary because nonlinear face fluxes and regional volume averaging do
    not commute.
    """
    source = (condition[:, 5] * 1.0e6).clamp_min(
        torch.finfo(condition.dtype).tiny
    )
    fluid = fluid_energy_w_m3 / source[:, None, None]
    solid = solid_energy_w_m3 / source[:, None, None]
    if fluid.shape[-1] != len(fluid_volume_m3):
        raise ValueError("fluid residual and regional volumes differ")
    if solid.shape[-1] != len(solid_volume_m3):
        raise ValueError("solid residual and regional volumes differ")
    fluid_weight = fluid_volume_m3 / fluid_volume_m3.sum()
    solid_weight = solid_volume_m3 / solid_volume_m3.sum()
    total_volume = fluid_volume_m3.sum() + solid_volume_m3.sum()
    source_power = source * solid_volume_m3.sum()
    fluid_power = (
        fluid_energy_w_m3 * fluid_volume_m3[None, None, :]
    ).sum(dim=-1)
    solid_power = (
        solid_energy_w_m3 * solid_volume_m3[None, None, :]
    ).sum(dim=-1)
    global_closure = (fluid_power + solid_power) / source_power[:, None]
    local_l1 = (
        (
            fluid_energy_w_m3.abs()
            * fluid_volume_m3[None, None, :]
        ).sum(dim=-1)
        + (
            solid_energy_w_m3.abs()
            * solid_volume_m3[None, None, :]
        ).sum(dim=-1)
    ) / source_power[:, None]
    fluid_mse = float(fluid.square().mean().cpu())
    solid_mse = float(solid.square().mean().cpu())
    fluid_volume_mse = float(
        (fluid.square() * fluid_weight[None, None, :]).sum(dim=-1).mean().cpu()
    )
    solid_volume_mse = float(
        (solid.square() * solid_weight[None, None, :]).sum(dim=-1).mean().cpu()
    )
    volume_weighted_mse = float(
        (
            (
                fluid.square() * fluid_volume_m3[None, None, :]
            ).sum(dim=-1)
            + (
                solid.square() * solid_volume_m3[None, None, :]
            ).sum(dim=-1)
        ).mean().cpu()
        / float(total_volume.cpu())
    )
    return {
        "fluid_mse": fluid_mse,
        "solid_mse": solid_mse,
        "combined_mse": 0.5 * (fluid_mse + solid_mse),
        "fluid_volume_weighted_mse": fluid_volume_mse,
        "solid_volume_weighted_mse": solid_volume_mse,
        "volume_weighted_mse": volume_weighted_mse,
        "global_closure_mse": float(global_closure.square().mean().cpu()),
        "local_l1_mse": float(local_l1.square().mean().cpu()),
    }


def summarize_endpoint_energy_groups(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, float | int]]:
    """Combine per-curve energy errors without mixing endpoint novelty levels."""
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        label = row.get("endpoint_novelty_class")
        if label is None:
            continue
        grouped.setdefault(str(label), []).append(row)
    output: dict[str, dict[str, float | int]] = {}
    row_metric = "projection_aware_volume_weighted_energy_RMSE"
    for label, selected in grouped.items():
        values = np.asarray([float(row[row_metric]) for row in selected])
        output[label] = {
            "curve_count": len(selected),
            PRIMARY_ENERGY_METRIC: float(math.sqrt(np.mean(np.square(values)))),
        }
    return output


def summarize_role(
    *,
    artifact: dict[str, np.ndarray],
    geometry,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    required = {
        "sequence_id",
        "time_s",
        "condition_physical",
        "fixed_hydrodynamics_physical",
        "fluid_internal_mass_flux_kg_s",
        "fluid_boundary_mass_flux_kg_s",
        "node_type",
        "node_volume_m3",
    }
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"prediction artifact lacks {sorted(missing)}")
    prediction, target = temperature_fields(artifact)
    count, _, nodes = prediction.shape
    if artifact["fixed_hydrodynamics_physical"].shape != (count, nodes, 4):
        raise ValueError("fixed hydrodynamic fields differ from temperature fields")
    if artifact["fluid_internal_mass_flux_kg_s"].shape != (
        count,
        len(geometry.fluid_mesh.internal_owner),
    ):
        raise ValueError("fixed internal mass flux differs from the residual geometry")
    if artifact["fluid_boundary_mass_flux_kg_s"].shape != (
        count,
        len(geometry.fluid_mesh.boundary_owner),
    ):
        raise ValueError("fixed boundary mass flux differs from the residual geometry")
    expected_nodes = len(geometry.fluid_global_region) + len(geometry.solid_global_region)
    if nodes != expected_nodes:
        raise ValueError("prediction nodes differ from residual geometry")
    rows = []
    totals = {
        "prediction_fluid_mse": 0.0,
        "prediction_solid_mse": 0.0,
        "prediction_combined_mse": 0.0,
        "prediction_fluid_volume_weighted_mse": 0.0,
        "prediction_solid_volume_weighted_mse": 0.0,
        "prediction_volume_weighted_mse": 0.0,
        "prediction_global_closure_mse": 0.0,
        "prediction_local_l1_mse": 0.0,
        "reference_fluid_mse": 0.0,
        "reference_solid_mse": 0.0,
        "reference_combined_mse": 0.0,
        "reference_fluid_volume_weighted_mse": 0.0,
        "reference_solid_volume_weighted_mse": 0.0,
        "reference_volume_weighted_mse": 0.0,
        "reference_global_closure_mse": 0.0,
        "reference_local_l1_mse": 0.0,
        "projection_difference_fluid_mse": 0.0,
        "projection_difference_solid_mse": 0.0,
        "projection_difference_combined_mse": 0.0,
        "projection_difference_fluid_volume_weighted_mse": 0.0,
        "projection_difference_solid_volume_weighted_mse": 0.0,
        "projection_difference_volume_weighted_mse": 0.0,
        "projection_difference_global_closure_mse": 0.0,
        "projection_difference_local_l1_mse": 0.0,
    }
    for index in range(count):
        condition = torch.as_tensor(
            artifact["condition_physical"][index : index + 1],
            dtype=torch.float32,
            device=device,
        )
        time_s = torch.as_tensor(
            artifact["time_s"][index : index + 1],
            dtype=torch.float32,
            device=device,
        )
        case = {}
        residuals = {}
        for name, temperature in (("prediction", prediction), ("reference", target)):
            state = torch.as_tensor(
                physical_state(
                    temperature[index],
                    artifact["fixed_hydrodynamics_physical"][index],
                )[None],
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                residual = assemble_p418_transient_regional_residual(
                    geometry=geometry,
                    step_condition=condition,
                    state_physical=state,
                    time_s=time_s,
                    fluid_internal_mass_flux_kg_s=torch.as_tensor(
                        artifact["fluid_internal_mass_flux_kg_s"][
                            index : index + 1
                        ],
                        dtype=torch.float32,
                        device=device,
                    ),
                    fluid_boundary_mass_flux_kg_s=torch.as_tensor(
                        artifact["fluid_boundary_mass_flux_kg_s"][
                            index : index + 1
                        ],
                        dtype=torch.float32,
                        device=device,
                    ),
                )
                residuals[name] = residual
                case[name] = residual_components(
                    residual,
                    condition,
                    geometry.fluid_mesh.cell_volume,
                    geometry.solid_mesh.cell_volume,
                )
        case["projection_difference"] = residual_field_components(
            residuals["prediction"].fluid_energy_w_m3
            - residuals["reference"].fluid_energy_w_m3,
            residuals["prediction"].solid_energy_w_m3
            - residuals["reference"].solid_energy_w_m3,
            condition,
            geometry.fluid_mesh.cell_volume,
            geometry.solid_mesh.cell_volume,
        )
        row = {
            "sequence_id": str(artifact["sequence_id"][index]),
            "prediction_fluid_energy_RMSE": math.sqrt(case["prediction"]["fluid_mse"]),
            "prediction_solid_energy_RMSE": math.sqrt(case["prediction"]["solid_mse"]),
            "prediction_combined_energy_RMSE": math.sqrt(
                case["prediction"]["combined_mse"]
            ),
            "openfoam_reference_fluid_energy_RMSE": math.sqrt(
                case["reference"]["fluid_mse"]
            ),
            "openfoam_reference_solid_energy_RMSE": math.sqrt(
                case["reference"]["solid_mse"]
            ),
            "openfoam_reference_combined_energy_RMSE": math.sqrt(
                case["reference"]["combined_mse"]
            ),
            "prediction_volume_weighted_energy_RMSE": math.sqrt(
                case["prediction"]["volume_weighted_mse"]
            ),
            "openfoam_reference_volume_weighted_energy_RMSE": math.sqrt(
                case["reference"]["volume_weighted_mse"]
            ),
            "prediction_global_energy_closure_RMSE": math.sqrt(
                case["prediction"]["global_closure_mse"]
            ),
            "openfoam_reference_global_energy_closure_RMSE": math.sqrt(
                case["reference"]["global_closure_mse"]
            ),
            "prediction_local_energy_l1_RMSE": math.sqrt(
                case["prediction"]["local_l1_mse"]
            ),
            "openfoam_reference_local_energy_l1_RMSE": math.sqrt(
                case["reference"]["local_l1_mse"]
            ),
            "projection_aware_fluid_energy_RMSE": math.sqrt(
                case["projection_difference"]["fluid_mse"]
            ),
            "projection_aware_solid_energy_RMSE": math.sqrt(
                case["projection_difference"]["solid_mse"]
            ),
            "projection_aware_combined_energy_RMSE": math.sqrt(
                case["projection_difference"]["combined_mse"]
            ),
            "projection_aware_volume_weighted_energy_RMSE": math.sqrt(
                case["projection_difference"]["volume_weighted_mse"]
            ),
            "projection_aware_global_energy_closure_RMSE": math.sqrt(
                case["projection_difference"]["global_closure_mse"]
            ),
            "projection_aware_local_energy_l1_RMSE": math.sqrt(
                case["projection_difference"]["local_l1_mse"]
            ),
        }
        if "endpoint_novelty_class" in artifact:
            row["endpoint_novelty_class"] = str(
                artifact["endpoint_novelty_class"][index]
            )
        tiny = np.finfo(np.float64).tiny
        row["prediction_to_openfoam_energy_residual_ratio"] = (
            row["prediction_combined_energy_RMSE"]
            / max(row["openfoam_reference_combined_energy_RMSE"], tiny)
        )
        row["prediction_to_openfoam_volume_weighted_energy_ratio"] = (
            row["prediction_volume_weighted_energy_RMSE"]
            / max(row["openfoam_reference_volume_weighted_energy_RMSE"], tiny)
        )
        rows.append(row)
        for source_name in ("prediction", "reference", "projection_difference"):
            for component in (
                "fluid",
                "solid",
                "combined",
                "fluid_volume_weighted",
                "solid_volume_weighted",
                "volume_weighted",
                "global_closure",
                "local_l1",
            ):
                totals[f"{source_name}_{component}_mse"] += case[source_name][
                    f"{component}_mse"
                ]
    for key in totals:
        totals[key] /= count
    summary = {
        "curve_count": count,
        "prediction_fluid_energy_equation_normalized_RMSE": math.sqrt(
            totals["prediction_fluid_mse"]
        ),
        "prediction_solid_energy_equation_normalized_RMSE": math.sqrt(
            totals["prediction_solid_mse"]
        ),
        "prediction_combined_energy_equation_normalized_RMSE": math.sqrt(
            totals["prediction_combined_mse"]
        ),
        "openfoam_reference_fluid_energy_equation_normalized_RMSE": math.sqrt(
            totals["reference_fluid_mse"]
        ),
        "openfoam_reference_solid_energy_equation_normalized_RMSE": math.sqrt(
            totals["reference_solid_mse"]
        ),
        "openfoam_reference_combined_energy_equation_normalized_RMSE": math.sqrt(
            totals["reference_combined_mse"]
        ),
        "prediction_fluid_volume_weighted_energy_equation_normalized_RMSE": math.sqrt(
            totals["prediction_fluid_volume_weighted_mse"]
        ),
        "prediction_solid_volume_weighted_energy_equation_normalized_RMSE": math.sqrt(
            totals["prediction_solid_volume_weighted_mse"]
        ),
        "prediction_volume_weighted_energy_equation_normalized_RMSE": math.sqrt(
            totals["prediction_volume_weighted_mse"]
        ),
        "openfoam_reference_fluid_volume_weighted_energy_equation_normalized_RMSE": math.sqrt(
            totals["reference_fluid_volume_weighted_mse"]
        ),
        "openfoam_reference_solid_volume_weighted_energy_equation_normalized_RMSE": math.sqrt(
            totals["reference_solid_volume_weighted_mse"]
        ),
        "openfoam_reference_volume_weighted_energy_equation_normalized_RMSE": math.sqrt(
            totals["reference_volume_weighted_mse"]
        ),
        "prediction_global_energy_closure_normalized_RMSE": math.sqrt(
            totals["prediction_global_closure_mse"]
        ),
        "openfoam_reference_global_energy_closure_normalized_RMSE": math.sqrt(
            totals["reference_global_closure_mse"]
        ),
        "prediction_local_energy_l1_normalized_RMSE": math.sqrt(
            totals["prediction_local_l1_mse"]
        ),
        "openfoam_reference_local_energy_l1_normalized_RMSE": math.sqrt(
            totals["reference_local_l1_mse"]
        ),
        "projection_aware_fluid_energy_equation_normalized_RMSE": math.sqrt(
            totals["projection_difference_fluid_mse"]
        ),
        "projection_aware_solid_energy_equation_normalized_RMSE": math.sqrt(
            totals["projection_difference_solid_mse"]
        ),
        "projection_aware_combined_energy_equation_normalized_RMSE": math.sqrt(
            totals["projection_difference_combined_mse"]
        ),
        "projection_aware_volume_weighted_energy_equation_normalized_RMSE": math.sqrt(
            totals["projection_difference_volume_weighted_mse"]
        ),
        "projection_aware_global_energy_closure_normalized_RMSE": math.sqrt(
            totals["projection_difference_global_closure_mse"]
        ),
        "projection_aware_local_energy_l1_normalized_RMSE": math.sqrt(
            totals["projection_difference_local_l1_mse"]
        ),
    }
    tiny = np.finfo(np.float64).tiny
    summary["prediction_to_openfoam_energy_residual_ratio"] = (
        summary["prediction_combined_energy_equation_normalized_RMSE"]
        / max(
            summary["openfoam_reference_combined_energy_equation_normalized_RMSE"],
            tiny,
        )
    )
    summary["prediction_to_openfoam_volume_weighted_energy_residual_ratio"] = (
        summary["prediction_volume_weighted_energy_equation_normalized_RMSE"]
        / max(
            summary[
                "openfoam_reference_volume_weighted_energy_equation_normalized_RMSE"
            ],
            tiny,
        )
    )
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-summary", type=Path, required=True)
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--residual-geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--roles", nargs="+", choices=ROLES, default=list(ROLES))
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model_summary_path = args.model_summary.resolve()
    model_summary = json.loads(model_summary_path.read_text(encoding="utf-8"))
    index = json.loads(args.dataset_index.resolve().read_text(encoding="utf-8"))
    geometry = load_p418_subface_geometry(
        args.residual_geometry.resolve(),
        fluid_patch_names=index["boundary_patch_names"]["fluid"],
        solid_patch_names=index["boundary_patch_names"]["solid"],
        device=device,
        dtype=torch.float32,
    )
    roles = tuple(args.roles)
    if len(roles) != len(set(roles)):
        raise ValueError("energy-evaluation roles are repeated")
    files = prediction_files(model_summary, roles)
    role_summaries = {}
    role_failures = {}
    case_rows = {}
    endpoint_novelty_metrics = {}
    for role in roles:
        path = model_summary_path.parent / files[role]
        with np.load(path, allow_pickle=False) as loaded:
            artifact = {name: loaded[name].copy() for name in loaded.files}
        range_diagnostics = registered_temperature_range_diagnostics(artifact)
        if not range_diagnostics[
            "reference_within_registered_thermophysical_range"
        ]:
            raise ValueError(
                f"OpenFOAM reference for {role} is outside the registered "
                "thermophysical range"
            )
        if not range_diagnostics[
            "prediction_within_registered_thermophysical_range"
        ]:
            role_failures[role] = {
                "status": "prediction_outside_registered_thermophysical_range",
                **range_diagnostics,
            }
            continue
        role_summaries[role], case_rows[role] = summarize_role(
            artifact=artifact, geometry=geometry, device=device
        )
        grouped = summarize_endpoint_energy_groups(case_rows[role])
        if grouped:
            endpoint_novelty_metrics[role] = grouped
    output = {
        "status": (
            "completed_p418_common_transient_energy_balance"
            if not role_failures
            else "completed_p418_common_transient_energy_balance_with_rejected_roles"
        ),
        "model_summary": str(model_summary_path),
        "model_status": model_summary.get("status"),
        "split_name": model_summary.get("split_name"),
        "dataset_index": str(args.dataset_index.resolve()),
        "residual_geometry": str(args.residual_geometry.resolve()),
        "compute_device": str(device),
        "requested_roles": list(roles),
        "evaluated_roles": list(role_summaries),
        "rejected_roles": list(role_failures),
        "normalization": (
            "fluid and solid residuals divided by target volumetric heat source; "
            "primary model comparison additionally weights every regional residual by "
            "its physical volume"
        ),
        "primary_physical_energy_metric": PRIMARY_ENERGY_METRIC,
        "projection_aware_definition": (
            "regional prediction equation field minus the equation field obtained by "
            "projecting the OpenFOAM reference onto the same regional geometry"
        ),
        "absolute_residual_role": (
            "prediction and OpenFOAM absolute regional residuals are reported only to "
            "quantify projection/discretization effects; native OpenFOAM solver balances "
            "remain the independent conservation reference"
        ),
        "global_closure_definition": (
            "RMS of the volume-integrated fluid-plus-solid residual divided by target "
            "particle heat generation"
        ),
        "role_metrics": role_summaries,
        "role_failures": role_failures,
        "endpoint_novelty_metrics": endpoint_novelty_metrics,
        "per_curve_metrics": case_rows,
        "new_physical_parameters": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
