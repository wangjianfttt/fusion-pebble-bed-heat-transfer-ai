#!/usr/bin/env python3
"""Apply a trained diffusion refiner after the steady-PINN/graph-Transformer chain."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from hccb_p418_temporal_temperature_diffusion import (
    P418TemporalTemperatureResidualRefiner,
    sample_temporal_temperature_residual,
)
from evaluate_hccb_p418_chained_initial_state import verify_registered_file
from hccb_p418_chain_roles import DETERMINISTIC_CHAIN_STATUS, FUSED_CHAIN_STATUS
from train_hccb_p418_temporal_temperature_diffusion import (
    ensemble_crps_k,
    finalized_interval_metrics,
    finalized_selected_mean,
    interval_weighted_sums,
    load_predictions,
    temperature_rmse_k,
    weighted_selected_sums,
)


def ensemble_temperature_metrics(
    *,
    members: np.ndarray,
    target: np.ndarray,
    node_type: np.ndarray,
    node_volume: np.ndarray,
    temperature_std_by_type: np.ndarray,
    observation_mask: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Return volume-weighted ensemble-mean error and interval diagnostics."""
    if members.ndim != 5 or target.shape != members.shape[1:]:
        raise ValueError("members must be [sample,curve,time,node,1]")
    mean = members.mean(axis=0)
    lower = np.quantile(members, 0.05, axis=0)
    upper = np.quantile(members, 0.95, axis=0)
    scale = temperature_std_by_type[node_type]
    volume = node_volume[None, None, :]
    covered = (target[..., 0] >= lower[..., 0]) & (
        target[..., 0] <= upper[..., 0]
    )
    width_k = (upper[..., 0] - lower[..., 0]) * scale[None, None, :]
    predictive_std_k = members.std(axis=0)[..., 0] * scale[None, None, :]
    crps_k = ensemble_crps_k(members, target, scale)
    denominator = target.shape[0] * target.shape[1] * node_volume.sum()
    metrics = {
        "ensemble_mean_temperature_RMSE_K": temperature_rmse_k(
            mean, target, node_type, node_volume, temperature_std_by_type
        ),
        "ensemble_mean_fluid_temperature_RMSE_K": temperature_rmse_k(
            mean,
            target,
            node_type,
            node_volume,
            temperature_std_by_type,
            material=0,
        ),
        "ensemble_mean_solid_temperature_RMSE_K": temperature_rmse_k(
            mean,
            target,
            node_type,
            node_volume,
            temperature_std_by_type,
            material=1,
        ),
        "interval_90pct_coverage_fraction": float(
            np.sum(covered * volume) / denominator
        ),
        "interval_90pct_mean_width_K": float(
            np.sum(width_k * volume) / denominator
        ),
    }
    if observation_mask is None:
        observation_mask = np.zeros_like(target, dtype=bool)
    if observation_mask.shape != target.shape or observation_mask.dtype != bool:
        raise ValueError("observation mask must be boolean and match the target")
    unobserved_dynamic = ~observation_mask[..., 0].copy()
    unobserved_dynamic[:, 0, :] = False
    for material_name, material_selection in (
        ("", np.ones_like(node_type, dtype=bool)),
        ("_fluid", node_type == 0),
        ("_solid", node_type == 1),
    ):
        selected = unobserved_dynamic & material_selection[None, None, :]
        sums = interval_weighted_sums(
            covered=covered,
            width_k=width_k,
            predictive_std_k=predictive_std_k,
            node_volume=node_volume,
            selection=selected,
        )
        metrics.update(
            finalized_interval_metrics(
                sums,
                prefix=f"unobserved_dynamic{material_name}",
            )
        )
        metrics[f"unobserved_dynamic{material_name}_CRPS_K"] = finalized_selected_mean(
            weighted_selected_sums(crps_k, node_volume, selected)
        )
    return metrics


def complete_chain_timing(
    *,
    chained_summary: dict[str, object],
    diffusion_seconds: float,
    curve_count: int,
) -> dict[str, float | str]:
    """Combine measured steady, graph--Transformer and diffusion inference times."""
    if curve_count <= 0:
        raise ValueError("curve count must be positive")
    timing = chained_summary.get("timing")
    if not isinstance(timing, dict):
        raise ValueError("chained summary lacks complete inference timing")
    graph_seconds = float(timing.get("graph_transformer_inference_seconds", math.nan))
    warm_deterministic = float(
        timing.get("warm_start_deterministic_chain_inference_seconds", math.nan)
    )
    cold_deterministic = float(
        timing.get("cold_start_deterministic_chain_inference_seconds", math.nan)
    )
    for name, value in (
        ("graph--Transformer inference", graph_seconds),
        ("warm deterministic chain", warm_deterministic),
        ("cold deterministic chain", cold_deterministic),
        ("diffusion inference", diffusion_seconds),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid {name} time: {value}")
    if not math.isclose(graph_seconds, warm_deterministic, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("warm deterministic timing differs from graph--Transformer timing")
    warm_total = warm_deterministic + diffusion_seconds
    cold_total = cold_deterministic + diffusion_seconds
    return {
        "graph_transformer_inference_seconds": graph_seconds,
        "graph_transformer_inference_seconds_per_curve": graph_seconds / curve_count,
        "diffusion_refiner_inference_seconds": diffusion_seconds,
        "diffusion_refiner_inference_seconds_per_curve": diffusion_seconds / curve_count,
        "warm_start_complete_chain_inference_seconds": warm_total,
        "warm_start_complete_chain_inference_seconds_per_curve": warm_total / curve_count,
        "cold_start_complete_chain_inference_seconds": cold_total,
        "cold_start_complete_chain_inference_seconds_per_curve": cold_total / curve_count,
        "definition": (
            "Warm start includes graph--Transformer and diffusion inference with steady endpoint "
            "fields already available. Cold start additionally includes registered measured "
            "steady-PINN inference for the unique endpoint conditions in this trajectory set."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chained-summary", type=Path, required=True)
    parser.add_argument("--diffusion-summary", type=Path, required=True)
    parser.add_argument("--role", choices=("validation", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    chained_summary_path = args.chained_summary.resolve()
    diffusion_summary_path = args.diffusion_summary.resolve()
    chained_summary = json.loads(chained_summary_path.read_text(encoding="utf-8"))
    diffusion_summary = json.loads(diffusion_summary_path.read_text(encoding="utf-8"))
    if chained_summary.get("status") != DETERMINISTIC_CHAIN_STATUS:
        raise ValueError("chained deterministic prediction is incomplete")
    if diffusion_summary.get("status") != "completed_p418_temporal_temperature_diffusion":
        raise ValueError("trained diffusion refiner is incomplete")
    if chained_summary.get("role") != args.role:
        raise ValueError("chained deterministic summary uses a different role")
    if chained_summary.get("transient_split_name") != diffusion_summary.get("split_name"):
        raise ValueError("chained and diffusion models use different transient splits")
    if int(diffusion_summary["metrics"][args.role].get("observation_count", -1)) != 0:
        raise ValueError("chained diffusion evaluation currently requires no hidden observations")

    chained_file = chained_summary_path.parent / chained_summary["prediction_files"][args.role]
    chained_records = chained_summary.get("prediction_file_records")
    if not isinstance(chained_records, dict):
        raise ValueError("chained deterministic summary does not record prediction files")
    verify_registered_file(
        chained_file,
        chained_records.get(args.role),
        f"chained deterministic {args.role} prediction",
    )
    data = load_predictions(chained_file)
    expected_ids = [str(value) for value in diffusion_summary["split_case_ids"][args.role]]
    if [str(value) for value in data["sequence_id"]] != expected_ids:
        raise ValueError("chained and diffusion test curves differ")
    deterministic_dir = Path(diffusion_summary["deterministic_prediction_dir"]).resolve()
    if Path(chained_summary["upstream_exact_initial_prediction_file"]).resolve().parent != deterministic_dir:
        raise ValueError("chained predictions and diffusion training use different deterministic models")

    maximum_times = []
    for role in ("train", "validation", "test"):
        with np.load(
            deterministic_dir / f"{role}_temporal_temperature_predictions.npz",
            allow_pickle=False,
        ) as loaded:
            maximum_times.append(float(loaded["time_s"].max()))
    maximum_time_s = max(maximum_times)
    node_type = data["node_type"].astype(np.int64)
    node_volume = data["node_volume_m3"].astype(np.float32)
    structure = data["structural_features"].astype(np.float32)
    temperature_std = data["temperature_std_K_by_node_type"].astype(np.float32)
    baseline = torch.as_tensor(
        data["baseline_temperature_normalized"], dtype=torch.float32, device=device
    )
    condition = torch.as_tensor(
        data["condition_normalized"], dtype=torch.float32, device=device
    )
    normalized_time = torch.as_tensor(
        (data["time_s"] / maximum_time_s).astype(np.float32), device=device
    )
    structure_tensor = torch.as_tensor(structure, device=device)
    observed = torch.zeros_like(baseline)
    mask = torch.zeros_like(baseline, dtype=torch.bool)

    architecture = diffusion_summary["architecture"]
    model = P418TemporalTemperatureResidualRefiner(
        structural_dim=structure.shape[1],
        condition_dim=condition.shape[1],
        hidden_dim=int(architecture["hidden_dim"]),
        spatial_layers=int(architecture["spatial_layers"]),
        spatial_attention_heads=int(architecture["spatial_attention_heads"]),
        physics_slices=int(architecture["physics_slices"]),
        temporal_layers=int(architecture["temporal_layers"]),
        temporal_heads=int(architecture["temporal_heads"]),
        num_refinement_steps=int(architecture["refinement_steps"]),
    ).to(device)
    model.load_state_dict(
        torch.load(
            diffusion_summary_path.parent / "ema_model_state.pt",
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()
    scale = float(diffusion_summary["residual_scale_in_normalized_temperature"])
    ensemble_samples = int(diffusion_summary["ensemble_samples"])
    seed = int(diffusion_summary["seed"])
    activation_precision = str(diffusion_summary["activation_precision"])
    members_by_sample: list[list[np.ndarray]] = [
        [] for _ in range(ensemble_samples)
    ]
    started = time.perf_counter()
    for case_index in range(len(data["sequence_id"])):
        baseline_case = baseline[case_index : case_index + 1]
        condition_case = condition[case_index : case_index + 1]
        time_case = normalized_time[case_index : case_index + 1]
        observed_case = observed[case_index : case_index + 1]
        mask_case = mask[case_index : case_index + 1]
        for sample_index in range(ensemble_samples):
            sample_seed = seed + 1000 + case_index * ensemble_samples + sample_index
            noise = torch.randn(
                baseline_case.shape,
                generator=torch.Generator(device=device).manual_seed(sample_seed),
                device=device,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=activation_precision == "bfloat16",
            ):
                residual = sample_temporal_temperature_residual(
                    model,
                    baseline_case,
                    condition_case,
                    structure_tensor,
                    time_case,
                    observed_case,
                    mask_case,
                    initial_noise=noise,
                    num_refinement_steps=int(architecture["refinement_steps"]),
                )
            members_by_sample[sample_index].append(
                (baseline_case + scale * residual).cpu().numpy()[0]
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    members = np.stack(
        [np.stack(values, axis=0) for values in members_by_sample], axis=0
    )
    target = data["target_temperature_normalized"].astype(np.float32)
    refined = members.mean(axis=0).astype(np.float32)
    metrics = ensemble_temperature_metrics(
        members=members,
        target=target,
        node_type=node_type,
        node_volume=node_volume,
        temperature_std_by_type=temperature_std,
        observation_mask=mask.cpu().numpy(),
    )
    timing = complete_chain_timing(
        chained_summary=chained_summary,
        diffusion_seconds=inference_seconds,
        curve_count=len(data["sequence_id"]),
    )
    deterministic_cost = chained_summary.get("model_cost")
    if not isinstance(deterministic_cost, dict):
        raise ValueError("chained summary lacks deterministic model cost")
    diffusion_parameters = int(diffusion_summary.get("model_parameter_count", 0))
    diffusion_training_seconds = float(
        diffusion_summary.get("training_seconds", math.nan)
    )
    if diffusion_parameters <= 0:
        raise ValueError("diffusion model parameter count is missing")
    if not math.isfinite(diffusion_training_seconds) or diffusion_training_seconds <= 0.0:
        raise ValueError("diffusion measured training time is missing")
    model_cost = {
        **deterministic_cost,
        "diffusion_refiner_model_parameter_count": diffusion_parameters,
        "diffusion_refiner_training_seconds": diffusion_training_seconds,
        "complete_chain_model_parameter_count": (
            int(deterministic_cost["deterministic_chain_model_parameter_count"])
            + diffusion_parameters
        ),
        "complete_chain_training_seconds": (
            float(deterministic_cost["deterministic_chain_training_seconds"])
            + diffusion_training_seconds
        ),
    }
    metrics.update(
        {
            "chained_deterministic_temperature_RMSE_K": temperature_rmse_k(
                data["baseline_temperature_normalized"],
                target,
                node_type,
                node_volume,
                temperature_std,
            ),
            "chained_deterministic_solid_temperature_RMSE_K": temperature_rmse_k(
                data["baseline_temperature_normalized"],
                target,
                node_type,
                node_volume,
                temperature_std,
                material=1,
            ),
            "inference_seconds": inference_seconds,
            "inference_seconds_per_curve": inference_seconds / len(data["sequence_id"]),
            **timing,
        }
    )
    if "endpoint_novelty_class" not in data:
        raise ValueError("chained prediction lacks steady-endpoint novelty labels")
    endpoint_labels = data["endpoint_novelty_class"].astype(str)
    endpoint_novelty_metrics: dict[str, dict[str, float | None]] = {}
    for label in sorted(set(endpoint_labels)):
        selected = endpoint_labels == label
        selected_metrics = ensemble_temperature_metrics(
            members=members[:, selected],
            target=target[selected],
            node_type=node_type,
            node_volume=node_volume,
            temperature_std_by_type=temperature_std,
            observation_mask=mask.cpu().numpy()[selected],
        )
        selected_metrics.update(
            {
                "curve_count": int(selected.sum()),
                "chained_deterministic_solid_temperature_RMSE_K": temperature_rmse_k(
                    data["baseline_temperature_normalized"][selected],
                    target[selected],
                    node_type,
                    node_volume,
                    temperature_std,
                    material=1,
                ),
            }
        )
        endpoint_novelty_metrics[label] = selected_metrics

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / f"{args.role}_chained_diffusion_temperature.npz"
    np.savez_compressed(
        prediction_path,
        **{
            **data,
            "chained_deterministic_temperature_normalized": data[
                "baseline_temperature_normalized"
            ],
            "refined_temperature_normalized": refined,
            "refined_temperature_std_normalized": members.std(axis=0).astype(np.float32),
        },
    )
    summary = {
        "status": FUSED_CHAIN_STATUS,
        "split_name": diffusion_summary["split_name"],
        "role": args.role,
        "seed": seed,
        "ensemble_samples": ensemble_samples,
        "chained_deterministic_summary": str(chained_summary_path),
        "trained_diffusion_summary": str(diffusion_summary_path),
        "prediction_files": {args.role: prediction_path.name},
        "metrics": {args.role: metrics},
        "endpoint_novelty_metrics": endpoint_novelty_metrics,
        "complete_chain_timing": timing,
        "complete_chain_model_cost": model_cost,
        "compute_device": str(device),
        "new_physical_parameters": [],
        "scientific_scope": (
            "The already-trained diffusion refiner is applied to held-out trajectories "
            "generated from a steady-PINN initial field and the transient graph--Transformer. "
            "No test target is used to retrain either model."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
