#!/usr/bin/env python3
"""Measure formal temporal-diffusion memory on the actual P418 regional graph."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from build_hccb_p418_step_response_cases import field_write_stages
from hccb_p418_regional_diffusion_refiner import make_velocity_training_pair
from hccb_p418_spatiotemporal_regional_operator import P418ThermalStepRegionalGraph
from hccb_p418_temporal_temperature_diffusion import (
    FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE,
    P418TemporalTemperatureResidualRefiner,
)
from smoke_hccb_p418_actual_spatiotemporal_operator import (
    sha256,
    validate_graph_provenance,
    write_model_graph,
)
from train_hccb_p418_temporal_temperature_diffusion import weighted_velocity_loss


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subface-geometry", type=Path, required=True)
    parser.add_argument("--regional-topology", type=Path, required=True)
    parser.add_argument("--step-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--precision", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--spatial-time-chunk-size", type=int, default=1)
    parser.add_argument("--temporal-node-chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.precision == "bfloat16" and args.device != "cuda":
        raise ValueError("the bfloat16 resource test is restricted to CUDA")
    if args.torch_threads <= 0:
        raise ValueError("torch threads must be positive")
    if args.spatial_time_chunk_size <= 0 or args.temporal_node_chunk_size <= 0:
        raise ValueError("diffusion chunk sizes must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_provenance = validate_graph_provenance(
        args.subface_geometry.resolve(), args.regional_topology.resolve()
    )
    graph_path = args.output_dir / "regional_sequence_geometry.npz"
    graph_summary = write_model_graph(
        args.subface_geometry.resolve(),
        args.regional_topology.resolve(),
        graph_path,
    )
    graph = P418ThermalStepRegionalGraph.from_npz(graph_path, device=device)
    structure = graph.structural_features()

    plan = json.loads(args.step_plan.resolve().read_text(encoding="utf-8"))
    numerical = plan["numerical_time_design"]
    duration = float(numerical["duration_s"])
    _, snapshot_times = field_write_stages(numerical)
    normalized_time = torch.tensor(
        np.asarray(snapshot_times, dtype=np.float32) / duration,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    time_count = len(snapshot_times)
    node_count = graph.node_count

    # A finite residual is sufficient for a resource test.  It is deliberately
    # dimensionless and is never reported as a temperature prediction.
    time_shape = normalized_time[:, :, None, None]
    spatial_shape = structure[:, :1][None, None]
    clean_residual = 0.01 * time_shape * (1.0 + 0.1 * spatial_shape)
    baseline = torch.zeros_like(clean_residual)
    condition = torch.zeros((1, 8), dtype=torch.float32, device=device)
    observed = torch.zeros_like(clean_residual)
    observation_mask = torch.zeros_like(clean_residual, dtype=torch.bool)
    refinement_step = torch.ones((1,), dtype=torch.long, device=device)
    noise = torch.randn_like(clean_residual)
    noised, target_velocity = make_velocity_training_pair(
        clean_residual, refinement_step, noise=noise
    )

    model = P418TemporalTemperatureResidualRefiner(
        structural_dim=structure.shape[1],
        hidden_dim=FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE["hidden_dim"],
        spatial_layers=FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE["spatial_layers"],
        spatial_attention_heads=FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE[
            "spatial_attention_heads"
        ],
        physics_slices=FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE["physics_slices"],
        temporal_layers=FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE["temporal_layers"],
        temporal_heads=FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE["temporal_heads"],
        spatial_time_chunk_size=args.spatial_time_chunk_size,
        temporal_node_chunk_size=args.temporal_node_chunk_size,
    ).to(device).train()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=args.precision == "bfloat16",
    ):
        prediction = model(
            baseline,
            noised,
            condition,
            structure,
            normalized_time,
            observed,
            observation_mask,
            refinement_step,
        )
        loss = weighted_velocity_loss(prediction, target_velocity, graph.volume_m3)
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    peak_gpu_gb = (
        torch.cuda.max_memory_allocated() / 1.0e9 if device.type == "cuda" else None
    )
    summary = {
        "status": "formal_actual_graph_temporal_diffusion_backward_passed",
        **graph_summary,
        "time_points": time_count,
        "curve_batch_size": 1,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": elapsed,
        "peak_gpu_GB": peak_gpu_gb,
        "compute_device": str(device),
        "activation_precision": args.precision,
        "spatial_time_chunk_size": args.spatial_time_chunk_size,
        "temporal_node_chunk_size": args.temporal_node_chunk_size,
        "torch_num_threads": torch.get_num_threads(),
        "input_provenance": input_provenance,
        "step_plan_sha256": sha256(args.step_plan.resolve()),
        "loss_finite": bool(torch.isfinite(loss).detach().cpu()),
        "trainable_tensor_count": len(gradients),
        "all_gradients_present": all(value is not None for value in gradients),
        "all_gradients_finite": all(
            value is not None and torch.isfinite(value).all().item()
            for value in gradients
        ),
        "architecture": FORMAL_TEMPORAL_DIFFUSION_ARCHITECTURE,
        "physical_parameter_ids": [
            "P388",
            "P389",
            "P403",
            "P418",
            "P428",
            "P429",
            "P430",
            "P431",
        ],
        "new_physical_parameters": [],
        "scientific_scope": (
            "GPU memory and backward-pass test for one full-size curve on the "
            f"actual regional graph and the {time_count} output times declared by the supplied plan. The finite "
            "dimensionless residual is not a temperature-accuracy result."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
