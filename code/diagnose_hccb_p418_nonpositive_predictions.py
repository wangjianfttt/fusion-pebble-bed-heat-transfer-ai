#!/usr/bin/env python3
"""Locate non-positive physical states in a saved P418 model checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hccb_p418_spatiotemporal_regional_operator import (
    HCCBP418SpatiotemporalRegionalOperator,
    P418ThermalStepRegionalGraph,
)
from train_hccb_p418_spatiotemporal_regional_operator import (
    load_sequence,
    sequence_records,
    state_scale_tensors,
    training_statistics,
)


def minimum_record(
    values: torch.Tensor,
    *,
    time_s: torch.Tensor,
    graph: P418ThermalStepRegionalGraph,
    node_mask: torch.Tensor,
) -> dict[str, object]:
    selected = values[0, :, node_mask]
    flat_index = int(torch.argmin(selected).cpu())
    time_index, local_node_index = np.unravel_index(flat_index, selected.shape)
    global_node_index = int(torch.nonzero(node_mask, as_tuple=False)[local_node_index])
    return {
        "value": float(selected[time_index, local_node_index].cpu()),
        "time_s": float(time_s[0, time_index].cpu()),
        "time_index": int(time_index),
        "node_index": global_node_index,
        "node_centroid_m": [
            float(value)
            for value in graph.centroid_m[global_node_index].detach().cpu()
        ],
        "nonpositive_count": int((selected <= 0).sum().cpu()),
        "value_count": int(selected.numel()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA diagnosis requested but CUDA is unavailable")

    checkpoint = torch.load(
        args.checkpoint.resolve(), map_location=device, weights_only=False
    )
    contract = checkpoint["contract"]
    index_path = args.dataset_index.resolve()
    if Path(contract["dataset_index"]).resolve() != index_path:
        raise ValueError("checkpoint and requested dataset index differ")

    root = index_path.parent
    index = json.loads(index_path.read_text(encoding="utf-8"))
    records = sequence_records(index)
    split = {
        role: [str(value) for value in contract["split_case_ids"][role]]
        for role in ("train", "validation", "test")
    }
    graph = P418ThermalStepRegionalGraph.from_npz(
        root / str(index["regional_geometry_file"]), device=device
    )
    statistics = training_statistics(
        root,
        records,
        split["train"],
        graph.node_type.detach().cpu().numpy(),
    )
    condition_mean = torch.as_tensor(statistics["condition_mean"], device=device)
    condition_std = torch.as_tensor(statistics["condition_std"], device=device)
    state_mean, state_std = state_scale_tensors(
        statistics, graph.node_type, device
    )
    temperature_output_mode = str(
        contract.get("temperature_output_mode", "additive_normalized")
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
                    np.asarray(statistics["state_mean"])[:, 4], device=device
                ),
                "temperature_std_k_by_node_type": torch.as_tensor(
                    np.asarray(statistics["state_std"])[:, 4], device=device
                ),
                "temperature_bounds_k_by_node_type": torch.tensor(
                    ((300.0, 1000.0), (298.0, 1300.0)), device=device
                ),
            }
        )

    model = HCCBP418SpatiotemporalRegionalOperator(
        condition_dim=len(index["condition_names"]),
        hidden_dim=int(contract["hidden_dim"]),
        local_pre_iterations=int(contract["local_pre_iterations"]),
        physics_attention_blocks=int(contract["physics_attention_blocks"]),
        local_post_iterations=int(contract["local_post_iterations"]),
        physics_attention_heads=int(contract["physics_attention_heads"]),
        physics_slices=int(contract["physics_slices"]),
        temporal_layers=int(contract["temporal_layers"]),
        temporal_heads=int(contract["temporal_heads"]),
        spatial_time_chunk_size=int(contract["spatial_time_chunk_size"]),
        temporal_node_chunk_size=int(contract["temporal_node_chunk_size"]),
        spatial_temporal_mode=str(contract["spatial_temporal_mode"]),
        boundary_role_count=graph.boundary_role_count,
        **temperature_output_arguments,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    np.random.set_state(checkpoint["numpy_random_state"])
    next_training_order = [
        str(value) for value in np.random.permutation(split["train"])
    ]
    fluid_mask = graph.node_type == 0
    solid_mask = graph.node_type == 1
    report: dict[str, object] = {
        "status": "p418_checkpoint_physical_domain_diagnosis",
        "checkpoint": str(args.checkpoint.resolve()),
        "completed_epochs": int(checkpoint["next_epoch"]),
        "next_training_order": next_training_order,
        "records": [],
    }

    with torch.no_grad():
        for role in ("train", "validation", "test"):
            for sequence_id in split[role]:
                time_np, condition_np, state_np, _, _ = load_sequence(
                    root, records[sequence_id]
                )
                time_s = torch.as_tensor(time_np, device=device).unsqueeze(0)
                condition = torch.as_tensor(
                    condition_np, device=device
                ).unsqueeze(0)
                state = torch.as_tensor(state_np, device=device).unsqueeze(0)
                normalized_state = (
                    state - state_mean[None, None]
                ) / state_std[None, None]
                normalized_condition = (
                    condition - condition_mean
                ) / condition_std
                normalized_time = time_s / float(statistics["maximum_time_s"])
                prediction_norm = model(
                    normalized_state[:, 0],
                    normalized_condition,
                    normalized_time,
                    graph,
                )
                prediction = (
                    prediction_norm * state_std[None, None]
                    + state_mean[None, None]
                )
                record = {
                    "role": role,
                    "sequence_id": sequence_id,
                    "fluid_absolute_pressure_Pa": minimum_record(
                        prediction[..., 3],
                        time_s=time_s,
                        graph=graph,
                        node_mask=fluid_mask,
                    ),
                    "fluid_temperature_K": minimum_record(
                        prediction[..., 4],
                        time_s=time_s,
                        graph=graph,
                        node_mask=fluid_mask,
                    ),
                    "solid_temperature_K": minimum_record(
                        prediction[..., 4],
                        time_s=time_s,
                        graph=graph,
                        node_mask=solid_mask,
                    ),
                }
                report["records"].append(record)
                print(
                    role,
                    sequence_id,
                    "p_min=",
                    record["fluid_absolute_pressure_Pa"]["value"],
                    "Tf_min=",
                    record["fluid_temperature_K"]["value"],
                    "Ts_min=",
                    record["solid_temperature_K"]["value"],
                    flush=True,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
