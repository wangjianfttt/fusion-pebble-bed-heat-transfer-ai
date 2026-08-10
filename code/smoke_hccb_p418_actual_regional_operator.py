#!/usr/bin/env python3
"""Execute graph and Physics-Attention blocks on the actual P418 hierarchy."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from hccb_p418_parametric_regional_operator import (
    HCCBP418ParametricRegionalOperator,
    collapse_mesh_to_level,
    load_p418_regional_mesh,
)


def exercise(
    *,
    name: str,
    mesh,
    processor_kind: str,
    fine_cells: int,
    hidden_dim: int,
    processor_steps: int,
    attention_heads: int,
    physics_slices: int,
) -> dict[str, object]:
    model = HCCBP418ParametricRegionalOperator(
        boundary_role_count=int(mesh.fine_boundary_role.shape[1]),
        hidden_dim=hidden_dim,
        processor_steps=processor_steps,
        active_levels=1,
        processor_kind=processor_kind,
        attention_heads=attention_heads,
        attention_start_level=0,
        physics_slices=physics_slices,
    )
    condition = torch.zeros((1, 5), requires_grad=True)
    started = time.time()
    regional = model.encode_regions(condition, mesh)
    stop = min(fine_cells, mesh.n_fine)
    prediction = model.decode_fine_chunk(condition, regional, mesh, 0, stop)
    prediction.square().mean().backward()
    elapsed = time.time() - started
    finite = bool(
        torch.isfinite(prediction).all()
        and condition.grad is not None
        and torch.isfinite(condition.grad).all()
    )
    return {
        "name": name,
        "processor_kind": processor_kind,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "regional_output_shape": list(regional.shape),
        "decoded_fine_shape": list(prediction.shape),
        "forward_backward_seconds": elapsed,
        "finite_prediction_and_gradient": finite,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regional-topology", type=Path, required=True)
    parser.add_argument("--model-geometry", type=Path, required=True)
    parser.add_argument("--regional-level", type=int, default=5)
    parser.add_argument("--fine-cells", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--source-width", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    full = load_p418_regional_mesh(
        args.regional_topology.resolve(), args.model_geometry.resolve()
    )
    mesh = collapse_mesh_to_level(full, args.regional_level)
    level = mesh.levels[0]
    if args.source_width:
        graph_settings = (128, 12, 8, 32)
        attention_settings = (256, 5, 8, 32)
        execution_scope = "source-width architecture execution"
        graph_name = "regional_graph_source_width"
        attention_name = "regional_physics_attention_source_width"
    else:
        graph_settings = (16, 1, 4, 4)
        attention_settings = (16, 1, 4, 4)
        execution_scope = "reduced-width software execution"
        graph_name = "regional_graph_reduced_width"
        attention_name = "regional_physics_attention_reduced_width"
    payload = {
        "status": "actual_mesh_software_execution",
        "scope": (
            f"{execution_scope} on the actual regional graph. "
            "This is a software and memory-path check, not a trained accuracy result."
        ),
        "mesh": {
            "fine_fluid_solid_cells": mesh.n_fine,
            "regional_level": args.regional_level,
            "regional_nodes": len(level.node_type),
            "directed_regional_edges": len(level.edge_source),
            "decoded_cells_per_model": min(args.fine_cells, mesh.n_fine),
        },
        "models": [
            exercise(
                name=graph_name,
                mesh=mesh,
                processor_kind="message_passing",
                fine_cells=args.fine_cells,
                hidden_dim=graph_settings[0],
                processor_steps=graph_settings[1],
                attention_heads=graph_settings[2],
                physics_slices=graph_settings[3],
            ),
            exercise(
                name=attention_name,
                mesh=mesh,
                processor_kind="hybrid_physics_attention",
                fine_cells=args.fine_cells,
                hidden_dim=attention_settings[0],
                processor_steps=attention_settings[1],
                attention_heads=attention_settings[2],
                physics_slices=attention_settings[3],
            ),
        ],
    }
    all_finite = all(
        bool(item["finite_prediction_and_gradient"]) for item in payload["models"]
    )
    payload["all_checks_pass"] = all_finite
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all_finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
