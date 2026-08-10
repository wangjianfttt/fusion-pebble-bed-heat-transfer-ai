#!/usr/bin/env python3
"""Run the formal step operator and transient physics on the actual regional graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from build_hccb_p418_step_response_cases import field_write_stages
from export_hccb_p418_step_regional_sequences import matching_regional_graph
from hccb_p418_regional_cht_adapter import load_p418_subface_geometry
from hccb_p418_spatiotemporal_regional_operator import (
    HCCBP418SpatiotemporalRegionalOperator,
    P418ThermalStepRegionalGraph,
    SPATIAL_TEMPORAL_MODES,
)
from hccb_p418_transient_regional_physics import (
    assemble_p418_transient_regional_residual,
    volume_weighted_mean_square,
)
from train_hccb_p418_spatiotemporal_regional_operator import (
    area_weighted_flux_density_mean_square,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_summary(path: Path) -> dict:
    summary_path = path.parent / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing source summary: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def validate_graph_provenance(
    subface_path: Path, regional_topology_path: Path
) -> dict:
    hashes = {
        "subface_geometry_sha256": sha256(subface_path),
        "regional_topology_sha256": sha256(regional_topology_path),
    }
    subface = artifact_summary(subface_path)
    topology = artifact_summary(regional_topology_path)
    expected = {
        "subface summary geometry": (
            subface["geometry_sha256"], hashes["subface_geometry_sha256"]
        ),
        "subface source topology": (
            subface["source_regional_topology_sha256"],
            hashes["regional_topology_sha256"],
        ),
        "topology summary": (
            topology["regional_topology_sha256"],
            hashes["regional_topology_sha256"],
        ),
    }
    inconsistent = [
        f"{name}: recorded={recorded}, actual={actual}"
        for name, (recorded, actual) in expected.items()
        if recorded != actual
    ]
    if inconsistent:
        raise ValueError("inconsistent graph inputs:\n" + "\n".join(inconsistent))
    return {**hashes, "checks": {name: True for name in expected}}


def validate_input_provenance(
    subface_path: Path,
    regional_topology_path: Path,
    regional_state_targets_path: Path,
    dataset_index_path: Path,
) -> dict:
    graph_provenance = validate_graph_provenance(
        subface_path, regional_topology_path
    )
    hashes = {
        "subface_geometry_sha256": graph_provenance["subface_geometry_sha256"],
        "regional_topology_sha256": graph_provenance[
            "regional_topology_sha256"
        ],
        "regional_state_targets_sha256": sha256(regional_state_targets_path),
        "steady_dataset_index_sha256": sha256(dataset_index_path),
    }
    subface = artifact_summary(subface_path)
    state_targets = artifact_summary(regional_state_targets_path)
    expected = {
        "subface source dataset": (
            subface["source_dataset_sha256"], hashes["steady_dataset_index_sha256"]
        ),
        "state summary target": (
            state_targets["target_sha256"],
            hashes["regional_state_targets_sha256"],
        ),
        "state source subface": (
            state_targets["source_subface_geometry_sha256"],
            hashes["subface_geometry_sha256"],
        ),
        "state source dataset": (
            state_targets["source_dataset_sha256"],
            hashes["steady_dataset_index_sha256"],
        ),
    }
    inconsistent = [
        f"{name}: recorded={recorded}, actual={actual}"
        for name, (recorded, actual) in expected.items()
        if recorded != actual
    ]
    if inconsistent:
        raise ValueError("inconsistent model inputs:\n" + "\n".join(inconsistent))
    return {
        **hashes,
        "checks": {**graph_provenance["checks"], **{name: True for name in expected}},
    }


def write_model_graph(
    subface_path: Path, regional_topology_path: Path, output_path: Path
) -> dict[str, object]:
    with np.load(subface_path, allow_pickle=False) as data:
        fluid = data["fluid_global_region"].astype(np.int64)
        solid = data["solid_global_region"].astype(np.int64)
        node_count = len(fluid) + len(solid)
        node_type = np.ones(node_count, dtype=np.int8)
        node_type[fluid] = 0
        volume = np.zeros(node_count, dtype=np.float64)
        centroid = np.zeros((node_count, 3), dtype=np.float64)
        volume[fluid] = data["fluid_cell_volume_m3"]
        volume[solid] = data["solid_cell_volume_m3"]
        centroid[fluid] = data["fluid_cell_centroid_m"]
        centroid[solid] = data["solid_cell_centroid_m"]
    level, graph = matching_regional_graph(regional_topology_path, node_type)
    np.savez_compressed(
        output_path,
        node_centroid_m=centroid,
        node_volume_m3=volume,
        node_type=node_type,
        regional_graph_level=np.asarray(level, dtype=np.int64),
        **graph,
    )
    return {
        "regional_graph_level": level,
        "nodes": node_count,
        "edges": len(graph["edge_source"]),
        "subface_geometry_sha256": sha256(subface_path),
        "regional_topology_sha256": sha256(regional_topology_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subface-geometry", type=Path, required=True)
    parser.add_argument("--regional-topology", type=Path, required=True)
    parser.add_argument("--regional-state-targets", type=Path, required=True)
    parser.add_argument("--steady-dataset-index", type=Path, required=True)
    parser.add_argument("--step-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--physics-mode",
        choices=("data_only", "energy_and_flux"),
        default="energy_and_flux",
    )
    parser.add_argument(
        "--spatial-temporal-mode",
        choices=SPATIAL_TEMPORAL_MODES,
        default="repeated_query_spatial",
    )
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    if args.torch_threads is not None:
        if args.torch_threads <= 0:
            raise ValueError("torch threads must be positive")
        torch.set_num_threads(args.torch_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_provenance = validate_input_provenance(
        args.subface_geometry.resolve(),
        args.regional_topology.resolve(),
        args.regional_state_targets.resolve(),
        args.steady_dataset_index.resolve(),
    )
    graph_path = args.output_dir / "regional_sequence_geometry.npz"
    graph_summary = write_model_graph(
        args.subface_geometry.resolve(), args.regional_topology.resolve(), graph_path
    )
    graph = P418ThermalStepRegionalGraph.from_npz(graph_path, device=device)
    with np.load(args.regional_state_targets.resolve(), allow_pickle=False) as data:
        state = torch.as_tensor(
            data["state_physical"][0], dtype=torch.float32, device=device
        ).unsqueeze(0)
        condition = data["condition_physical"][0]
    step_condition = torch.tensor(
        [[
            condition[0], condition[1], condition[2] / 1.0e6,
            condition[0], condition[1], condition[2] / 1.0e6,
            condition[3], condition[4],
        ]],
        dtype=torch.float32,
        device=device,
    )
    dataset = json.loads(args.steady_dataset_index.resolve().read_text(encoding="utf-8"))
    residual_geometry = load_p418_subface_geometry(
        args.subface_geometry.resolve(),
        fluid_patch_names=dataset["boundary_patch_names"]["fluid"],
        solid_patch_names=dataset["boundary_patch_names"]["solid"],
        device=device,
        dtype=torch.float32,
    )
    plan = json.loads(args.step_plan.resolve().read_text(encoding="utf-8"))
    numerical = plan["numerical_time_design"]
    duration = float(numerical["duration_s"])
    _, snapshot_times = field_write_stages(numerical)
    time_count = len(snapshot_times)
    physical_time = torch.tensor(snapshot_times, dtype=torch.float32, device=device)
    normalized_time = physical_time / duration
    truth = state[:, None].expand(-1, time_count, -1, -1).clone()
    with torch.no_grad():
        reference = assemble_p418_transient_regional_residual(
            geometry=residual_geometry,
            step_condition=step_condition,
            state_physical=truth,
            time_s=physical_time,
        )
    reference_fields = {
        name: getattr(reference, name).detach()
        for name in (
            "fluid_energy_w_m3",
            "solid_energy_w_m3",
            "fluid_internal_energy_flux_w",
            "solid_internal_heat_flux_w",
        )
    }
    model = HCCBP418SpatiotemporalRegionalOperator(
        spatial_temporal_mode=args.spatial_temporal_mode
    ).to(device).train()
    model_condition = torch.zeros((1, 8), dtype=torch.float32, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    prediction = model(state, model_condition, normalized_time, graph)
    temperature_loss = 5.0 * (prediction[..., 4] - truth[..., 4]).square().mean()
    energy_loss = temperature_loss.new_zeros(())
    flux_loss = temperature_loss.new_zeros(())
    if args.physics_mode == "energy_and_flux":
        predicted = assemble_p418_transient_regional_residual(
            geometry=residual_geometry,
            step_condition=step_condition,
            state_physical=prediction,
            time_s=physical_time,
        )
        source_density = step_condition[:, 5] * 1.0e6
        source_power = source_density * residual_geometry.solid_mesh.cell_volume.sum()
        energy_loss = 0.5 * (
            volume_weighted_mean_square(
                (predicted.fluid_energy_w_m3 - reference_fields["fluid_energy_w_m3"])
                / source_density[:, None, None],
                residual_geometry.fluid_mesh.cell_volume,
            )
            + volume_weighted_mean_square(
                (predicted.solid_energy_w_m3 - reference_fields["solid_energy_w_m3"])
                / source_density[:, None, None],
                residual_geometry.solid_mesh.cell_volume,
            )
        )
        flux_loss = 0.5 * (
            area_weighted_flux_density_mean_square(
                predicted.fluid_internal_energy_flux_w
                - reference_fields["fluid_internal_energy_flux_w"],
                torch.linalg.vector_norm(
                    residual_geometry.fluid_mesh.internal_area_vector, dim=1
                ),
                source_power,
            )
            + area_weighted_flux_density_mean_square(
                predicted.solid_internal_heat_flux_w
                - reference_fields["solid_internal_heat_flux_w"],
                torch.linalg.vector_norm(
                    residual_geometry.solid_mesh.internal_area_vector, dim=1
                ),
                source_power,
            )
        )
        loss = temperature_loss + energy_loss + flux_loss
    else:
        loss = temperature_loss
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    summary = {
        "status": (
            "formal_actual_graph_model_data_loss_backward_passed"
            if args.physics_mode == "data_only"
            else "formal_actual_graph_model_and_transient_physics_backward_passed"
        ),
        **graph_summary,
        "time_points": time_count,
        "physics_mode": args.physics_mode,
        "regional_energy_weighting": (
            "fluid_and_solid_finite_volume_weighted_separately"
            if args.physics_mode == "energy_and_flux"
            else "not_used"
        ),
        "internal_heat_flux_measure": (
            "face_area_weighted_heat_flux_density_difference"
            if args.physics_mode == "energy_and_flux"
            else "not_used"
        ),
        "spatial_temporal_mode": args.spatial_temporal_mode,
        "torch_num_threads": torch.get_num_threads(),
        "random_seed": args.seed,
        "input_provenance": input_provenance,
        "step_plan_sha256": sha256(args.step_plan.resolve()),
        "model_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "elapsed_seconds": elapsed,
        "peak_gpu_GB": (
            torch.cuda.max_memory_allocated() / 1.0e9 if device.type == "cuda" else None
        ),
        "initial_maximum_absolute_error": float(
            (prediction[:, 0] - state).detach().abs().max().cpu()
        ),
        "hydrodynamic_maximum_absolute_error": float(
            (prediction[..., :4] - state[:, None, :, :4]).detach().abs().max().cpu()
        ),
        "loss_finite": bool(torch.isfinite(loss).detach().cpu()),
        "weighted_temperature_loss": float(temperature_loss.detach().cpu()),
        "projection_aware_volume_energy_loss": float(energy_loss.detach().cpu()),
        "area_weighted_heat_flux_density_loss": float(flux_loss.detach().cpu()),
        "total_smoke_objective": float(loss.detach().cpu()),
        "loss_scale_interpretation": (
            "The smoke target repeats one steady field over time. These initial "
            "loss magnitudes check finite evaluation only and are not used to tune "
            "the formal loss weights; weight comparison requires the 12 solved steps."
        ),
        "gpu_reproducibility_note": (
            "The random seed is fixed. CUDA graph reductions can still differ at "
            "round-off level because their parallel summation order is not bitwise fixed; "
            "formal accuracy is therefore reported over three independent seeds."
        ),
        "trainable_tensor_count": len(gradients),
        "all_gradients_present": all(value is not None for value in gradients),
        "all_gradients_finite": all(
            value is not None and torch.isfinite(value).all().item()
            for value in gradients
        ),
        "physical_parameter_ids": ["P388", "P389", "P403", "P418", "P428", "P429", "P430", "P431"],
        "new_physical_parameters": [],
        "scientific_scope": (
            "One full-size forward/loss/backward resource measurement on a repeated "
            "steady regional state; not a transient accuracy result."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
