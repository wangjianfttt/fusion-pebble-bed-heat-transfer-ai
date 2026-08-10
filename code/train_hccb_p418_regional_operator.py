#!/usr/bin/env python3
"""Train graph-operator or Transolver baselines on the P418 steady CHT fields.

The five physical inputs and every target field come from the P418/OpenFOAM
dataset.  Neural-network settings are read from the archived algorithm-source
registry; they never replace a physical parameter.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import nn

from hccb_p418_parametric_regional_operator import (
    HCCBP418ParametricRegionalOperator,
    P418RegionalMesh,
    collapse_mesh_to_level,
    load_p418_regional_mesh,
)
from hccb_p418_coordinate_pinn import HCCBP418CoordinatePINNOperator


CONDITION_KEYS = (
    "inlet_velocity_m_s",
    "inlet_temperature_K",
    "solid_heat_source_W_m3",
    "outlet_pressure_Pa",
    "cooling_wall_temperature_K",
)


@dataclass(frozen=True)
class FieldScales:
    condition_mean: np.ndarray
    condition_std: np.ndarray
    velocity_mean: np.ndarray
    velocity_std: np.ndarray
    pressure_mean: float
    pressure_std: float
    fluid_temperature_mean: float
    fluid_temperature_std: float
    solid_temperature_mean: float
    solid_temperature_std: float


@dataclass(frozen=True)
class EngineeringGeometry:
    fluid_boundary_owner: np.ndarray
    fluid_boundary_patch: np.ndarray
    fluid_boundary_area_m2: np.ndarray
    fluid_boundary_area_vector_m2: np.ndarray
    inlet_patch: int
    outlet_patch: int


def load_scales(path: Path, split_name: str) -> FieldScales:
    payload = json.loads(path.read_text(encoding="utf-8"))
    split = payload["splits"][split_name]
    condition = split["condition_input"]
    target = split["targets"]
    condition_std = np.asarray(condition["standard_deviation"], dtype=np.float64)
    return FieldScales(
        condition_mean=np.asarray(condition["mean"], dtype=np.float64),
        condition_std=condition_std,
        velocity_mean=np.asarray(target["fluid_velocity_m_s"]["mean"], dtype=np.float64),
        velocity_std=np.asarray(
            target["fluid_velocity_m_s"]["standard_deviation"], dtype=np.float64
        ),
        pressure_mean=float(target["fluid_gauge_pressure_Pa"]["mean"][0]),
        pressure_std=float(
            target["fluid_gauge_pressure_Pa"]["standard_deviation"][0]
        ),
        fluid_temperature_mean=float(target["fluid_temperature_K"]["mean"][0]),
        fluid_temperature_std=float(
            target["fluid_temperature_K"]["standard_deviation"][0]
        ),
        solid_temperature_mean=float(target["solid_temperature_K"]["mean"][0]),
        solid_temperature_std=float(
            target["solid_temperature_K"]["standard_deviation"][0]
        ),
    )


def normalized_condition_values(
    physical: np.ndarray, scales: FieldScales
) -> np.ndarray:
    """Normalize supported inputs and zero inputs that are constant in training."""
    physical = np.asarray(physical, dtype=np.float64)
    constant = scales.condition_std <= 0.0
    safe_std = np.where(constant, 1.0, scales.condition_std)
    normalized = (physical - scales.condition_mean) / safe_std
    normalized[..., constant] = 0.0
    return normalized.astype(np.float32)


def normalized_condition(record: dict[str, object], scales: FieldScales) -> np.ndarray:
    physical = np.asarray([float(record[key]) for key in CONDITION_KEYS], dtype=np.float64)
    return normalized_condition_values(physical, scales)


def normalized_target_chunk(
    field: dict[str, np.ndarray],
    *,
    fluid_count: int,
    start: int,
    stop: int,
    outlet_pressure_pa: float,
    scales: FieldScales,
) -> tuple[np.ndarray, np.ndarray]:
    """Return [Ux,Uy,Uz,gauge-p,T] and its valid-channel mask."""
    if not 0 <= start < stop <= fluid_count + len(field["solid_temperature_K"]):
        raise ValueError("target chunk is outside the fluid-solid field")
    target = np.zeros((stop - start, 5), dtype=np.float32)
    valid = np.zeros_like(target, dtype=bool)
    fluid_stop = min(stop, fluid_count)
    if start < fluid_stop:
        local = slice(0, fluid_stop - start)
        source = slice(start, fluid_stop)
        target[local, :3] = (
            field["fluid_velocity_m_s"][source] - scales.velocity_mean
        ) / scales.velocity_std
        target[local, 3] = (
            field["fluid_pressure_Pa"][source]
            - outlet_pressure_pa
            - scales.pressure_mean
        ) / scales.pressure_std
        target[local, 4] = (
            field["fluid_temperature_K"][source] - scales.fluid_temperature_mean
        ) / scales.fluid_temperature_std
        valid[local, :] = True
    solid_start = max(start, fluid_count)
    if solid_start < stop:
        local = slice(solid_start - start, stop - start)
        source = slice(solid_start - fluid_count, stop - fluid_count)
        target[local, 4] = (
            field["solid_temperature_K"][source] - scales.solid_temperature_mean
        ) / scales.solid_temperature_std
        valid[local, 4] = True
    return target, valid


def channel_denominators(mesh: P418RegionalMesh) -> torch.Tensor:
    fluid = mesh.fine_node_type == 0
    solid = mesh.fine_node_type == 1
    fluid_volume = mesh.fine_volume_m3[fluid].sum()
    solid_volume = mesh.fine_volume_m3[solid].sum()
    return torch.stack(
        (fluid_volume, fluid_volume, fluid_volume, fluid_volume, fluid_volume, solid_volume)
    )


def chunk_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    node_type: torch.Tensor,
    volume: torch.Tensor,
    denominators: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equal-weighted volume MSE for five fluid fields and solid temperature."""
    if prediction.shape != target.shape or prediction.shape != valid.shape:
        raise ValueError("prediction, target and mask shapes differ")
    squared = (prediction - target).square() * volume.view(1, -1, 1)
    fluid = node_type == 0
    solid = node_type == 1
    numerators = torch.stack(
        tuple(squared[:, fluid, channel].sum() for channel in range(5))
        + (squared[:, solid, 4].sum(),)
    )
    expected_valid = torch.zeros_like(valid)
    expected_valid[:, fluid, :] = True
    expected_valid[:, solid, 4] = True
    if not torch.equal(valid, expected_valid):
        raise ValueError("target mask does not match fluid and solid channel definitions")
    normalized = numerators / denominators
    return normalized.mean(), normalized.detach()


def load_field(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def load_engineering_geometry(
    dataset_root: Path, dataset: dict[str, object]
) -> EngineeringGeometry:
    patch_names = list(dataset["boundary_patch_names"]["fluid"])
    for required in ("inlet", "outlet"):
        if required not in patch_names:
            raise ValueError(f"fluid patch list has no {required!r} patch")
    with np.load(
        dataset_root / str(dataset["shared_topology_file"]), allow_pickle=False
    ) as mesh:
        return EngineeringGeometry(
            fluid_boundary_owner=mesh["fluid_boundary_face_owner"].astype(
                np.int64, copy=True
            ),
            fluid_boundary_patch=mesh["fluid_boundary_face_patch"].astype(
                np.int64, copy=True
            ),
            fluid_boundary_area_m2=mesh["fluid_boundary_face_area_m2"].astype(
                np.float64, copy=True
            ),
            fluid_boundary_area_vector_m2=mesh[
                "fluid_boundary_face_area_vector_outward_m2"
            ].astype(np.float64, copy=True),
            inlet_patch=patch_names.index("inlet"),
            outlet_patch=patch_names.index("outlet"),
        )


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(np.sum(weights))
    if denominator <= 0.0:
        raise ValueError("engineering-average weights must have a positive sum")
    return float(np.sum(values * weights) / denominator)


def volume_weighted_rmse(
    predicted: np.ndarray, reference: np.ndarray, volume_m3: np.ndarray
) -> float:
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    volume = np.asarray(volume_m3, dtype=np.float64)
    if predicted.shape != reference.shape or predicted.shape != volume.shape:
        raise ValueError("field values and cell volumes must have the same shape")
    denominator = float(np.sum(volume))
    if denominator <= 0.0 or np.any(volume <= 0.0):
        raise ValueError("cell volumes must be positive")
    return float(np.sqrt(np.sum(volume * np.square(predicted - reference)) / denominator))


def helium_density_p389(
    pressure_pa: np.ndarray, temperature_k: np.ndarray
) -> np.ndarray:
    """P389 helium density correlation; pressure is converted from Pa to MPa."""
    if np.any(temperature_k <= 0.0) or np.any(pressure_pa <= 0.0):
        raise ValueError("helium density requires positive pressure and temperature")
    return 480.19 * (pressure_pa / 1.0e6) / temperature_k


def cell_adjacent_engineering_metrics(
    *,
    velocity_m_s: np.ndarray,
    pressure_pa: np.ndarray,
    fluid_temperature_k: np.ndarray,
    solid_temperature_k: np.ndarray,
    geometry: EngineeringGeometry,
) -> dict[str, float]:
    owner = geometry.fluid_boundary_owner
    patch = geometry.fluid_boundary_patch
    area = geometry.fluid_boundary_area_m2
    area_vector = geometry.fluid_boundary_area_vector_m2
    inlet = patch == geometry.inlet_patch
    outlet = patch == geometry.outlet_patch
    if not np.any(inlet) or not np.any(outlet):
        raise ValueError("engineering geometry has no inlet or outlet faces")
    cell_pressure = pressure_pa[owner]
    cell_temperature = fluid_temperature_k[owner]
    cell_velocity = velocity_m_s[owner]
    density = helium_density_p389(cell_pressure, cell_temperature)
    mass_flux = density * np.einsum("ij,ij->i", cell_velocity, area_vector)
    inlet_mass = float(-np.sum(mass_flux[inlet]))
    outlet_mass = float(np.sum(mass_flux[outlet]))
    relative_mass_difference = (
        abs(outlet_mass - inlet_mass) / abs(inlet_mass)
        if inlet_mass != 0.0
        else math.inf
    )
    inlet_pressure = weighted_average(cell_pressure[inlet], area[inlet])
    outlet_pressure = weighted_average(cell_pressure[outlet], area[outlet])
    return {
        "pressure_drop_boundary_adjacent_cells_Pa": inlet_pressure
        - outlet_pressure,
        "outlet_temperature_boundary_adjacent_cells_K": weighted_average(
            cell_temperature[outlet], area[outlet]
        ),
        "solid_maximum_temperature_K": float(np.max(solid_temperature_k)),
        "inlet_mass_flow_boundary_adjacent_cells_kg_s": inlet_mass,
        "outlet_mass_flow_boundary_adjacent_cells_kg_s": outlet_mass,
        "relative_mass_difference_boundary_adjacent_cells": relative_mass_difference,
    }


def exact_boundary_reference_metrics(
    field: dict[str, np.ndarray], geometry: EngineeringGeometry
) -> dict[str, float]:
    patch = geometry.fluid_boundary_patch
    area = geometry.fluid_boundary_area_m2
    inlet = patch == geometry.inlet_patch
    outlet = patch == geometry.outlet_patch
    pressure = field["fluid_boundary_pressure_Pa"]
    temperature = field["fluid_boundary_temperature_K"]
    mass_flux = field["fluid_boundary_face_mass_flow_kg_s"]
    inlet_mass = float(-np.sum(mass_flux[inlet]))
    outlet_mass = float(np.sum(mass_flux[outlet]))
    return {
        "pressure_drop_boundary_faces_Pa": weighted_average(
            pressure[inlet], area[inlet]
        )
        - weighted_average(pressure[outlet], area[outlet]),
        "outlet_temperature_boundary_faces_K": weighted_average(
            temperature[outlet], area[outlet]
        ),
        "solid_maximum_temperature_K": float(
            np.max(field["solid_temperature_K"])
        ),
        "inlet_mass_flow_boundary_faces_kg_s": inlet_mass,
        "outlet_mass_flow_boundary_faces_kg_s": outlet_mass,
        "relative_mass_difference_boundary_faces": abs(outlet_mass - inlet_mass)
        / inlet_mass,
    }


def source_learning_rate(step: int, total_steps: int) -> float:
    """RIGNO's 2% warmup, 88% cosine decay and 10% exponential tail."""
    if total_steps <= 1:
        return 2.0e-4
    fraction = step / (total_steps - 1)
    if fraction < 0.02:
        return 1.0e-5 + (2.0e-4 - 1.0e-5) * fraction / 0.02
    if fraction < 0.90:
        progress = (fraction - 0.02) / 0.88
        return 1.0e-5 + 0.5 * (2.0e-4 - 1.0e-5) * (1.0 + math.cos(math.pi * progress))
    progress = (fraction - 0.90) / 0.10
    return 1.0e-5 * (1.0e-6 / 1.0e-5) ** progress


def iter_chunks(total: int, size: int) -> Iterator[tuple[int, int]]:
    for start in range(0, total, size):
        yield start, min(start + size, total)


def build_model(
    architecture: str, boundary_role_count: int
) -> tuple[nn.Module, dict[str, float | int | str]]:
    if architecture == "graph":
        settings: dict[str, float | int | str] = {
            "hidden_dim": 128,
            "processor_steps": 12,
            "processor_kind": "message_passing",
            "weight_decay": 1.0e-8,
            "epochs": 2000,
            "effective_batch_size": 2,
            "optimizer": "AdamW",
            "learning_rate": 2.0e-4,
            "scheduler": "RIGNO warmup-cosine-exponential",
        }
    elif architecture == "transolver":
        settings = {
            "hidden_dim": 256,
            "processor_steps": 5,
            "processor_kind": "hybrid_physics_attention",
            "weight_decay": 1.0e-5,
            "epochs": 500,
            "effective_batch_size": 8,
            "optimizer": "AdamW",
            "learning_rate": 1.0e-3,
            "scheduler": "OneCycleLR",
        }
    elif architecture == "pinn":
        settings = {
            "hidden_dim": 50,
            "hidden_layers": 6,
            "weight_decay": 0.0,
            "epochs": 3000,
            "effective_batch_size": 1,
            "optimizer": "Adam",
            "learning_rate": 1.0e-2,
            "scheduler": "constant",
            "activation": "tanh",
            "initialization": "Glorot normal",
        }
        model = HCCBP418CoordinatePINNOperator(
            boundary_role_count=boundary_role_count,
            hidden_dim=int(settings["hidden_dim"]),
            hidden_layers=int(settings["hidden_layers"]),
        )
        return model, settings
    else:
        raise ValueError("architecture must be pinn, graph or transolver")
    model = HCCBP418ParametricRegionalOperator(
        boundary_role_count=boundary_role_count,
        hidden_dim=int(settings["hidden_dim"]),
        processor_steps=int(settings["processor_steps"]),
        active_levels=1,
        processor_kind=str(settings["processor_kind"]),
        attention_heads=8,
        attention_start_level=0,
        physics_slices=32,
    )
    return model, settings


def evaluate(
    *,
    model: nn.Module,
    mesh: P418RegionalMesh,
    records: dict[str, dict[str, object]],
    case_ids: list[str],
    dataset_root: Path,
    scales: FieldScales,
    engineering_geometry: EngineeringGeometry,
    chunk_size: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    denominators = channel_denominators(mesh)
    fluid_count = int((mesh.fine_node_type == 0).sum())
    results: list[dict[str, object]] = []
    with torch.no_grad():
        for case_id in case_ids:
            record = records[case_id]
            field = load_field(dataset_root / str(record["field_file"]))
            condition = torch.as_tensor(
                normalized_condition(record, scales), device=device
            ).view(1, -1)
            regional = model.encode_regions(condition, mesh)
            channel_sum = torch.zeros(6, device=device)
            predicted_velocity = np.empty((fluid_count, 3), dtype=np.float32)
            predicted_pressure = np.empty(fluid_count, dtype=np.float32)
            predicted_fluid_temperature = np.empty(fluid_count, dtype=np.float32)
            predicted_solid_temperature = np.empty(
                mesh.n_fine - fluid_count, dtype=np.float32
            )
            for start, stop in iter_chunks(mesh.n_fine, chunk_size):
                prediction = model.decode_fine_chunk(condition, regional, mesh, start, stop)
                target_np, valid_np = normalized_target_chunk(
                    field,
                    fluid_count=fluid_count,
                    start=start,
                    stop=stop,
                    outlet_pressure_pa=float(record["outlet_pressure_Pa"]),
                    scales=scales,
                )
                target = torch.as_tensor(target_np, device=device).unsqueeze(0)
                valid = torch.as_tensor(valid_np, device=device).unsqueeze(0)
                _, channel = chunk_loss(
                    prediction,
                    target,
                    valid,
                    mesh.fine_node_type[start:stop],
                    mesh.fine_volume_m3[start:stop],
                    denominators,
                )
                channel_sum += channel
                physical = prediction[0].detach().cpu().numpy()
                fluid_stop = min(stop, fluid_count)
                if start < fluid_stop:
                    local_stop = fluid_stop - start
                    predicted_velocity[start:fluid_stop] = (
                        physical[:local_stop, :3] * scales.velocity_std
                        + scales.velocity_mean
                    )
                    predicted_pressure[start:fluid_stop] = (
                        physical[:local_stop, 3] * scales.pressure_std
                        + scales.pressure_mean
                        + float(record["outlet_pressure_Pa"])
                    )
                    predicted_fluid_temperature[start:fluid_stop] = (
                        physical[:local_stop, 4] * scales.fluid_temperature_std
                        + scales.fluid_temperature_mean
                    )
                solid_start = max(start, fluid_count)
                if solid_start < stop:
                    local_start = solid_start - start
                    predicted_solid_temperature[
                        solid_start - fluid_count : stop - fluid_count
                    ] = (
                        physical[local_start:, 4] * scales.solid_temperature_std
                        + scales.solid_temperature_mean
                    )
            predicted_engineering = cell_adjacent_engineering_metrics(
                velocity_m_s=predicted_velocity,
                pressure_pa=predicted_pressure,
                fluid_temperature_k=predicted_fluid_temperature,
                solid_temperature_k=predicted_solid_temperature,
                geometry=engineering_geometry,
            )
            reference_cell_engineering = cell_adjacent_engineering_metrics(
                velocity_m_s=field["fluid_velocity_m_s"],
                pressure_pa=field["fluid_pressure_Pa"],
                fluid_temperature_k=field["fluid_temperature_K"],
                solid_temperature_k=field["solid_temperature_K"],
                geometry=engineering_geometry,
            )
            exact_boundary = exact_boundary_reference_metrics(
                field, engineering_geometry
            )
            engineering_errors = {
                "pressure_drop_absolute_error_Pa": abs(
                    predicted_engineering[
                        "pressure_drop_boundary_adjacent_cells_Pa"
                    ]
                    - reference_cell_engineering[
                        "pressure_drop_boundary_adjacent_cells_Pa"
                    ]
                ),
                "outlet_temperature_absolute_error_K": abs(
                    predicted_engineering[
                        "outlet_temperature_boundary_adjacent_cells_K"
                    ]
                    - reference_cell_engineering[
                        "outlet_temperature_boundary_adjacent_cells_K"
                    ]
                ),
                "solid_maximum_temperature_absolute_error_K": abs(
                    predicted_engineering["solid_maximum_temperature_K"]
                    - reference_cell_engineering["solid_maximum_temperature_K"]
                ),
            }
            fine_volume = mesh.fine_volume_m3.detach().cpu().numpy()
            fine_centroid = mesh.fine_centroid_m.detach().cpu().numpy()
            predicted_hotspot = int(np.argmax(predicted_solid_temperature))
            reference_hotspot = int(np.argmax(field["solid_temperature_K"]))
            solid_centroid = fine_centroid[fluid_count:]
            field_errors = {
                "fluid_temperature_volume_weighted_rmse_K": volume_weighted_rmse(
                    predicted_fluid_temperature,
                    field["fluid_temperature_K"],
                    fine_volume[:fluid_count],
                ),
                "solid_temperature_volume_weighted_rmse_K": volume_weighted_rmse(
                    predicted_solid_temperature,
                    field["solid_temperature_K"],
                    fine_volume[fluid_count:],
                ),
                "solid_hotspot_location_error_m": float(
                    np.linalg.norm(
                        solid_centroid[predicted_hotspot]
                        - solid_centroid[reference_hotspot]
                    )
                ),
                "predicted_hotspot_cell": predicted_hotspot,
                "reference_hotspot_cell": reference_hotspot,
            }
            results.append(
                {
                    "condition_id": case_id,
                    "normalized_rmse": float(torch.sqrt(channel_sum.mean()).cpu()),
                    "channel_rmse": torch.sqrt(channel_sum).cpu().tolist(),
                    "predicted_engineering_quantities": predicted_engineering,
                    "reference_boundary_adjacent_cell_quantities": reference_cell_engineering,
                    "reference_exact_boundary_quantities": exact_boundary,
                    "engineering_absolute_errors": engineering_errors,
                    "physical_field_errors": field_errors,
                    "cooling_wall_heat_flow_note": (
                        "Not inferred from state values alone; use the conservative "
                        "energy-flux model for cooling-wall heat flow."
                    ),
                }
            )
    return {
        "case_count": len(results),
        "mean_normalized_rmse": float(
            np.mean([float(item["normalized_rmse"]) for item in results])
        ),
        "cases": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--regional-topology", type=Path, required=True)
    parser.add_argument("--model-geometry", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--training-statistics", type=Path, required=True)
    parser.add_argument("--split-name", default="interleaved_all_ranges")
    parser.add_argument("--architecture", choices=("pinn", "graph", "transolver"), required=True)
    parser.add_argument("--regional-level", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--effective-batch-size", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_path = args.dataset_index.resolve()
    dataset_root = dataset_path.parent
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    engineering_geometry = load_engineering_geometry(dataset_root, dataset)
    split_payload = json.loads(args.split_file.read_text(encoding="utf-8"))
    split = split_payload["splits"][args.split_name]
    records = {str(item["condition_id"]): item for item in dataset["conditions"]}
    required = set(split["train"] + split["validation"] + split["test"])
    missing = sorted(required - set(records))
    if missing:
        raise ValueError(
            f"dataset is incomplete for {args.split_name}: {len(missing)} cases missing"
        )
    scales = load_scales(args.training_statistics.resolve(), args.split_name)
    full_mesh = load_p418_regional_mesh(
        args.regional_topology.resolve(), args.model_geometry.resolve()
    )
    mesh = collapse_mesh_to_level(full_mesh, args.regional_level)
    del full_mesh
    device = torch.device(args.device)
    mesh = mesh.to(device)
    model, settings = build_model(
        args.architecture, int(mesh.fine_boundary_role.shape[1])
    )
    model = model.to(device)
    if args.checkpoint is not None:
        checkpoint = torch.load(
            args.checkpoint.resolve(), map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
    if args.evaluate_only and args.checkpoint is None:
        raise ValueError("--evaluate-only requires --checkpoint")
    epochs = args.epochs or int(settings["epochs"])
    effective_batch = args.effective_batch_size or int(settings["effective_batch_size"])
    if min(epochs, effective_batch, args.chunk_size) <= 0:
        raise ValueError("epochs, batch size and chunk size must be positive")
    if args.architecture == "pinn":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(settings["learning_rate"])
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(settings["learning_rate"]),
            weight_decay=float(settings["weight_decay"]),
        )
    update_count = epochs * math.ceil(len(split["train"]) / effective_batch)
    scheduler = None
    if args.architecture == "transolver":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(settings["learning_rate"]),
            total_steps=update_count,
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.evaluate_only:
        evaluation = {
            "status": "evaluation_complete",
            "architecture": args.architecture,
            "split_name": args.split_name,
            "validation": evaluate(
                model=model,
                mesh=mesh,
                records=records,
                case_ids=list(split["validation"]),
                dataset_root=dataset_root,
                scales=scales,
                engineering_geometry=engineering_geometry,
                chunk_size=args.chunk_size,
                device=device,
            ),
            "test": evaluate(
                model=model,
                mesh=mesh,
                records=records,
                case_ids=list(split["test"]),
                dataset_root=dataset_root,
                scales=scales,
                engineering_geometry=engineering_geometry,
                chunk_size=args.chunk_size,
                device=device,
            ),
            "engineering_quantity_note": (
                "Model pressure drop, outlet temperature and mass flow use "
                "boundary-adjacent cells; exact OpenFOAM boundary-face values "
                "are reported separately. Helium density uses literature P389."
            ),
        }
        (output / "evaluation.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(evaluation, ensure_ascii=False, indent=2))
        return 0
    log_path = output / "training_history.jsonl"
    denominators = channel_denominators(mesh)
    fluid_count = int((mesh.fine_node_type == 0).sum())
    best_validation = math.inf
    update = 0
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(epochs):
            model.train()
            training_ids = list(split["train"])
            random.shuffle(training_ids)
            optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            cases_since_update = 0
            for case_index, case_id in enumerate(training_ids):
                group_start = (case_index // effective_batch) * effective_batch
                group_size = min(effective_batch, len(training_ids) - group_start)
                record = records[case_id]
                field = load_field(dataset_root / str(record["field_file"]))
                condition = torch.as_tensor(
                    normalized_condition(record, scales), device=device
                ).view(1, -1)
                regional = model.encode_regions(condition, mesh)
                chunks = list(iter_chunks(mesh.n_fine, args.chunk_size))
                case_loss = 0.0
                for chunk_index, (start, stop) in enumerate(chunks):
                    prediction = model.decode_fine_chunk(
                        condition, regional, mesh, start, stop
                    )
                    target_np, valid_np = normalized_target_chunk(
                        field,
                        fluid_count=fluid_count,
                        start=start,
                        stop=stop,
                        outlet_pressure_pa=float(record["outlet_pressure_Pa"]),
                        scales=scales,
                    )
                    target = torch.as_tensor(target_np, device=device).unsqueeze(0)
                    valid = torch.as_tensor(valid_np, device=device).unsqueeze(0)
                    loss, _ = chunk_loss(
                        prediction,
                        target,
                        valid,
                        mesh.fine_node_type[start:stop],
                        mesh.fine_volume_m3[start:stop],
                        denominators,
                    )
                    (loss / group_size).backward(
                        retain_graph=chunk_index + 1 < len(chunks)
                    )
                    case_loss += float(loss.detach().cpu())
                epoch_loss += case_loss
                cases_since_update += 1
                final_case = case_index + 1 == len(training_ids)
                if cases_since_update == effective_batch or final_case:
                    if args.architecture == "graph":
                        learning_rate = source_learning_rate(update, update_count)
                        for group in optimizer.param_groups:
                            group["lr"] = learning_rate
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
                    update += 1
                    cases_since_update = 0

            validation = evaluate(
                model=model,
                mesh=mesh,
                records=records,
                case_ids=list(split["validation"]),
                dataset_root=dataset_root,
                scales=scales,
                engineering_geometry=engineering_geometry,
                chunk_size=args.chunk_size,
                device=device,
            )
            record = {
                "epoch": epoch + 1,
                "training_mean_loss": epoch_loss / len(training_ids),
                "validation_mean_normalized_rmse": validation["mean_normalized_rmse"],
                "elapsed_s": time.time() - started,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            torch.save(
                {"model": model.state_dict(), "epoch": epoch + 1, "settings": settings},
                output / "latest.pt",
            )
            if float(validation["mean_normalized_rmse"]) < best_validation:
                best_validation = float(validation["mean_normalized_rmse"])
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch + 1, "settings": settings},
                    output / "best.pt",
                )
                (output / "best_validation.json").write_text(
                    json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

    summary = {
        "status": "training_complete",
        "architecture": args.architecture,
        "split_name": args.split_name,
        "regional_level": args.regional_level,
        "epochs": epochs,
        "best_validation_mean_normalized_rmse": best_validation,
        "training_seconds": time.time() - started,
        "settings_from_archived_source": settings,
        "physical_conditions": "P418 conditions from the dataset index; no generated condition",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
