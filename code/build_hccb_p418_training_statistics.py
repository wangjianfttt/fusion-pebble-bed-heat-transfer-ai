#!/usr/bin/env python3
"""Compute train-only scaling for the common P418 model comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CONDITION_KEYS = (
    "inlet_velocity_m_s",
    "inlet_temperature_K",
    "solid_heat_source_W_m3",
    "outlet_pressure_Pa",
    "cooling_wall_temperature_K",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class WeightedMoments:
    def __init__(self, channels: int) -> None:
        self.weight = np.zeros(channels, dtype=np.float64)
        self.total = np.zeros(channels, dtype=np.float64)
        self.square = np.zeros(channels, dtype=np.float64)

    def add(self, values: np.ndarray, weights: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if weights.shape != (values.shape[0],):
            raise ValueError("weights must have one entry per sample")
        self.weight += weights.sum()
        self.total += np.sum(values * weights[:, None], axis=0)
        self.square += np.sum(values * values * weights[:, None], axis=0)

    def result(self) -> dict[str, list[float]]:
        if np.any(self.weight <= 0.0):
            raise ValueError("normalization variable has no training support")
        mean = self.total / self.weight
        variance = np.maximum(self.square / self.weight - mean * mean, 0.0)
        standard_deviation = np.sqrt(variance)
        if np.any(standard_deviation <= 0.0):
            raise ValueError("normalization variable is constant in the training set")
        return {
            "mean": mean.tolist(),
            "standard_deviation": standard_deviation.tolist(),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset_path = args.dataset_index.resolve()
    split_path = args.split_file.resolve()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    dataset_root = dataset_path.parent
    topology_path = dataset_root / dataset["shared_topology_file"]
    with np.load(topology_path, allow_pickle=False) as topology:
        fluid_volume = topology["fluid_cell_volume_m3"].astype(np.float64)
        solid_volume = topology["solid_cell_volume_m3"].astype(np.float64)
        fluid_boundary_area = topology["fluid_boundary_face_area_m2"].astype(
            np.float64
        )
        solid_boundary_area = topology["solid_boundary_face_area_m2"].astype(
            np.float64
        )
        fluid_boundary_temperature_mask = topology[
            "fluid_boundary_temperature_value_mask"
        ].astype(bool)
        solid_boundary_temperature_mask = topology[
            "solid_boundary_temperature_value_mask"
        ].astype(bool)
    records = {item["condition_id"]: item for item in dataset["conditions"]}
    expected = {item["condition_id"] for item in splits["conditions"]}
    if set(records) != expected:
        missing = sorted(expected - set(records))
        extra = sorted(set(records) - expected)
        raise ValueError(f"dataset and P418 matrix differ; missing={missing}, extra={extra}")

    output_splits: dict[str, object] = {}
    for split_name, split in splits["splits"].items():
        training_ids = list(split["train"])
        condition_matrix = np.asarray(
            [[float(records[item][key]) for key in CONDITION_KEYS] for item in training_ids],
            dtype=np.float64,
        )
        condition_mean = condition_matrix.mean(axis=0)
        condition_std = condition_matrix.std(axis=0)
        if np.any(condition_std <= 0.0):
            constant = [CONDITION_KEYS[i] for i in np.flatnonzero(condition_std <= 0.0)]
        else:
            constant = []

        fluid_velocity = WeightedMoments(3)
        fluid_gauge_pressure = WeightedMoments(1)
        fluid_temperature = WeightedMoments(1)
        solid_temperature = WeightedMoments(1)
        mass_flux_square_sum = 0.0
        mass_flux_count = 0
        boundary_mass_flux_square_sum = 0.0
        boundary_mass_flux_count = 0
        fluid_boundary_temperature = WeightedMoments(1)
        solid_boundary_temperature = WeightedMoments(1)
        for condition_id in training_ids:
            record = records[condition_id]
            with np.load(dataset_root / record["field_file"], allow_pickle=False) as field:
                fluid_velocity.add(field["fluid_velocity_m_s"], fluid_volume)
                fluid_gauge_pressure.add(
                    field["fluid_pressure_Pa"] - float(record["outlet_pressure_Pa"]),
                    fluid_volume,
                )
                fluid_temperature.add(field["fluid_temperature_K"], fluid_volume)
                solid_temperature.add(field["solid_temperature_K"], solid_volume)
                face_flux = field["fluid_internal_face_mass_flow_kg_s"].astype(np.float64)
                mass_flux_square_sum += float(np.dot(face_flux, face_flux))
                mass_flux_count += len(face_flux)
                boundary_flux = field["fluid_boundary_face_mass_flow_kg_s"].astype(
                    np.float64
                )
                boundary_mass_flux_square_sum += float(
                    np.dot(boundary_flux, boundary_flux)
                )
                boundary_mass_flux_count += len(boundary_flux)
                fluid_boundary_temperature.add(
                    field["fluid_boundary_temperature_K"][
                        fluid_boundary_temperature_mask
                    ],
                    fluid_boundary_area[fluid_boundary_temperature_mask],
                )
                solid_boundary_temperature.add(
                    field["solid_boundary_temperature_K"][
                        solid_boundary_temperature_mask
                    ],
                    solid_boundary_area[solid_boundary_temperature_mask],
                )

        mass_flux_rms = float(np.sqrt(mass_flux_square_sum / mass_flux_count))
        boundary_mass_flux_rms = float(
            np.sqrt(boundary_mass_flux_square_sum / boundary_mass_flux_count)
        )
        if not mass_flux_rms > 0.0:
            raise ValueError("training mass-flow scale is non-positive")
        if not boundary_mass_flux_rms > 0.0:
            raise ValueError("training boundary mass-flow scale is non-positive")
        output_splits[split_name] = {
            "train_conditions": training_ids,
            "validation_conditions": split["validation"],
            "test_conditions": split["test"],
            "unused_conditions": split.get("unused", []),
            "question": split["question"],
            "condition_input": {
                "keys": list(CONDITION_KEYS),
                "mean": condition_mean.tolist(),
                "standard_deviation": condition_std.tolist(),
                "constant_training_inputs": constant,
                "constant_input_rule": "retain the published physical value and use normalized value zero",
            },
            "targets": {
                "fluid_velocity_m_s": fluid_velocity.result(),
                "fluid_gauge_pressure_Pa": fluid_gauge_pressure.result(),
                "fluid_temperature_K": fluid_temperature.result(),
                "solid_temperature_K": solid_temperature.result(),
                "fluid_internal_face_mass_flow_kg_s": {
                    "mean": 0.0,
                    "root_mean_square": mass_flux_rms,
                },
                "fluid_boundary_face_mass_flow_kg_s": {
                    "mean": 0.0,
                    "root_mean_square": boundary_mass_flux_rms,
                },
                "fluid_boundary_temperature_K": fluid_boundary_temperature.result(),
                "solid_boundary_temperature_K": solid_boundary_temperature.result(),
            },
        }

    payload = {
        "status": "p418_train_only_model_scaling_ready",
        "target_definition": {
            "fluid_cells": ["Ux", "Uy", "Uz", "p_minus_outlet_pressure", "temperature"],
            "solid_cells": ["temperature"],
            "fluid_internal_faces": ["signed_mass_flow"],
            "fluid_boundary_faces": ["signed_mass_flow"],
            "density": "computed from the literature helium state relation rather than freely predicted",
        },
        "normalization_rule": "each comparison uses statistics from its training conditions only",
        "splits": output_splits,
        "source": {
            "dataset_index": str(dataset_path),
            "dataset_index_sha256": sha256(dataset_path),
            "shared_topology_sha256": sha256(topology_path),
            "split_file": str(split_path),
            "split_file_sha256": sha256(split_path),
            "physical_parameter_source": splits["source_doi"],
        },
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
