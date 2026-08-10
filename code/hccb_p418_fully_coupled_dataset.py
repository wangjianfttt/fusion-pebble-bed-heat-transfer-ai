#!/usr/bin/env python3
"""Dataset utilities for fully coupled P418 flow--heat step histories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from export_hccb_p418_step_regional_sequences import (
    CONDITION_NAMES,
    STATE_NAMES,
    validate_sequence_arrays,
)


def load_index(path: Path) -> dict[str, object]:
    index = json.loads(path.resolve().read_text(encoding="utf-8"))
    if index.get("history_mode") != "fully_coupled_flow_heat":
        raise ValueError("dataset is not a fully coupled P418 flow--heat history")
    if index.get("state_names") != list(STATE_NAMES):
        raise ValueError("fully coupled dataset state order differs from Ux,Uy,Uz,p,T")
    if index.get("condition_names") != list(CONDITION_NAMES):
        raise ValueError("fully coupled dataset condition order differs from the contract")
    records = index.get("sequences", [])
    identifiers = [str(row["sequence_id"]) for row in records]
    if len(identifiers) != len(set(identifiers)) or len(records) != int(index["sequence_count"]):
        raise ValueError("fully coupled sequence records are duplicated or incomplete")
    return index


def sequence_records(index: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(row["sequence_id"]): row for row in index["sequences"]}


def load_sequence(
    root: Path, record: dict[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(root / str(record["sequence_file"]), allow_pickle=False) as data:
        times = data["time_s"].astype(np.float32)
        condition = data["condition_physical"].astype(np.float32)
        state = data["state_physical"].astype(np.float32)
        internal_flux = data["fluid_internal_mass_flux_kg_s"].astype(np.float32)
        boundary_flux = data["fluid_boundary_mass_flux_kg_s"].astype(np.float32)
    if condition.shape != (len(CONDITION_NAMES),):
        raise ValueError(f"invalid condition vector in {record['sequence_id']}")
    validate_sequence_arrays(
        times=times,
        state=state,
        internal_mass_flux=internal_flux,
        boundary_mass_flux=boundary_flux,
        history_mode="fully_coupled_flow_heat",
    )
    return times, condition, state, internal_flux, boundary_flux


def selected_split(
    sequence_ids: set[str], split_path: Path, split_name: str
) -> dict[str, list[str]]:
    raw = json.loads(split_path.resolve().read_text(encoding="utf-8"))["splits"][
        split_name
    ]
    split = {
        role: [str(value) for value in raw[role]]
        for role in ("train", "validation", "test")
    }
    role_sets = {role: set(values) for role, values in split.items()}
    if any(len(role_sets[role]) != len(split[role]) for role in split):
        raise ValueError("fully coupled split contains a duplicate curve")
    if any(
        role_sets[left] & role_sets[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise ValueError("one fully coupled curve appears in more than one role")
    if set().union(*role_sets.values()) != sequence_ids:
        raise ValueError("fully coupled split does not cover the available curves exactly")
    return split


def _mean_std(sum_: np.ndarray, square: np.ndarray, count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros_like(sum_, dtype=np.float64)
    std = np.ones_like(sum_, dtype=np.float64)
    populated = count > 0
    mean[populated] = sum_[populated] / count[populated]
    variance = np.zeros_like(mean)
    variance[populated] = square[populated] / count[populated] - np.square(
        mean[populated]
    )
    std[populated] = np.sqrt(np.maximum(variance[populated], 0.0))
    std[std < 1.0e-12] = 1.0
    return mean, std


def training_statistics(
    root: Path,
    records: dict[str, dict[str, object]],
    training_ids: list[str],
    node_type: np.ndarray,
) -> dict[str, np.ndarray | float | list[str]]:
    """Calculate every normalization quantity from complete training curves only."""
    if not training_ids:
        raise ValueError("fully coupled training split is empty")
    if node_type.ndim != 1 or np.any((node_type != 0) & (node_type != 1)):
        raise ValueError("node type must contain fluid=0 and solid=1")
    conditions: list[np.ndarray] = []
    state_sum = np.zeros((2, len(STATE_NAMES)), dtype=np.float64)
    state_square = np.zeros_like(state_sum)
    state_count = np.zeros_like(state_sum, dtype=np.int64)
    valid_channels = {0: np.arange(5), 1: np.asarray([4])}
    internal_values: list[np.ndarray] = []
    boundary_values: list[np.ndarray] = []
    maximum_time = 0.0
    for sequence_id in training_ids:
        times, condition, state, internal_flux, boundary_flux = load_sequence(
            root, records[sequence_id]
        )
        if state.shape[1] != len(node_type):
            raise ValueError(f"node count differs in {sequence_id}")
        conditions.append(condition.astype(np.float64))
        maximum_time = max(maximum_time, float(times.max()))
        for material in (0, 1):
            selected = state[:, node_type == material]
            for channel in valid_channels[material]:
                values = selected[..., channel].astype(np.float64)
                state_sum[material, channel] += values.sum()
                state_square[material, channel] += np.square(values).sum()
                state_count[material, channel] += values.size
        internal_values.append(internal_flux.astype(np.float64).reshape(-1))
        boundary_values.append(boundary_flux.astype(np.float64).reshape(-1))
    condition_values = np.stack(conditions)
    condition_mean = condition_values.mean(axis=0)
    condition_std = condition_values.std(axis=0)
    condition_std[condition_std < 1.0e-12] = 1.0
    state_mean, state_std = _mean_std(state_sum, state_square, state_count)
    internal_joined = np.concatenate(internal_values)
    boundary_joined = np.concatenate(boundary_values)
    internal_mean = float(internal_joined.mean())
    internal_std = float(internal_joined.std())
    boundary_mean = float(boundary_joined.mean())
    boundary_std = float(boundary_joined.std())
    if internal_std < 1.0e-12:
        internal_std = 1.0
    if boundary_std < 1.0e-12:
        boundary_std = 1.0
    if maximum_time <= 0.0:
        raise ValueError("fully coupled training curves contain no positive time")
    return {
        "condition_mean": condition_mean.astype(np.float32),
        "condition_std": condition_std.astype(np.float32),
        "state_mean": state_mean.astype(np.float32),
        "state_std": state_std.astype(np.float32),
        "internal_mass_flux_mean_kg_s": internal_mean,
        "internal_mass_flux_std_kg_s": internal_std,
        "boundary_mass_flux_mean_kg_s": boundary_mean,
        "boundary_mass_flux_std_kg_s": boundary_std,
        "maximum_time_s": maximum_time,
        "training_sequence_ids": list(training_ids),
        "new_physical_parameters": [],
    }
