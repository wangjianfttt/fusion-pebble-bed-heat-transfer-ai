#!/usr/bin/env python3
"""Shared data, split and metric contract for the steady P418 comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


STEADY_METRIC_CONTRACT = {
    "version": "p418_steady_volume_weighted_v1",
    "state_normalized_rmse": (
        "equal mean of six regional-volume-weighted channel MSE values: fluid Ux, "
        "Uy, Uz, gauge pressure and temperature plus solid temperature"
    ),
    "engineering_errors": {
        "pressure_drop_Pa": "area-averaged inlet pressure minus area-averaged outlet pressure",
        "outlet_temperature_K": "area-averaged outlet temperature",
        "solid_maximum_temperature_K": "maximum regional solid temperature",
        "cooling_wall_heat_into_fluid_W": (
            "negative outward fluid energy flow summed over the cooling-wall boundary"
        ),
        "solid_to_fluid_interphase_net_W": (
            "negative owner-to-neighbour energy flow summed over fluid-to-solid regional edges"
        ),
        "fluid_solid_interphase_absolute_flow_W": (
            "sum of absolute energy flow over fluid-to-solid regional edges"
        ),
    },
    "comparison_ranking": (
        "state and engineering metrics are reported separately; no field type is "
        "silently substituted for another"
    ),
}


def integrated_heat_transfer_metrics(
    *,
    internal_energy_flow_w: np.ndarray,
    boundary_energy_flow_w: np.ndarray,
    internal_kind: np.ndarray,
    internal_kind_name: np.ndarray,
    boundary_kind: np.ndarray,
    boundary_kind_name: np.ndarray,
) -> dict[str, float]:
    """Return integrated wall and fluid--solid heat-transfer quantities."""
    internal_names = [str(value) for value in internal_kind_name]
    boundary_names = [str(value) for value in boundary_kind_name]
    try:
        interface_kind = internal_names.index("fluid_to_solid")
        cooling_wall_kind = boundary_names.index("fluid:coolingWall")
    except ValueError as error:
        raise ValueError("energy targets lack declared interface or cooling-wall kinds") from error
    interface_flow = np.asarray(internal_energy_flow_w, dtype=np.float64)[
        np.asarray(internal_kind) == interface_kind
    ]
    cooling_wall_flow = np.asarray(boundary_energy_flow_w, dtype=np.float64)[
        np.asarray(boundary_kind) == cooling_wall_kind
    ]
    if not len(interface_flow) or not len(cooling_wall_flow):
        raise ValueError("energy targets contain no interface or cooling-wall faces")
    return {
        "cooling_wall_heat_into_fluid_W": float(-np.sum(cooling_wall_flow)),
        "solid_to_fluid_interphase_net_W": float(-np.sum(interface_flow)),
        "fluid_solid_interphase_absolute_flow_W": float(np.sum(np.abs(interface_flow))),
    }


def sha256_file(path: Path) -> str:
    path = path.resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def numerical_state_sha256(values: object) -> str:
    """Hash an ordered model state by key, shape, dtype and numerical bytes."""
    if not hasattr(values, "items"):
        raise TypeError("model state must provide ordered key/value items")
    digest = hashlib.sha256()
    for key, value in values.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(key).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def validate_split_and_statistics(
    *,
    split_file: Path,
    training_statistics: Path,
    split_name: str,
    condition_ids: np.ndarray,
) -> tuple[dict[str, list[str]], dict[str, object]]:
    """Return exact split identifiers after checking the train-only statistics."""
    identifiers = [str(value) for value in condition_ids]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("P418 target files contain repeated condition identifiers")

    split_path = split_file.resolve()
    statistics_path = training_statistics.resolve()
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    statistics_payload = json.loads(statistics_path.read_text(encoding="utf-8"))
    if split_name not in split_payload.get("splits", {}):
        raise ValueError(f"split {split_name!r} is absent from {split_path}")
    if split_name not in statistics_payload.get("splits", {}):
        raise ValueError(f"split {split_name!r} is absent from {statistics_path}")

    split = split_payload["splits"][split_name]
    exact: dict[str, list[str]] = {}
    seen: set[str] = set()
    available = set(identifiers)
    for role in ("train", "validation", "test"):
        values = [str(value) for value in split.get(role, [])]
        if not values:
            raise ValueError(f"{split_name} has no {role} conditions")
        if len(values) != len(set(values)):
            raise ValueError(f"{split_name} repeats a condition inside {role}")
        overlap = seen.intersection(values)
        if overlap:
            raise ValueError(f"{split_name} reuses conditions across roles: {sorted(overlap)}")
        missing = set(values) - available
        if missing:
            raise ValueError(f"{split_name} contains conditions absent from targets: {sorted(missing)}")
        exact[role] = values
        seen.update(values)

    has_unused_role = "unused" in split
    unused = [str(value) for value in split.get("unused", [])]
    if has_unused_role:
        if len(unused) != len(set(unused)):
            raise ValueError(f"{split_name} repeats a condition inside unused")
        overlap = seen.intersection(unused)
        if overlap:
            raise ValueError(
                f"{split_name} reuses conditions between model roles and unused: "
                f"{sorted(overlap)}"
            )
        missing = set(unused) - available
        if missing:
            raise ValueError(
                f"{split_name} contains unused conditions absent from targets: "
                f"{sorted(missing)}"
            )

    covered = seen.union(unused)
    if covered != available:
        raise ValueError(
            f"{split_name} does not partition all target conditions; "
            f"missing={sorted(available - covered)}, "
            f"extra={sorted(covered - available)}"
        )

    recorded = statistics_payload["splits"][split_name]
    role_keys = {
        "train": "train_conditions",
        "validation": "validation_conditions",
        "test": "test_conditions",
    }
    for role, key in role_keys.items():
        values = [str(value) for value in recorded.get(key, [])]
        if values != exact[role]:
            raise ValueError(
                f"training-statistics {role} conditions differ from {split_name}: "
                f"statistics={values}, split={exact[role]}"
            )
    if has_unused_role:
        recorded_unused = [str(value) for value in recorded.get("unused_conditions", [])]
        if recorded_unused != unused:
            raise ValueError(
                f"training-statistics unused conditions differ from {split_name}: "
                f"statistics={recorded_unused}, split={unused}"
            )
    recorded_split_hash = (
        statistics_payload.get("source", {}).get("split_file_sha256")
    )
    actual_split_hash = sha256_file(split_path)
    if recorded_split_hash != actual_split_hash:
        raise ValueError(
            "training statistics were produced from a different split file: "
            f"recorded={recorded_split_hash}, current={actual_split_hash}"
        )
    return exact, statistics_payload


def split_indices(
    split_case_ids: dict[str, list[str]], condition_ids: np.ndarray
) -> dict[str, np.ndarray]:
    index = {str(value): position for position, value in enumerate(condition_ids)}
    return {
        role: np.asarray([index[value] for value in values], dtype=np.int64)
        for role, values in split_case_ids.items()
    }


def run_provenance(
    *,
    architecture: str,
    comparison_epochs: int,
    split_name: str,
    split_case_ids: dict[str, list[str]],
    common_inputs: dict[str, Path],
    implementation_files: Iterable[Path],
) -> dict[str, object]:
    inputs = {name: file_record(path) for name, path in common_inputs.items()}
    implementations = {
        str(path.resolve()): file_record(path) for path in implementation_files
    }
    common_contract = {
        "split_name": split_name,
        "split_case_ids": split_case_ids,
        "common_input_sha256": {
            name: record["sha256"] for name, record in inputs.items()
        },
        "metric_contract": STEADY_METRIC_CONTRACT,
    }
    common_fingerprint = canonical_sha256(common_contract)
    run_contract = {
        **common_contract,
        "architecture": architecture,
        "comparison_epochs": int(comparison_epochs),
        "implementation_sha256": {
            name: record["sha256"] for name, record in implementations.items()
        },
    }
    return {
        "architecture": architecture,
        "comparison_epochs": int(comparison_epochs),
        "split_name": split_name,
        "split_case_ids": split_case_ids,
        "metric_contract": STEADY_METRIC_CONTRACT,
        "common_inputs": inputs,
        "implementation_files": implementations,
        "common_comparison_fingerprint": common_fingerprint,
        "run_fingerprint": canonical_sha256(run_contract),
    }
