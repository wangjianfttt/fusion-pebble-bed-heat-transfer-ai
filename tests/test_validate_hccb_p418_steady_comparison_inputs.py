#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from hccb_p418_comparison_contract import sha256_file  # noqa: E402
from validate_hccb_p418_steady_comparison_inputs import validate  # noqa: E402


def write_inputs(root: Path) -> dict[str, Path]:
    ids = np.asarray(["a", "b", "c"])
    node_type = np.asarray([0, 0, 1], dtype=np.int8)
    volume = np.asarray([1.0, 2.0, 3.0])
    state = root / "state.npz"
    np.savez_compressed(
        state,
        condition_id=ids,
        condition_physical=np.asarray(
            [[0.05, 300.0, 4.85e6, 1.2e5, 635.0],
             [0.15, 500.0, 6.85e6, 1.2e5, 635.0],
             [0.25, 900.0, 8.85e6, 1.2e5, 635.0]]
        ),
        state_physical=np.ones((3, 3, 5)),
        node_type=node_type,
        node_volume_m3=volume,
        fluid_global_region=np.asarray([0, 1]),
        solid_global_region=np.asarray([2]),
    )
    mass = root / "mass.npz"
    np.savez_compressed(
        mass,
        condition_id=ids,
        fluid_global_region=np.asarray([0, 1]),
        internal_owner=np.asarray([0]),
        internal_neighbour=np.asarray([1]),
        internal_face_area_m2=np.asarray([0.5]),
        boundary_owner=np.asarray([0, 1]),
        boundary_patch=np.asarray([0, 1]),
        boundary_face_area_m2=np.asarray([0.4, 0.4]),
        internal_mass_flow_kg_s=np.ones((3, 1)),
        boundary_mass_flow_kg_s=np.asarray([[-1.0, 1.0]] * 3),
    )
    energy = root / "energy.npz"
    np.savez_compressed(
        energy,
        condition_id=ids,
        node_type=node_type,
        node_volume_m3=volume,
        internal_owner=np.asarray([0, 0]),
        internal_neighbour=np.asarray([1, 2]),
        internal_kind=np.asarray([0, 2]),
        internal_kind_name=np.asarray(
            ["fluid_to_fluid", "solid_to_solid", "fluid_to_solid"]
        ),
        internal_face_area_m2=np.asarray([0.5, 0.6]),
        boundary_owner=np.asarray([0, 2]),
        boundary_kind=np.asarray([0, 1]),
        boundary_kind_name=np.asarray(["fluid:coolingWall", "solid:wall"]),
        boundary_face_area_m2=np.asarray([0.4, 0.7]),
        internal_energy_flow_W=np.ones((3, 2)),
        boundary_energy_flow_W=np.ones((3, 2)),
        node_source_power_W=np.ones((3, 3)),
    )
    split = root / "splits.json"
    split.write_text(
        json.dumps(
            {"splits": {"formal": {"train": ["a"], "validation": ["b"], "test": ["c"]}}}
        ),
        encoding="utf-8",
    )
    statistics = root / "statistics.json"
    statistics.write_text(
        json.dumps(
            {
                "splits": {
                    "formal": {
                        "train_conditions": ["a"],
                        "validation_conditions": ["b"],
                        "test_conditions": ["c"],
                    }
                },
                "source": {"split_file_sha256": sha256_file(split)},
            }
        ),
        encoding="utf-8",
    )
    return {
        "state_targets": state,
        "mass_targets": mass,
        "energy_targets": energy,
        "split_file": split,
        "training_statistics": statistics,
    }


def run(paths: dict[str, Path]) -> dict[str, object]:
    return validate(**paths, expected_cases=3)


def test_common_steady_inputs_are_accepted() -> None:
    with tempfile.TemporaryDirectory() as directory:
        result = run(write_inputs(Path(directory)))
        assert result["condition_count"] == 3
        assert result["regional_node_count"] == 3
        assert result["split_case_counts"]["formal"] == {
            "train": 1,
            "validation": 1,
            "test": 1,
        }
        assert result["new_physical_parameters"] == []


def test_changed_energy_node_volume_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        paths = write_inputs(Path(directory))
        with np.load(paths["energy_targets"], allow_pickle=False) as loaded:
            data = {name: loaded[name] for name in loaded.files}
        data["node_volume_m3"] = np.asarray([1.0, 2.0, 4.0])
        np.savez_compressed(paths["energy_targets"], **data)
        with pytest.raises(ValueError, match="different node volumes"):
            run(paths)


def test_changed_mass_condition_order_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        paths = write_inputs(Path(directory))
        with np.load(paths["mass_targets"], allow_pickle=False) as loaded:
            data = {name: loaded[name] for name in loaded.files}
        data["condition_id"] = np.asarray(["b", "a", "c"])
        np.savez_compressed(paths["mass_targets"], **data)
        with pytest.raises(ValueError, match="condition orders differ"):
            run(paths)


def test_nonfinite_energy_target_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        paths = write_inputs(Path(directory))
        with np.load(paths["energy_targets"], allow_pickle=False) as loaded:
            data = {name: loaded[name] for name in loaded.files}
        data["internal_energy_flow_W"] = data["internal_energy_flow_W"].astype(float)
        data["internal_energy_flow_W"][1, 0] = np.nan
        np.savez_compressed(paths["energy_targets"], **data)
        with pytest.raises(ValueError, match="non-finite"):
            run(paths)


def test_missing_cooling_wall_kind_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        paths = write_inputs(Path(directory))
        with np.load(paths["energy_targets"], allow_pickle=False) as loaded:
            data = {name: loaded[name] for name in loaded.files}
        data["boundary_kind_name"] = np.asarray(["fluid:inlet", "solid:wall"])
        np.savez_compressed(paths["energy_targets"], **data)
        with pytest.raises(ValueError, match="cooling wall"):
            run(paths)
