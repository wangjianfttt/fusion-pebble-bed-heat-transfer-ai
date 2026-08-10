#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from hccb_p418_fully_coupled_dataset import (  # noqa: E402
    load_index,
    load_sequence,
    selected_split,
    sequence_records,
    training_statistics,
)


def write_sequence(root: Path, sequence_id: str, offset: float) -> dict[str, object]:
    path = root / f"{sequence_id}.npz"
    state = np.zeros((3, 4, 5), dtype=np.float32)
    state[:, :2, :4] = offset
    state[:, :, 4] = 300.0 + offset
    np.savez_compressed(
        path,
        sequence_id=np.asarray(sequence_id),
        time_s=np.asarray([0.0, 0.1, 1.0]),
        condition_physical=np.arange(8, dtype=np.float64) + offset,
        state_physical=state,
        fluid_internal_mass_flux_kg_s=np.full((3, 5), 0.01 + offset),
        fluid_boundary_mass_flux_kg_s=np.full((3, 2), 0.02 + offset),
    )
    return {
        "sequence_id": sequence_id,
        "sequence_file": path.name,
        "complete": True,
    }


def write_index(root: Path, records: list[dict[str, object]]) -> Path:
    index = {
        "history_mode": "fully_coupled_flow_heat",
        "sequence_count": len(records),
        "state_names": ["Ux_m_s", "Uy_m_s", "Uz_m_s", "pressure_Pa", "temperature_K"],
        "condition_names": [
            "source_inlet_velocity_m_s",
            "source_inlet_temperature_K",
            "source_solid_heat_source_MW_m3",
            "target_inlet_velocity_m_s",
            "target_inlet_temperature_K",
            "target_solid_heat_source_MW_m3",
            "target_outlet_pressure_Pa",
            "target_cooling_wall_temperature_K",
        ],
        "sequences": records,
    }
    path = root / "dataset_index.json"
    path.write_text(json.dumps(index), encoding="utf-8")
    return path


def test_fully_coupled_loader_requires_time_dependent_flux(tmp_path: Path) -> None:
    record = write_sequence(tmp_path, "train", 0.0)
    index = load_index(write_index(tmp_path, [record]))
    loaded = load_sequence(tmp_path, sequence_records(index)["train"])
    assert loaded[2].shape == (3, 4, 5)
    assert loaded[3].shape == (3, 5)
    assert loaded[4].shape == (3, 2)


def test_training_statistics_ignore_validation_and_test_curves(tmp_path: Path) -> None:
    records = [
        write_sequence(tmp_path, "train", 0.0),
        write_sequence(tmp_path, "validation", 1000.0),
        write_sequence(tmp_path, "test", 2000.0),
    ]
    index = load_index(write_index(tmp_path, records))
    stats = training_statistics(
        tmp_path,
        sequence_records(index),
        ["train"],
        np.asarray([0, 0, 1, 1]),
    )
    np.testing.assert_allclose(stats["condition_mean"], np.arange(8))
    assert np.isclose(stats["internal_mass_flux_mean_kg_s"], 0.01)
    assert np.isclose(stats["boundary_mass_flux_mean_kg_s"], 0.02)
    assert stats["training_sequence_ids"] == ["train"]


def test_complete_curve_split_cannot_overlap(tmp_path: Path) -> None:
    split_path = tmp_path / "splits.json"
    split_path.write_text(
        json.dumps(
            {
                "splits": {
                    "bad": {
                        "train": ["a"],
                        "validation": ["b"],
                        "test": ["a"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    try:
        selected_split({"a", "b"}, split_path, "bad")
    except ValueError as error:
        assert "more than one role" in str(error)
    else:
        raise AssertionError("overlapping complete curves were accepted")
