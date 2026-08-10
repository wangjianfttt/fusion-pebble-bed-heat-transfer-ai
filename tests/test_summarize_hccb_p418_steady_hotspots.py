#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from summarize_hccb_p418_steady_hotspots import (  # noqa: E402
    adjacent_movements,
    hotspot_records,
)


def add_case(
    root: Path,
    name: str,
    inlet_temperature: float,
    temperature: list[float],
    centroid: list[list[float]],
) -> None:
    case = root / name
    sample_dir = case / "training_sample_200_schema3"
    sample_dir.mkdir(parents=True)
    sample = sample_dir / "fields_and_topology.npz"
    np.savez(
        sample,
        solid_temperature_K=np.asarray(temperature, dtype=np.float64),
        solid_cell_centroid_m=np.asarray(centroid, dtype=np.float64),
    )
    (case / "formal_sample_complete.json").write_text(
        json.dumps({"time": "200", "training_sample": str(sample)}),
        encoding="utf-8",
    )
    (case / "cht_result_summary_200.json").write_text(
        json.dumps(
            {
                "physical_conditions": {
                    "inlet_velocity_m_s": 0.05,
                    "inlet_temperature_K": inlet_temperature,
                    "solid_heat_source_W_m3": 4.85e6,
                },
                "temperature": {"solid_maximum_K": max(temperature)},
            }
        ),
        encoding="utf-8",
    )


def test_hotspot_records_and_adjacent_movement() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        add_case(
            root,
            "low",
            300.0,
            [600.0, 620.0],
            [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]],
        )
        add_case(
            root,
            "high",
            500.0,
            [630.0, 625.0],
            [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]],
        )
        records = hotspot_records(root)
        assert len(records) == 2
        low = next(item for item in records if item["condition_id"] == "low")
        high = next(item for item in records if item["condition_id"] == "high")
        assert low["solid_hot_cell_index"] == 1
        assert high["solid_hot_cell_index"] == 0
        movements = adjacent_movements(records)
        temperature = next(
            item
            for item in movements
            if item["varied_factor"] == "inlet_temperature_K"
        )
        assert np.isclose(temperature["hotspot_distance_m"], 0.001)
        assert np.isclose(temperature["solid_maximum_temperature_change_K"], 10.0)
        assert not temperature["same_hot_cell"]
