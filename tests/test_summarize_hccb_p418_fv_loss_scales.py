#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from hccb_p418_comparison_contract import sha256_file  # noqa: E402
from summarize_hccb_p418_fv_loss_scales import summarize  # noqa: E402


def write_inputs(root: Path) -> dict[str, Path]:
    ids = np.asarray(["train", "validation", "test", "unused"])
    factors = np.asarray([1.0, 10.0, 100.0, 1000.0])
    state = root / "state.npz"
    np.savez_compressed(
        state,
        condition_id=ids,
        node_type=np.asarray([0, 0, 1]),
    )
    mass = root / "mass.npz"
    np.savez_compressed(
        mass,
        condition_id=ids,
        internal_owner=np.asarray([0]),
        internal_neighbour=np.asarray([1]),
        boundary_owner=np.asarray([0, 1]),
        boundary_patch=np.asarray([0, 1]),
        internal_mass_flow_kg_s=factors[:, None],
        boundary_mass_flow_kg_s=np.stack((-factors, factors), axis=1),
    )
    energy = root / "energy.npz"
    base_internal = np.asarray([2.0, 1.0])
    base_boundary = np.asarray([-2.0, 1.0, 2.0])
    base_source = np.asarray([0.0, 0.0, 1.0])
    np.savez_compressed(
        energy,
        condition_id=ids,
        internal_owner=np.asarray([0, 1]),
        internal_neighbour=np.asarray([1, 2]),
        boundary_owner=np.asarray([0, 1, 2]),
        internal_energy_flow_W=factors[:, None] * base_internal,
        boundary_energy_flow_W=factors[:, None] * base_boundary,
        node_source_power_W=factors[:, None] * base_source,
    )
    split = root / "split.json"
    split.write_text(
        json.dumps(
            {
                "splits": {
                    "formal": {
                        "train": ["train"],
                        "validation": ["validation"],
                        "test": ["test"],
                        "unused": ["unused"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    statistics = root / "statistics.json"
    statistics.write_text(
        json.dumps(
            {
                "splits": {
                    "formal": {
                        "train_conditions": ["train"],
                        "validation_conditions": ["validation"],
                        "test_conditions": ["test"],
                        "unused_conditions": ["unused"],
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
        "split_name": "formal",
    }


def test_scales_match_finite_volume_orientation() -> None:
    with tempfile.TemporaryDirectory() as directory:
        payload = summarize(**write_inputs(Path(directory)))
    scales = payload["training_only_normalization"]["scales"]
    assert np.isclose(scales["internal_mass_flow_rms_kg_s"], 1.0)
    assert np.isclose(scales["boundary_mass_flow_rms_kg_s"], 1.0)
    assert np.isclose(scales["regional_incident_mass_rms_kg_s"], 2.0)
    assert np.isclose(scales["regional_incident_energy_rms_W"], 4.0)
    assert payload["training_only_normalization"][
        "validation_or_test_values_used_in_scale"
    ] is False
    for row in payload["cases"]:
        assert np.isclose(row["target_global_mass_imbalance_over_inlet"], 0.0)
        assert np.isclose(
            row["target_global_energy_imbalance_over_generated_power"], 0.0
        )
    assert payload["unused_conditions"] == {
        "condition_count": 1,
        "condition_ids": ["unused"],
    }
    assert payload["cases"][-1]["role"] == "unused"


def test_validation_and_test_magnitudes_do_not_change_training_scales() -> None:
    with tempfile.TemporaryDirectory() as directory:
        paths = write_inputs(Path(directory))
        first = summarize(**paths)["training_only_normalization"]["scales"]
        for key in ("mass_targets", "energy_targets"):
            with np.load(paths[key], allow_pickle=False) as loaded:
                data = {name: loaded[name] for name in loaded.files}
            for name in data:
                if name.endswith(("_kg_s", "_W")) and data[name].ndim >= 2:
                    data[name] = data[name].astype(np.float64)
                    data[name][1:] *= 1.0e4
            np.savez_compressed(paths[key], **data)
        second = summarize(**paths)["training_only_normalization"]["scales"]
    assert first == second
