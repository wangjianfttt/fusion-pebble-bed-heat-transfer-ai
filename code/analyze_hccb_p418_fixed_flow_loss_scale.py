#!/usr/bin/env python3
"""Quantify the effective contribution of the three fixed-flow loss terms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from train_hccb_p418_spatiotemporal_regional_operator import FORMAL_TRAINING


LOSS_KEYS = (
    "temperature_data_loss",
    "reference_edge_flux_loss",
    "projection_aware_energy_loss",
)

WEIGHT_KEYS = {
    "temperature_data_loss": "data_weight",
    "reference_edge_flux_loss": "edge_flux_weight",
    "projection_aware_energy_loss": "energy_weight",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contribution_record(row: dict[str, object]) -> dict[str, object]:
    contributions = {
        name: float(row[name]) * float(FORMAL_TRAINING[WEIGHT_KEYS[name]])
        for name in LOSS_KEYS
    }
    if any(not math.isfinite(value) or value < 0.0 for value in contributions.values()):
        raise ValueError("loss history contains invalid values")
    total = sum(contributions.values())
    if total <= 0.0:
        raise ValueError("weighted loss contribution is not positive")
    positive = [value for value in contributions.values() if value > 0.0]
    return {
        "epoch": int(row["epoch"]),
        "raw_losses": {name: float(row[name]) for name in LOSS_KEYS},
        "registered_weights": {
            name: float(FORMAL_TRAINING[WEIGHT_KEYS[name]]) for name in LOSS_KEYS
        },
        "weighted_contributions": contributions,
        "weighted_fraction": {
            name: value / total for name, value in contributions.items()
        },
        "weighted_dynamic_range": (
            max(positive) / min(positive) if positive else math.inf
        ),
        "validation_solid_temperature_RMSE_K": float(
            row["validation_solid_temperature_RMSE_K"]
        ),
        "validation_projection_aware_energy_normalized_RMSE": float(
            row["validation_projection_aware_energy_normalized_RMSE"]
        ),
        "validation_selection_score": float(row["validation_selection_score"]),
    }


def analyze_history(
    history: list[dict[str, object]], recent_epochs: int
) -> dict[str, object]:
    if not history:
        raise ValueError("training checkpoint has no history")
    if recent_epochs <= 0:
        raise ValueError("recent epoch count must be positive")
    records = [contribution_record(row) for row in history]
    recent = records[-min(recent_epochs, len(records)) :]
    fractions = {
        name: np.asarray(
            [row["weighted_fraction"][name] for row in recent], dtype=float
        )
        for name in LOSS_KEYS
    }
    dynamic_range = np.asarray(
        [row["weighted_dynamic_range"] for row in recent], dtype=float
    )
    return {
        "completed_epochs": len(records),
        "latest": records[-1],
        "recent_epoch_count": len(recent),
        "recent_weighted_fraction_median": {
            name: float(np.median(values)) for name, values in fractions.items()
        },
        "recent_weighted_fraction_minimum": {
            name: float(values.min()) for name, values in fractions.items()
        },
        "recent_weighted_fraction_maximum": {
            name: float(values.max()) for name, values in fractions.items()
        },
        "recent_weighted_dynamic_range_median": float(
            np.median(dynamic_range)
        ),
        "recent_weighted_dynamic_range_maximum": float(dynamic_range.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recent-epochs", type=int, default=10)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    candidate_source = args.candidate_source.resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    result = {
        "status": "p418_fixed_flow_loss_scale_diagnosed",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "candidate_source": str(candidate_source),
        "candidate_source_sha256": sha256(candidate_source),
        "fixed_weight_source": (
            "train_hccb_p418_spatiotemporal_regional_operator.FORMAL_TRAINING"
        ),
        "new_physical_parameters": [],
        **analyze_history(list(payload.get("history", [])), args.recent_epochs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
