#!/usr/bin/env python3
"""Summarize the numerical scale of the three fixed-flow training losses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


LOSS_FIELDS = {
    "temperature_data": "temperature_data_loss",
    "reference_edge_energy_flux": "reference_edge_flux_loss",
    "projection_aware_transient_energy": "projection_aware_energy_loss",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate(path: Path, candidate_id: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for candidate in payload.get("formal_candidates", []):
        if candidate.get("candidate_id") == candidate_id:
            return candidate
    raise ValueError(f"candidate not found: {candidate_id}")


def summarize_row(
    row: dict[str, object], candidate: dict[str, object]
) -> dict[str, object]:
    weights = {
        "temperature_data": float(candidate["temperature_data_weight"]),
        "reference_edge_energy_flux": float(
            candidate["reference_edge_energy_flux_weight"]
        ),
        "projection_aware_transient_energy": float(
            candidate["projection_aware_transient_energy_weight"]
        ),
    }
    raw = {name: float(row[field]) for name, field in LOSS_FIELDS.items()}
    if not all(math.isfinite(value) and value >= 0.0 for value in raw.values()):
        raise ValueError("loss history contains a non-finite or negative component")
    weighted = {name: raw[name] * weights[name] for name in raw}
    total = sum(weighted.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("weighted loss is not positive and finite")
    return {
        "epoch": int(row["epoch"]),
        "raw_loss_terms": raw,
        "weights": weights,
        "weighted_loss_terms": weighted,
        "weighted_total_recomputed": total,
        "reported_total_loss": float(row["total_loss"]),
        "relative_total_difference": abs(total - float(row["total_loss"]))
        / max(abs(float(row["total_loss"])), 1.0e-30),
        "objective_fraction": {
            name: value / total for name, value in weighted.items()
        },
        "validation_selection_score": float(row["validation_selection_score"]),
        "validation_fluid_temperature_RMSE_K": (
            float(row["validation_fluid_temperature_RMSE_K"])
            if row.get("validation_fluid_temperature_RMSE_K") is not None
            else None
        ),
        "validation_solid_temperature_RMSE_K": float(
            row["validation_solid_temperature_RMSE_K"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--candidate-id", default="fixed_registered_5_1_1"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    candidate_path = args.candidates.resolve()
    candidate = load_candidate(candidate_path, args.candidate_id)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    history = checkpoint.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("checkpoint contains no training history")
    rows = [summarize_row(dict(row), candidate) for row in history]
    if any(row["relative_total_difference"] > 1.0e-6 for row in rows):
        raise ValueError("recorded total loss is inconsistent with declared weights")
    best = min(rows, key=lambda row: row["validation_selection_score"])
    latest = rows[-1]
    payload = {
        "status": "p418_fixed_flow_loss_scale_preflight_complete",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "completed_epochs": len(rows),
        "candidate_source": str(candidate_path),
        "candidate_source_sha256": sha256(candidate_path),
        "fixed_candidate_id": args.candidate_id,
        "fixed_candidate_method": candidate["method"],
        "first_epoch": rows[0],
        "best_validation_epoch_so_far": best,
        "latest_epoch": latest,
        "interpretation": {
            "scope": (
                "Fractions describe numerical contributions to the declared "
                "training objective; they are not fractions of physical heat or energy."
            ),
            "formal_candidate_comparison_complete": False,
            "required_follow_up": (
                "Compare the four pre-registered fixed/ReLoBRaLo candidates on "
                "validation curves before reading the independent test curves."
            ),
        },
        "new_physical_parameters": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
