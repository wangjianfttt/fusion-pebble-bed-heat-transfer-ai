#!/usr/bin/env python3
"""Compare full-graph repeated and factorized spatiotemporal update times."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_EQUAL = (
    "nodes",
    "edges",
    "time_points",
    "torch_num_threads",
    "physical_parameter_ids",
    "new_physical_parameters",
)


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeated-summary", type=Path, required=True)
    parser.add_argument("--factorized-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repeated = load(args.repeated_summary.resolve())
    factorized = load(args.factorized_summary.resolve())
    if repeated.get("spatial_temporal_mode") != "repeated_query_spatial":
        raise ValueError("reference result is not the repeated-query spatial model")
    if factorized.get("spatial_temporal_mode") != "factorized_static_spatial":
        raise ValueError("comparison result is not the factorized fixed-spatial model")
    differences = {
        key: (repeated.get(key), factorized.get(key))
        for key in REQUIRED_EQUAL
        if repeated.get(key) != factorized.get(key)
    }
    if differences:
        raise ValueError(f"timing inputs differ: {differences}")
    for name, result in (("repeated", repeated), ("factorized", factorized)):
        if not (
            result.get("loss_finite")
            and result.get("all_gradients_present")
            and result.get("all_gradients_finite")
            and float(result.get("initial_maximum_absolute_error", 1.0)) == 0.0
            and float(result.get("hydrodynamic_maximum_absolute_error", 1.0)) == 0.0
        ):
            raise ValueError(f"{name} full-graph calculation did not pass its physical checks")
    reference_seconds = float(repeated["elapsed_seconds"])
    factorized_seconds = float(factorized["elapsed_seconds"])
    if reference_seconds <= 0 or factorized_seconds <= 0:
        raise ValueError("elapsed times must be positive")
    payload = {
        "status": "completed_full_graph_spatiotemporal_efficiency_comparison",
        "nodes": repeated["nodes"],
        "edges": repeated["edges"],
        "time_points": repeated["time_points"],
        "torch_num_threads": repeated["torch_num_threads"],
        "repeated_query_elapsed_seconds": reference_seconds,
        "factorized_static_elapsed_seconds": factorized_seconds,
        "factorized_update_speedup": reference_seconds / factorized_seconds,
        "initial_state_and_hydrodynamics_exact": True,
        "both_backward_passes_finite": True,
        "new_physical_parameters": [],
        "scientific_scope": (
            "One forward, transient-physics evaluation and backward update on a repeated "
            "steady regional state. This measures implementation cost, not prediction accuracy."
        ),
        "accuracy_result_pending": "12 OpenFOAM physical thermal-step histories",
        "source_summaries": {
            "repeated": str(args.repeated_summary.resolve()),
            "factorized": str(args.factorized_summary.resolve()),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
