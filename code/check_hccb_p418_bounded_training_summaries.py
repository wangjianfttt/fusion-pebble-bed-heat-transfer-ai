#!/usr/bin/env python3
"""Check the three completed bounded graph--Transformer summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = (
    ("data-only", "formal_data_only", "data_only"),
    ("physics-constrained", "formal", "energy_and_flux"),
    ("factorized", "formal_factorized", "energy_and_flux"),
)


def check_summary(
    path: Path,
    *,
    label: str,
    run_role: str,
    physics_mode: str,
) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} training summary is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = {
        "status": payload.get("status"),
        "run_role": payload.get("run_role"),
        "physics_mode": payload.get("physics_mode"),
        "split_name": payload.get("split_name"),
        "temperature_output_mode": payload.get("architecture", {}).get(
            "temperature_output_mode"
        ),
    }
    expected = {
        "status": "completed_p418_spatiotemporal_regional_operator",
        "run_role": run_role,
        "physics_mode": physics_mode,
        "split_name": "pair_disjoint_stress_test",
        "temperature_output_mode": "literature_bounded_residual",
    }
    mismatches = {
        name: {"actual": actual[name], "expected": wanted}
        for name, wanted in expected.items()
        if actual[name] != wanted
    }
    if mismatches:
        raise ValueError(f"unexpected {label} training summary values: {mismatches}")
    return {
        "label": label,
        "path": str(path.resolve()),
        **actual,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-only", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--factorized", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = (args.data_only, args.physics, args.factorized)
    records = [
        check_summary(
            path.resolve(),
            label=label,
            run_role=run_role,
            physics_mode=physics_mode,
        )
        for path, (label, run_role, physics_mode) in zip(paths, EXPECTED)
    ]
    result = {
        "status": "p418_three_bounded_graph_transformer_summaries_checked",
        "records": records,
        "new_physical_parameters": [],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
