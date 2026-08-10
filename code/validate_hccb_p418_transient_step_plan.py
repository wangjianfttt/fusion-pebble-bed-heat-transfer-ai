#!/usr/bin/env python3
"""Verify that every planned transient step uses exact published P418 endpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FIELDS = ["inlet_velocity_m_s", "inlet_temperature_K", "solid_heat_source_MW_m3"]
FAMILY_FIELD = {
    "inlet_temperature_step": "inlet_temperature_K",
    "inlet_velocity_step": "inlet_velocity_m_s",
    "solid_heat_source_step": "solid_heat_source_MW_m3",
}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=root / "parameters/hccb_p418_transient_step_plan.json")
    parser.add_argument("--matrix", type=Path, default=root / "parameters/hccb_p418_model_splits.json")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    conditions = {row["condition_id"]: row for row in matrix["conditions"]}
    seen = set()
    family_counts: dict[str, int] = {}
    for sequence in plan["sequences"]:
        sequence_id = sequence["sequence_id"]
        if sequence_id in seen:
            raise ValueError(f"duplicate sequence_id: {sequence_id}")
        seen.add(sequence_id)
        family = sequence["family"]
        expected_changed = FAMILY_FIELD[family]
        source = conditions[sequence["source_condition_id"]]
        target = conditions[sequence["target_condition_id"]]
        changed = [field for field in FIELDS if float(source[field]) != float(target[field])]
        if changed != [expected_changed]:
            raise ValueError(f"{sequence_id} changes {changed}, expected only {expected_changed}")
        family_counts[family] = family_counts.get(family, 0) + 1
    if plan.get("new_physical_parameters") != []:
        raise ValueError("transient plan must not introduce physical parameters")
    summary = {
        "status": "p418_transient_step_plan_uses_exact_published_endpoints",
        "sequence_count": len(seen),
        "family_counts": family_counts,
        "source_doi": plan["source_doi"],
        "new_physical_parameters": [],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
