#!/usr/bin/env python3
"""Collect integral boundary heat flows from completed P418 OpenFOAM cases."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def result_path_matches_condition(result_case: str, condition_id: str) -> bool:
    """Accept the matrix name or its archived run directory with a numeric suffix."""
    if result_case == condition_id:
        return True
    prefix = f"{condition_id}_"
    return result_case.startswith(prefix) and result_case[len(prefix) :].isdigit()


def finite_mapping(values: object, *, label: str) -> dict[str, float]:
    if not isinstance(values, dict) or not values:
        raise ValueError(f"{label} is missing or empty")
    converted = {str(key): float(value) for key, value in values.items()}
    if not all(math.isfinite(value) for value in converted.values()):
        raise ValueError(f"{label} contains a non-finite value")
    return converted


def build_summary(matrix_root: Path, expected_case_count: int | None) -> dict[str, object]:
    markers = sorted(matrix_root.glob("*/formal_sample_complete.json"))
    if not markers:
        raise ValueError("no completed P418 cases were found")
    if expected_case_count is not None and len(markers) != expected_case_count:
        raise ValueError(
            f"completed case count {len(markers)} != {expected_case_count}"
        )

    cases: list[dict[str, object]] = []
    for marker_path in markers:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        condition_id = str(marker["condition_id"])
        time_name = str(marker["time"])
        if not bool(marker.get("solver_finished")):
            raise ValueError(f"{condition_id}: completion marker is not solved")
        result_path = marker_path.parent / f"cht_result_summary_{time_name}.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result_case = Path(str(result.get("case", ""))).name
        if not result_path_matches_condition(result_case, condition_id):
            raise ValueError(f"{condition_id}: result summary case mismatch")
        if not bool(result.get("solver_finished")):
            raise ValueError(f"{condition_id}: result summary is not solved")
        heat = result["heat_balance"]
        generated = float(heat["solid_generated_power_W"])
        if not math.isfinite(generated) or generated <= 0.0:
            raise ValueError(f"{condition_id}: invalid generated power")
        fluid = finite_mapping(
            heat["all_fluid_boundary_conductive_heat_flows_W"],
            label=f"{condition_id} fluid boundary heat",
        )
        solid = finite_mapping(
            heat["all_solid_boundary_conductive_heat_flows_W"],
            label=f"{condition_id} solid boundary heat",
        )
        for key in ("fluid_to_solid",):
            if key not in fluid:
                raise ValueError(f"{condition_id}: fluid boundary lacks {key}")
        for key in ("solid_to_fluid", "coolingWall"):
            if key not in solid:
                raise ValueError(f"{condition_id}: solid boundary lacks {key}")
        solid_balance = abs(generated + sum(solid.values())) / generated
        cases.append(
            {
                "condition_id": condition_id,
                "source_final_time_s": float(time_name),
                "generated_power_W": generated,
                "fluid_boundary_heat_flow_into_region_W": fluid,
                "solid_boundary_heat_flow_into_region_W": solid,
                "interface_pair_difference_W": (
                    fluid["fluid_to_solid"] + solid["solid_to_fluid"]
                ),
                "solid_balance_relative": solid_balance,
                "combined_conductive_sum_W": sum(fluid.values()) + sum(solid.values()),
                "source_result_summary": str(result_path),
            }
        )

    maximum_interface_pair_difference = max(
        abs(float(row["interface_pair_difference_W"])) for row in cases
    )
    maximum_solid_balance = max(
        float(row["solid_balance_relative"]) for row in cases
    )
    return {
        "status": "p418_completed_boundary_heat_summary_ready",
        "case_count": len(cases),
        "cases": cases,
        "maximum_interface_pair_difference_W": maximum_interface_pair_difference,
        "maximum_solid_balance_relative": maximum_solid_balance,
        "method": (
            "Integral OpenFOAM-13 boundary heat flows read from each completed "
            "cht_result_summary_<time>.json; no heat-transfer coefficient is fitted."
        ),
        "sign_definition": "positive boundary heat flow enters the selected region",
        "new_physical_parameters": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-case-count", type=int)
    args = parser.parse_args()

    payload = build_summary(
        args.matrix_root.resolve(),
        args.expected_case_count,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
