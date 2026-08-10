#!/usr/bin/env python3
"""Build a compact nine-case seed202 result record from verified steady fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value}")
    return number


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--recovery-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=9)
    args = parser.parse_args()

    matrix_root = args.matrix_root.resolve()
    recovery_path = args.recovery_record.resolve()
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    if recovery.get("status") != "seed202_schema3_matrix_ready":
        raise ValueError("unexpected seed202 recovery status")
    cases = recovery.get("cases")
    if not isinstance(cases, list) or len(cases) != args.expected_cases:
        raise ValueError(
            f"expected {args.expected_cases} recovered cases, found "
            f"{len(cases) if isinstance(cases, list) else 'invalid'}"
        )
    if recovery.get("missing_conditions") or recovery.get("sha_mismatches"):
        raise ValueError("seed202 recovery record contains missing files or SHA errors")

    accepted: list[dict[str, object]] = []
    seen: set[str] = set()
    for record in cases:
        condition_id = str(record["condition_id"])
        if condition_id in seen:
            raise ValueError(f"duplicate condition id: {condition_id}")
        seen.add(condition_id)
        case_root = matrix_root / condition_id
        summary_path = case_root / "cht_result_summary_200.json"
        marker_path = case_root / "formal_sample_complete.json"
        heat_path = case_root / "boundary_heat_flows_200.json"
        for path in (summary_path, marker_path, heat_path):
            if not path.is_file():
                raise FileNotFoundError(f"missing seed202 result file: {path}")
        summary_sha = sha256(summary_path)
        if summary_sha != str(record["result_summary_sha256"]):
            raise ValueError(f"result summary SHA mismatch for {condition_id}")
        if sha256(marker_path) != str(record["marker_sha256"]):
            raise ValueError(f"completion marker SHA mismatch for {condition_id}")
        if sha256(heat_path) != str(record["boundary_heat_flows_sha256"]):
            raise ValueError(f"boundary heat-flow SHA mismatch for {condition_id}")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("solver_finished") is not True:
            raise ValueError(f"solver did not finish for {condition_id}")
        if float(summary.get("reported_iteration", -1)) != 200.0:
            raise ValueError(f"unexpected steady iteration for {condition_id}")
        if summary.get("all_reported_values_are_finite") is not True:
            raise ValueError(f"non-finite reported values for {condition_id}")
        flow = summary["flow"]
        temperature = summary["temperature"]
        heat = summary["heat_balance"]
        accepted.append(
            {
                "job": str(record["formal_source_job"]),
                "case_id": condition_id,
                "summary_sha256": summary_sha,
                "marker_sha256": str(record["marker_sha256"]),
                "mass_relative_difference": finite(
                    flow["relative_mass_difference"],
                    f"{condition_id} mass difference",
                ),
                "energy_relative_difference": finite(
                    heat["relative_energy_difference"],
                    f"{condition_id} energy difference",
                ),
                "outlet_temperature_K": finite(
                    temperature["outlet_average_K"],
                    f"{condition_id} outlet temperature",
                ),
                "maximum_solid_temperature_K": finite(
                    temperature["solid_maximum_K"],
                    f"{condition_id} maximum solid temperature",
                ),
                "pressure_drop_Pa": finite(
                    flow["pressure_drop_Pa"], f"{condition_id} pressure drop"
                ),
            }
        )

    accepted.sort(key=lambda item: str(item["case_id"]))
    output = {
        "status": "completed_seed202_nine_case_terminal_status",
        "packing_seed": 202,
        "matrix_root": str(matrix_root),
        "recovery_record": str(recovery_path),
        "accepted_case_count": len(accepted),
        "failed_case_count": 0,
        "accepted_cases": accepted,
        "failed_cases": [],
        "checks": {
            "all_registered_cases_present": len(accepted) == args.expected_cases,
            "all_solvers_finished_at_iteration_200": True,
            "all_reported_values_finite": True,
            "mass_energy_and_boundary_heat_records_present": True,
            "recovery_sha_values_recomputed": True,
            "old_failed_jobs_excluded": bool(
                recovery.get("checks", {}).get(
                    "old_failed_14356_6_14356_7_excluded"
                )
            ),
        },
        "new_physical_parameters": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
