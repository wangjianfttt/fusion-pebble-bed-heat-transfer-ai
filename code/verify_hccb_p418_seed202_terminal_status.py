#!/usr/bin/env python3
"""Verify the scientific boundary of the partial seed202 OpenFOAM matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ACCEPTED_CASES = {
    "u0p15_T700_q6p85",
    "u0p05_T300_q4p85",
    "u0p05_T300_q8p85",
    "u0p05_T900_q4p85",
    "u0p05_T900_q8p85",
    "u0p25_T300_q4p85",
    "u0p25_T300_q8p85",
}
FAILED_CASES = {
    "u0p25_T900_q4p85",
    "u0p25_T900_q8p85",
}
FINITE_FIELDS = (
    "mass_relative_difference",
    "energy_relative_difference",
    "outlet_temperature_K",
    "maximum_solid_temperature_K",
    "pressure_drop_Pa",
)


def verify(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "seed202_cloud_steady_terminal_status":
        raise ValueError("unexpected seed202 status")
    if int(payload.get("packing_seed", -1)) != 202:
        raise ValueError("terminal status is not for seed202")
    if int(payload.get("steady_iteration_target", -1)) != 200:
        raise ValueError("steady iteration target must be 200")

    accepted = payload.get("accepted_cases")
    failed = payload.get("failed_cases")
    if not isinstance(accepted, list) or not isinstance(failed, list):
        raise ValueError("accepted_cases and failed_cases must be lists")
    if len(accepted) != 7 or int(payload.get("accepted_case_count", -1)) != 7:
        raise ValueError("exactly seven seed202 cases must be accepted")
    if len(failed) != 2 or int(payload.get("failed_case_count", -1)) != 2:
        raise ValueError("exactly two seed202 cases must be failed")
    if {record.get("case_id") for record in accepted} != ACCEPTED_CASES:
        raise ValueError("accepted seed202 case set differs from the declared partial matrix")
    if {record.get("case_id") for record in failed} != FAILED_CASES:
        raise ValueError("failed seed202 case set differs from the declared partial matrix")

    jobs = [str(record.get("job", "")) for record in accepted + failed]
    if any(not job for job in jobs) or len(jobs) != len(set(jobs)):
        raise ValueError("seed202 job identifiers are missing or duplicated")
    for record in accepted:
        digest = str(record.get("summary_sha256", ""))
        if len(digest) != 64:
            raise ValueError(f"{record['case_id']} has no valid summary SHA256")
        for field in FINITE_FIELDS:
            value = float(record.get(field))
            if not math.isfinite(value):
                raise ValueError(f"{record['case_id']} has non-finite {field}")
        if float(record["mass_relative_difference"]) < 0.0:
            raise ValueError("mass relative difference must be nonnegative")
        if float(record["energy_relative_difference"]) < 0.0:
            raise ValueError("energy relative difference must be nonnegative")

    for record in failed:
        if record.get("slurm_status") != "FAILED/1:0":
            raise ValueError(f"{record['case_id']} must remain a failed case")
        pressure_range = record.get("viscosity_table_pressure_range_Pa")
        if not isinstance(pressure_range, list) or len(pressure_range) != 2:
            raise ValueError("viscosity-table pressure range is missing")
        pressure = float(record.get("pressure_at_failure_Pa"))
        if not pressure > float(pressure_range[1]):
            raise ValueError("failed pressure does not exceed the viscosity-table range")
        if record.get("reason") != "pressure outside the declared helium viscosity table":
            raise ValueError("failed case has an unexpected reason")

    return {
        "status": "verified_partial_seed202_terminal_status",
        "packing_seed": 202,
        "accepted_case_count": 7,
        "failed_case_count": 2,
        "accepted_case_ids": sorted(ACCEPTED_CASES),
        "failed_case_ids": sorted(FAILED_CASES),
        "complete_nine_case_comparison": False,
        "new_physical_parameters": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.input.resolve())
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
