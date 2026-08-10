#!/usr/bin/env python3
"""Check whether every MPI partition contains the fields needed for restart."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REQUIRED_FIELDS = (
    "fluid/T",
    "fluid/U",
    "fluid/p",
    "fluid/p_rgh",
    "fluid/phi",
    "solid/T",
    "uniform/time",
)


def numeric_time(name: str) -> float | None:
    try:
        value = float(name)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def check_case(case: Path, nprocs: int) -> dict[str, object]:
    processors = [case / f"processor{rank}" for rank in range(nprocs)]
    absent_processors = [
        rank for rank, processor in enumerate(processors) if not processor.is_dir()
    ]
    if absent_processors:
        raise FileNotFoundError(f"missing processor directories: {absent_processors}")

    time_names: set[str] = set()
    for processor in processors:
        for path in processor.iterdir():
            if path.is_dir() and numeric_time(path.name) is not None:
                time_names.add(path.name)

    records: list[dict[str, object]] = []
    for time_name in sorted(time_names, key=lambda name: (float(name), name)):
        missing: list[dict[str, object]] = []
        present_rank_count = 0
        for rank, processor in enumerate(processors):
            time_dir = processor / time_name
            if time_dir.is_dir():
                present_rank_count += 1
            absent_fields = [
                field for field in REQUIRED_FIELDS if not (time_dir / field).is_file()
            ]
            if absent_fields:
                missing.append({"rank": rank, "fields": absent_fields})
        records.append(
            {
                "time_name": time_name,
                "time_s": float(time_name),
                "complete": not missing,
                "present_rank_count": present_rank_count,
                "missing_rank_count": len(missing),
                "missing": missing,
            }
        )

    complete = [record for record in records if record["complete"]]
    incomplete = [record for record in records if not record["complete"]]
    return {
        "status": "checked_parallel_restart_fields",
        "case": str(case.resolve()),
        "nprocs": nprocs,
        "required_fields": list(REQUIRED_FIELDS),
        "time_count": len(records),
        "complete_time_count": len(complete),
        "incomplete_time_count": len(incomplete),
        "latest_complete_time_name": complete[-1]["time_name"] if complete else None,
        "latest_complete_time_s": complete[-1]["time_s"] if complete else None,
        "first_incomplete_time_name": incomplete[0]["time_name"] if incomplete else None,
        "first_incomplete_time_s": incomplete[0]["time_s"] if incomplete else None,
        "times": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--nprocs", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.nprocs <= 0:
        raise ValueError("nprocs must be positive")
    payload = check_case(args.case.resolve(), args.nprocs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "latest_complete_time_name",
        "first_incomplete_time_name",
        "complete_time_count",
        "incomplete_time_count",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
