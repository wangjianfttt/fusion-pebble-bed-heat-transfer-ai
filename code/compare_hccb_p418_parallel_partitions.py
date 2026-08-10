#!/usr/bin/env python3
"""Compare one OpenFOAM time record across two MPI decompositions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tarfile
from pathlib import Path
from typing import Any


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_text(archive: Path, member: str) -> str:
    with tarfile.open(archive, "r:*") as handle:
        source = handle.extractfile(member)
        if source is None:
            raise ValueError(f"archive member is not a regular file: {member}")
        return source.read().decode("utf-8", errors="strict")


def archive_json(archive: Path, member: str) -> dict[str, Any]:
    return json.loads(archive_text(archive, member))


def last_number(block: str, pattern: str, label: str) -> float:
    matches = re.findall(pattern, block, flags=re.MULTILINE)
    if not matches:
        raise ValueError(f"missing {label} in selected OpenFOAM time block")
    return float(matches[-1])


def time_block(log_text: str, time_s: float) -> str:
    target = f"{time_s:g}"
    match = re.search(
        rf"^\s*Time\s*=\s*{re.escape(target)}s\s*$",
        log_text,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"Time = {target}s is absent from the reference log")
    next_match = re.search(
        r"^\s*Time\s*=\s*" + FLOAT + r"s\s*$",
        log_text[match.end() :],
        flags=re.MULTILINE,
    )
    stop = match.end() + next_match.start() if next_match else len(log_text)
    return log_text[match.end() : stop]


def parse_reference_time(log_text: str, time_s: float) -> dict[str, Any]:
    block = time_block(log_text, time_s)
    inlet_mass = last_number(
        block, rf"sum\(inlet\) of phi\s*=\s*({FLOAT})", "inlet mass flow"
    )
    outlet_mass = last_number(
        block, rf"sum\(outlet\) of phi\s*=\s*({FLOAT})", "outlet mass flow"
    )
    inlet_pressure = last_number(
        block,
        rf"areaAverage\(inlet\) of p\s*=\s*({FLOAT})",
        "inlet pressure",
    )
    outlet_pressure = last_number(
        block,
        rf"areaAverage\(outlet\) of p\s*=\s*({FLOAT})",
        "outlet pressure",
    )
    result = {
        "reported_iteration_s": time_s,
        "flow": {
            "inlet_mass_flow_kg_s": inlet_mass,
            "outlet_mass_flow_kg_s": outlet_mass,
            "relative_mass_difference": abs(abs(outlet_mass) - abs(inlet_mass))
            / max(abs(inlet_mass), 1e-300),
            "inlet_average_pressure_Pa": inlet_pressure,
            "outlet_average_pressure_Pa": outlet_pressure,
            "pressure_drop_Pa": inlet_pressure - outlet_pressure,
        },
        "temperature": {
            "outlet_average_K": last_number(
                block,
                rf"areaAverage\(outlet\) of T\s*=\s*({FLOAT})",
                "outlet temperature",
            ),
            "solid_maximum_K": last_number(
                block, rf"max\(all\) of T\s*=\s*({FLOAT})", "maximum solid temperature"
            ),
        },
        "heat_balance": {
            "cooling_wall_heat_flow_W": last_number(
                block,
                rf"areaIntegrate\(coolingWall\) of wallHeatFlux\s*=\s*({FLOAT})",
                "cooling-wall heat flow",
            ),
            "relative_energy_difference": None,
            "relative_energy_difference_unavailable_reason": (
                "The solver log does not contain every conductive boundary term "
                "needed to reconstruct the full energy residual at this time."
            ),
        },
    }
    return result


def nested(record: dict[str, Any], path: str) -> float:
    value: Any = record
    for key in path.split("."):
        value = value[key]
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite value at {path}")
    return value


def compare(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    reference_np: int,
    candidate_np: int,
) -> dict[str, Any]:
    quantities = {
        "outlet_mass_flow_kg_s": "flow.outlet_mass_flow_kg_s",
        "pressure_drop_Pa": "flow.pressure_drop_Pa",
        "outlet_average_temperature_K": "temperature.outlet_average_K",
        "maximum_solid_temperature_K": "temperature.solid_maximum_K",
        "cooling_wall_heat_flow_W": "heat_balance.cooling_wall_heat_flow_W",
    }
    rows = []
    for quantity, path in quantities.items():
        left = nested(reference, path)
        right = nested(candidate, path)
        signed = right - left
        relative = abs(signed) / max(abs(left), 1e-300)
        rows.append(
            {
                "quantity": quantity,
                f"mpi_{reference_np}": left,
                f"mpi_{candidate_np}": right,
                "signed_difference": signed,
                "relative_difference": relative,
            }
        )
    largest = max(rows, key=lambda row: row["relative_difference"])
    return {
        "direct_time_comparison": rows,
        "largest_direct_relative_difference": {
            "quantity": largest["quantity"],
            "value": largest["relative_difference"],
        },
        f"mpi_{reference_np}_relative_mass_difference": nested(
            reference, "flow.relative_mass_difference"
        ),
        f"mpi_{candidate_np}_relative_mass_difference": nested(
            candidate, "flow.relative_mass_difference"
        ),
        f"mpi_{candidate_np}_relative_energy_difference": nested(
            candidate, "heat_balance.relative_energy_difference"
        ),
        f"mpi_{reference_np}_relative_energy_difference_at_selected_time": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-archive", type=Path, required=True)
    parser.add_argument("--reference-log-member", required=True)
    parser.add_argument("--reference-summary-member", required=True)
    parser.add_argument("--reference-np", type=int, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--candidate-np", type=int, required=True)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.reference_np < 1 or args.candidate_np < 1:
        raise SystemExit("MPI process counts must be positive")
    reference_summary = archive_json(
        args.reference_archive, args.reference_summary_member
    )
    reference = parse_reference_time(
        archive_text(args.reference_archive, args.reference_log_member), args.time
    )
    candidate = json.loads(args.candidate_summary.read_text(encoding="utf-8"))
    if float(candidate["reported_iteration"]) != args.time:
        raise ValueError("candidate summary time does not match --time")

    reference_conditions = reference_summary["physical_conditions"]
    candidate_conditions = candidate["physical_conditions"]
    same_conditions = reference_conditions == candidate_conditions
    if not same_conditions:
        raise ValueError("reference and candidate physical conditions differ")

    comparison = compare(
        reference, candidate, args.reference_np, args.candidate_np
    )
    comparison.update(
        {
            "status": "parallel_partition_comparison_complete",
            "comparison_time_s": args.time,
            "same_physical_inputs": same_conditions,
            "reference": {
                "mpi_process_count": args.reference_np,
                "archive": str(args.reference_archive),
                "archive_sha256": sha256(args.reference_archive),
                "solver_log_member": args.reference_log_member,
                "summary_member": args.reference_summary_member,
                "full_run_finished": bool(reference_summary["solver_finished"]),
            },
            "candidate": {
                "mpi_process_count": args.candidate_np,
                "summary": str(args.candidate_summary),
                "summary_sha256": sha256(args.candidate_summary),
                "solver_finished": bool(candidate["solver_finished"]),
            },
            "interpretation": (
                "This short-time comparison checks sensitivity to MPI partitioning. "
                "It is not a converged heat-transfer result. The full selected-time "
                "energy residual is reported only for the candidate because the "
                "archived reference log lacks the required boundary terms."
            ),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reference_selected_time.json").write_text(
        json.dumps(reference, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "parallel_partition_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
