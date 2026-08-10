#!/usr/bin/env python3
"""Apply an evidence-backed steady iteration endpoint to unfinished P418 cases."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def replace_end_time(path: Path, end_time: int) -> None:
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"(?m)^\s*endTime\s+[^;]+;",
        f"endTime {end_time};",
        text,
    )
    if count != 1:
        raise ValueError(f"expected one endTime in {path}, found {count}")
    path.write_text(updated, encoding="utf-8")


def latest_log_time(path: Path) -> float | None:
    if not path.is_file():
        return None
    values = re.findall(r"Time\s*=\s*([0-9.]+)s", path.read_text(encoding="utf-8", errors="replace"))
    return None if not values else float(values[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--end-time", type=int, required=True)
    parser.add_argument("--evidence-summary", type=Path, required=True)
    args = parser.parse_args()
    evidence_path = args.evidence_summary.resolve()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    accepted_statuses = {
        "steady_iteration_endpoint_comparison_complete",
        "steady_end_time_comparison_complete",  # historical result compatibility
    }
    if evidence.get("status") not in accepted_statuses:
        raise ValueError("steady-iteration endpoint comparison is incomplete")
    recommendation = evidence.get(
        "recommended_steady_end_iteration",
        evidence.get("recommended_steady_end_time_s", -1),
    )
    if int(float(recommendation)) != args.end_time:
        raise ValueError("requested steady iteration differs from the evidence summary")
    if int(evidence.get("completed_reference_case_count", 0)) < 5:
        raise ValueError("fewer than five completed reference cases")
    if int(evidence.get("decomposed_full_field_case_count", 0)) < 3:
        raise ValueError("fewer than three decomposed full-field cases")
    matrix = args.matrix_root.resolve()
    updated: list[str] = []
    skipped_completed: list[str] = []
    for case in sorted(matrix.glob("u*_T*_q*")):
        if (case / "formal_sample_complete.json").is_file():
            skipped_completed.append(case.name)
            continue
        current_time = latest_log_time(case / "log.foamMultiRun.formal")
        if current_time is not None and current_time >= args.end_time:
            raise ValueError(
                f"{case.name} has already reached {current_time}s, not below {args.end_time}s"
            )
        replace_end_time(case / "system/controlDict", args.end_time)
        metadata_path = case / "cht_smoke_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["end_time"] = args.end_time
        metadata["steady_iteration_end"] = args.end_time
        metadata["solver_time_semantics"] = "steady_iteration_index"
        metadata["physical_time_s"] = None
        metadata["steady_end_time_evidence"] = str(evidence_path)
        metadata["steady_end_time_is_numerical_not_physical"] = True
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        snapshot_path = case / "transient_snapshot_plan.json"
        if snapshot_path.is_file():
            plan = json.loads(snapshot_path.read_text(encoding="utf-8"))
            snapshots = plan.get("steady_snapshot_iterations", plan.get("snapshot_times_s", []))
            plan["steady_snapshot_iterations"] = [
                value for value in snapshots if float(value) <= args.end_time
            ]
            plan.pop("snapshot_times_s", None)
            plan["steady_endpoint_iteration"] = args.end_time
            plan["solver_time_semantics"] = "steady_iteration_index"
            plan["physical_time_s"] = None
            plan["steady_end_time_evidence"] = str(evidence_path)
            snapshot_path.write_text(
                json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        updated.append(case.name)
    record = {
        "status": "unfinished_p418_cases_use_evidence_backed_steady_endpoint",
        "steady_end_iteration": args.end_time,
        "solver_time_semantics": "steady_iteration_index",
        "physical_time_s": None,
        "updated_case_count": len(updated),
        "updated_cases": updated,
        "completed_300_s_case_count": len(skipped_completed),
        "completed_300_s_cases": skipped_completed,
        "evidence_summary": str(evidence_path),
        "physical_step_response_duration_changed": False,
        "new_physical_parameters": [],
    }
    output = matrix / "steady_end_time_update.json"
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
