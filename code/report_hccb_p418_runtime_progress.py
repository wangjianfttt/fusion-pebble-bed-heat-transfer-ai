#!/usr/bin/env python3
"""Report simple runtime progress for the 60-case P418 matrix."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
from pathlib import Path


CLOCK_PATTERN = re.compile(r"ClockTime\s*=\s*([0-9.]+)\s*s")
SOLVER_ITERATION_PATTERN = re.compile(
    r"(?m)^Time\s*=\s*([0-9.eE+\-]+)s?\s*$"
)
RESUME_PATTERN = re.compile(
    r"resumed from complete parallel time\s+([0-9.eE+\-]+)\s+at\s+([^\s=]+)"
)
RESTART_FIELDS = (
    "fluid/T",
    "fluid/U",
    "fluid/p",
    "fluid/p_rgh",
    "solid/T",
    "uniform/time",
)


def final_clock_time(path: Path) -> float | None:
    if not path.is_file():
        return None
    values = [float(value) for value in CLOCK_PATTERN.findall(path.read_text(encoding="utf-8", errors="replace"))]
    return values[-1] if values else None


def accumulated_solver_clock_time(path: Path) -> float | None:
    """Sum the maximum cumulative clock time from each restart segment."""
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    restart_positions = [match.start() for match in RESUME_PATTERN.finditer(text)]
    bounds = [0, *restart_positions, len(text)]
    segment_maxima: list[float] = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        values = [float(value) for value in CLOCK_PATTERN.findall(text[start:end])]
        if values:
            segment_maxima.append(max(values))
    return sum(segment_maxima) if segment_maxima else None


def completed_case_postprocess_seconds(case: Path) -> float | None:
    marker_path = case / "formal_sample_complete.json"
    if not marker_path.is_file():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if "formal_finalization_seconds" not in marker:
        return None
    elapsed = float(marker["formal_finalization_seconds"])
    if elapsed < 0:
        return None
    return elapsed


def resumed_segment(path: Path) -> dict[str, float | str | None]:
    """Read steady-iteration labels after the last explicit restart marker."""
    if not path.is_file():
        return {
            "restart_iteration": None,
            "restart_wall_time": None,
            "current_iteration": None,
        }
    text = path.read_text(encoding="utf-8", errors="replace")
    resume_matches = list(RESUME_PATTERN.finditer(text))
    if resume_matches:
        last_resume = resume_matches[-1]
        segment = text[last_resume.end() :]
        restart_time = float(last_resume.group(1))
        restart_wall_time = last_resume.group(2)
    else:
        segment = text
        restart_time = None
        restart_wall_time = None
    iterations = [float(value) for value in SOLVER_ITERATION_PATTERN.findall(segment)]
    return {
        "restart_iteration": restart_time,
        "restart_wall_time": restart_wall_time,
        "current_iteration": iterations[-1] if iterations else restart_time,
    }


def latest_new_complete_parallel_iteration(
    case: Path,
    *,
    restart_iteration: float | None,
    restart_wall_time: str | None,
    parallel_ranks: int,
) -> float | None:
    """Return a post-restart iteration with complete, newly written rank fields."""
    processor0 = case / "processor0"
    if not processor0.is_dir():
        return restart_iteration
    restart_epoch = (
        dt.datetime.fromisoformat(restart_wall_time).timestamp()
        if restart_wall_time is not None
        else None
    )
    candidates: list[tuple[float, str]] = []
    for path in processor0.iterdir():
        if not path.is_dir():
            continue
        try:
            value = float(path.name)
        except ValueError:
            continue
        if restart_iteration is None or value > restart_iteration:
            candidates.append((value, path.name))
    for value, name in sorted(candidates, reverse=True):
        complete_and_new = True
        for rank in range(parallel_ranks):
            for field in RESTART_FIELDS:
                path = case / f"processor{rank}" / name / field
                if not path.is_file() or (
                    restart_epoch is not None and path.stat().st_mtime < restart_epoch
                ):
                    complete_and_new = False
                    break
            if not complete_and_new:
                break
        if complete_and_new:
            return value
    return restart_iteration


def latest_new_observed_iteration(
    case: Path,
    *,
    restart_iteration: float | None,
    restart_wall_time: str | None,
) -> float | None:
    """Return the latest post-restart processor0 iteration with a new file."""
    processor0 = case / "processor0"
    if not processor0.is_dir():
        return restart_iteration
    restart_epoch = (
        dt.datetime.fromisoformat(restart_wall_time).timestamp()
        if restart_wall_time is not None
        else None
    )
    candidates: list[tuple[float, Path]] = []
    for path in processor0.iterdir():
        if not path.is_dir():
            continue
        try:
            value = float(path.name)
        except ValueError:
            continue
        if restart_iteration is None or value > restart_iteration:
            candidates.append((value, path))
    for value, path in sorted(candidates, reverse=True):
        if any(
            item.is_file()
            and (restart_epoch is None or item.stat().st_mtime >= restart_epoch)
            for item in path.rglob("*")
        ):
            return value
    return restart_iteration


def active_case_rows(
    case_dirs: list[Path],
    *,
    now: dt.datetime,
    active_log_max_age_seconds: float,
    parallel_ranks: int,
) -> list[dict[str, float | str | None]]:
    rows: list[dict[str, float | str | None]] = []
    for case in case_dirs:
        if (case / "formal_sample_complete.json").is_file():
            continue
        log = case / "log.foamMultiRun.formal"
        if not log.is_file():
            continue
        log_time = dt.datetime.fromtimestamp(log.stat().st_mtime, tz=now.tzinfo)
        age_seconds = max(0.0, (now - log_time).total_seconds())
        if age_seconds > active_log_max_age_seconds:
            continue
        row = resumed_segment(log)
        restart = row["restart_iteration"]
        restart_wall_text = row["restart_wall_time"]
        current_from_log = row["current_iteration"]
        current_from_fields = latest_new_complete_parallel_iteration(
            case,
            restart_iteration=float(restart) if restart is not None else None,
            restart_wall_time=(
                str(restart_wall_text) if restart_wall_text is not None else None
            ),
            parallel_ranks=parallel_ranks,
        )
        current_from_observations = latest_new_observed_iteration(
            case,
            restart_iteration=float(restart) if restart is not None else None,
            restart_wall_time=(
                str(restart_wall_text) if restart_wall_text is not None else None
            ),
        )
        available_times = [
            float(value)
            for value in (
                restart,
                current_from_log,
                current_from_fields,
                current_from_observations,
            )
            if value is not None
        ]
        current = max(available_times) if available_times else None
        rate = None
        remaining_hours = None
        metadata_path = case / "cht_smoke_metadata.json"
        end_iteration = None
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            end_iteration = float(
                metadata.get("steady_iteration_end", metadata["end_time"])
            )
        if restart is not None and current is not None and restart_wall_text is not None:
            restart_wall = dt.datetime.fromisoformat(str(restart_wall_text))
            wall_now = now.astimezone(restart_wall.tzinfo) if restart_wall.tzinfo else now.replace(tzinfo=None)
            elapsed_hours = (wall_now - restart_wall).total_seconds() / 3600.0
            if elapsed_hours > 0 and float(current) >= float(restart):
                rate = (float(current) - float(restart)) / elapsed_hours
                if rate > 0 and end_iteration is not None:
                    remaining_hours = max(0.0, end_iteration - float(current)) / rate
        rows.append(
            {
                "condition_id": case.name,
                "solver_time_semantics": "steady_iteration_index",
                "physical_time_s": None,
                "restart_iteration": restart,
                "current_iteration": current,
                "latest_new_observed_iteration": current_from_observations,
                "latest_new_complete_parallel_iteration": current_from_fields,
                "target_end_iteration": end_iteration,
                "log_age_seconds": age_seconds,
                "steady_iterations_per_wall_hour": rate,
                "estimated_hours_to_case_end": remaining_hours,
                "phase": (
                    "finalizing_three_dimensional_output"
                    if end_iteration is not None
                    and current is not None
                    and float(current) >= end_iteration
                    else "solving"
                ),
            }
        )
    return rows


def progress(
    matrix_root: Path,
    concurrent_cases: int,
    *,
    now: dt.datetime | None = None,
    active_log_max_age_seconds: float = 600.0,
    parallel_ranks: int = 32,
) -> dict[str, object]:
    case_dirs = sorted(matrix_root.glob("u*_T*_q*"))
    completed = [case for case in case_dirs if (case / "formal_sample_complete.json").is_file()]
    runtimes = [
        value
        for case in completed
        if (
            value := accumulated_solver_clock_time(case / "log.foamMultiRun.formal")
        )
        is not None
    ]
    postprocess_times = [
        value
        for case in completed
        if (value := completed_case_postprocess_seconds(case)) is not None
    ]
    remaining = len(case_dirs) - len(completed)
    mean_seconds = statistics.mean(runtimes) if runtimes else None
    median_seconds = statistics.median(runtimes) if runtimes else None
    mean_postprocess_seconds = (
        statistics.mean(postprocess_times) if postprocess_times else 0.0
    )
    mean_total_seconds = (
        mean_seconds + mean_postprocess_seconds if mean_seconds is not None else None
    )
    estimated_hours = (
        remaining * mean_total_seconds / concurrent_cases / 3600.0
        if mean_total_seconds is not None and concurrent_cases > 0
        else None
    )
    if now is None:
        now = dt.datetime.now().astimezone()
    active_rows = active_case_rows(
        case_dirs,
        now=now,
        active_log_max_age_seconds=active_log_max_age_seconds,
        parallel_ranks=parallel_ranks,
    )
    return {
        "status": "P418 matrix runtime progress",
        "total_cases": len(case_dirs),
        "completed_cases": len(completed),
        "remaining_cases": remaining,
        "completed_fraction": len(completed) / len(case_dirs) if case_dirs else 0.0,
        "runtime_sample_count": len(runtimes),
        "mean_completed_case_accumulated_clock_seconds": mean_seconds,
        "median_completed_case_accumulated_clock_seconds": median_seconds,
        "completed_case_postprocess_sample_count": len(postprocess_times),
        "mean_completed_case_postprocess_seconds": mean_postprocess_seconds,
        "mean_completed_case_total_seconds": mean_total_seconds,
        "concurrent_cases": concurrent_cases,
        "active_case_count": len(active_rows),
        "active_cases": active_rows,
        "simple_remaining_walltime_estimate_hours": estimated_hours,
        "estimate_note": (
            "The matrix estimate uses completed-case mean solver clock time plus the mean "
            "finalization duration recorded explicitly in each new completion marker. "
            "Copied pilot artifacts without an explicit duration are excluded from that mean. "
            "Active-case rates use only the final restart segment. Numeric OpenFOAM Time "
            "labels are steady nonlinear-iteration indices because both regions use the "
            "steadyState ddt scheme; they are not physical seconds."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--concurrent-cases", type=int, default=3)
    parser.add_argument("--active-log-max-age-seconds", type=float, default=600.0)
    parser.add_argument("--parallel-ranks", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = progress(
        args.matrix_root.resolve(),
        args.concurrent_cases,
        active_log_max_age_seconds=args.active_log_max_age_seconds,
        parallel_ranks=args.parallel_ranks,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(args.output)
    print(text, end="")


if __name__ == "__main__":
    main()
