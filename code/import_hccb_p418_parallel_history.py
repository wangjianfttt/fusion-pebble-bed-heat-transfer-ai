#!/usr/bin/env python3
"""Import a verified early-time parallel history into a fresh P418 step case."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path


CORE_FIELDS = (
    "fluid/T",
    "fluid/U",
    "fluid/p",
    "fluid/p_rgh",
    "fluid/phi",
    "solid/T",
    "uniform/time",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def numeric_time_directories(root: Path) -> dict[float, str]:
    rows: dict[float, str] = {}
    for path in root.iterdir():
        if not path.is_dir():
            continue
        try:
            value = float(path.name)
        except ValueError:
            continue
        rows[value] = path.name
    return rows


def matching_time_name(available: dict[float, str], requested: float) -> str:
    matches = [
        name
        for value, name in available.items()
        if math.isclose(value, requested, rel_tol=0.0, abs_tol=1.0e-10)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one parallel directory for t={requested:g} s, found {matches}"
        )
    return matches[0]


def truncate_history_file(source: Path, destination: Path, through_time: float) -> None:
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            kept.append(line)
            continue
        try:
            time_s = float(stripped.split()[0])
        except (ValueError, IndexError):
            kept.append(line)
            continue
        if time_s <= through_time + 1.0e-10:
            kept.append(line)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(kept) + "\n", encoding="utf-8")


def copy_postprocessing(source: Path, destination: Path, through_time: float) -> int:
    if not source.is_dir():
        return 0
    copied = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            truncate_history_file(path, target, through_time)
            copied += 1
    return copied


def import_history(
    source_case: Path,
    destination_case: Path,
    through_time: float,
    mpi_tasks: int,
) -> dict:
    source_metadata_path = source_case / "step_case_metadata.json"
    destination_metadata_path = destination_case / "step_case_metadata.json"
    source_metadata = read_json(source_metadata_path)
    destination_metadata = read_json(destination_metadata_path)
    for key in ("sequence_id", "source_condition_id", "target_condition_id"):
        if source_metadata.get(key) != destination_metadata.get(key):
            raise ValueError(
                f"parallel-history metadata mismatch for {key}: "
                f"{source_metadata.get(key)!r} != {destination_metadata.get(key)!r}"
            )

    requested_times = [
        float(value)
        for value in destination_metadata["snapshot_times_s"]
        if 0.0 < float(value) <= through_time + 1.0e-10
    ]
    if not requested_times or not math.isclose(
        requested_times[-1], through_time, rel_tol=0.0, abs_tol=1.0e-10
    ):
        raise ValueError(
            f"resume time {through_time:g} s is not a requested full-field snapshot"
        )

    available = numeric_time_directories(source_case / "processor0")
    source_names = {
        time_s: matching_time_name(available, time_s) for time_s in requested_times
    }
    copied_files = 0
    for time_s in requested_times:
        source_name = source_names[time_s]
        destination_name = f"{time_s:g}"
        for rank in range(mpi_tasks):
            source_time = source_case / f"processor{rank}" / source_name
            destination_time = destination_case / f"processor{rank}" / destination_name
            missing = [
                field for field in CORE_FIELDS if not (source_time / field).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"incomplete source state at rank {rank}, t={source_name}: {missing}"
                )
            if destination_time.exists():
                shutil.rmtree(destination_time)
            shutil.copytree(source_time, destination_time, symlinks=True)
            copied_files += sum(path.is_file() for path in destination_time.rglob("*"))

    postprocessing_files = copy_postprocessing(
        source_case / "postProcessing",
        destination_case / "postProcessing",
        through_time,
    )
    record = {
        "status": "imported_verified_parallel_history",
        "sequence_id": destination_metadata["sequence_id"],
        "source_case": str(source_case),
        "destination_case": str(destination_case),
        "through_time_s": through_time,
        "mpi_tasks": mpi_tasks,
        "imported_snapshot_times_s": requested_times,
        "parallel_files_copied": copied_files,
        "postprocessing_files_copied": postprocessing_files,
        "source_metadata_sha256": sha256(source_metadata_path),
        "destination_metadata_sha256": sha256(destination_metadata_path),
    }
    record_path = destination_case / "parallel_history_import_complete.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    record["record_sha256"] = sha256(record_path)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-case", type=Path, required=True)
    parser.add_argument("--destination-case", type=Path, required=True)
    parser.add_argument("--through-time", type=float, required=True)
    parser.add_argument("--mpi-tasks", type=int, default=32)
    args = parser.parse_args()
    record = import_history(
        args.source_case.resolve(),
        args.destination_case.resolve(),
        args.through_time,
        args.mpi_tasks,
    )
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
