#!/usr/bin/env python3
"""Run the missing P418 formal model jobs in declared dependency order."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import time
from pathlib import Path


def load_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("formal training manifest contains no jobs")
    identifiers = [str(job["job_id"]) for job in jobs]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("formal training manifest contains duplicate job ids")
    known: set[str] = set()
    for job in jobs:
        dependencies = [str(value) for value in job.get("depends_on", [])]
        missing = sorted(set(dependencies) - known)
        if missing:
            raise ValueError(
                f"{job['job_id']} appears before its dependencies: {missing}"
            )
        known.add(str(job["job_id"]))
    return payload


def valid_completion(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if path.suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        status = str(payload.get("status", "")).lower()
        incomplete_tokens = (
            "failed",
            "failure",
            "incomplete",
            "not_started",
            "in_progress",
            "running",
            "blocked",
        )
        if any(token in status for token in incomplete_tokens):
            return False
    return True


def wait_for_processes(process_ids: list[int], interval_s: int) -> None:
    remaining = set(process_ids)
    while remaining:
        finished = {
            process_id
            for process_id in remaining
            if not Path(f"/proc/{process_id}").exists()
        }
        remaining -= finished
        if remaining:
            print(
                "waiting for existing P418 processes:",
                ", ".join(str(value) for value in sorted(remaining)),
                flush=True,
            )
            time.sleep(interval_s)


def write_state(
    path: Path,
    *,
    status: str,
    manifest_path: Path,
    completed: list[str],
    skipped: list[str],
    current_job: str | None,
) -> None:
    payload = {
        "status": status,
        "manifest": str(manifest_path.resolve()),
        "completed_job_ids_this_run": completed,
        "existing_job_ids_retained": skipped,
        "current_job_id": current_job,
        "updated_unix_s": time.time(),
        "new_physical_parameters": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int, action="append", default=[])
    parser.add_argument("--required-ready-file", type=Path, action="append", default=[])
    parser.add_argument("--wait-interval-s", type=int, default=60)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--cpu-list")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    manifest_path = args.manifest.resolve()
    state_path = args.state_file.resolve()
    log_dir = args.log_dir.resolve()
    lock_path = args.lock_file.resolve()
    if args.wait_interval_s <= 0:
        raise ValueError("wait interval must be positive")

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SystemExit("another P418 formal model chain is active") from error

    wait_for_processes(args.wait_pid, args.wait_interval_s)
    missing_ready = [
        str(path.resolve())
        for path in args.required_ready_file
        if not valid_completion(path.resolve())
    ]
    if missing_ready:
        raise FileNotFoundError(
            "required preceding results are missing or unreadable: "
            + ", ".join(missing_ready)
        )

    payload = load_manifest(manifest_path)
    jobs = payload["jobs"]
    assert isinstance(jobs, list)
    completed_by_id = {
        str(job["job_id"]): valid_completion(
            Path(str(job["completion_file"])).resolve()
        )
        for job in jobs
    }
    missing_jobs = [
        str(job["job_id"]) for job in jobs if not completed_by_id[str(job["job_id"])]
    ]
    print(
        json.dumps(
            {
                "registered_jobs": len(jobs),
                "already_complete": len(jobs) - len(missing_jobs),
                "remaining": len(missing_jobs),
                "remaining_job_ids": missing_jobs,
                "execute": args.execute,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if not args.execute:
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    skipped: list[str] = []
    for job in jobs:
        job_id = str(job["job_id"])
        completion_path = Path(str(job["completion_file"])).resolve()
        if valid_completion(completion_path):
            completed_by_id[job_id] = True
            skipped.append(job_id)
            continue
        dependencies = [str(value) for value in job.get("depends_on", [])]
        unavailable = [
            dependency
            for dependency in dependencies
            if not completed_by_id.get(dependency, False)
        ]
        if unavailable:
            raise RuntimeError(
                f"{job_id} cannot start because dependencies are incomplete: "
                + ", ".join(unavailable)
            )
        command = str(job["command"])
        if args.cpu_list:
            command = f"taskset -c {shlex.quote(args.cpu_list)} {command}"
        environment = os.environ.copy()
        if str(job.get("device")) == "cuda":
            environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        stdout_path = log_dir / f"{job_id}.stdout.log"
        stderr_path = log_dir / f"{job_id}.stderr.log"
        write_state(
            state_path,
            status="p418_formal_model_chain_running",
            manifest_path=manifest_path,
            completed=completed,
            skipped=skipped,
            current_job=job_id,
        )
        print(f"starting {job_id}", flush=True)
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            result = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                cwd=root,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"{job_id} exited with code {result.returncode}; "
                f"see {stderr_path}"
            )
        if not valid_completion(completion_path):
            raise RuntimeError(
                f"{job_id} returned successfully but did not produce {completion_path}"
            )
        completed_by_id[job_id] = True
        completed.append(job_id)

    write_state(
        state_path,
        status="completed_p418_formal_model_chain",
        manifest_path=manifest_path,
        completed=completed,
        skipped=skipped,
        current_job=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
