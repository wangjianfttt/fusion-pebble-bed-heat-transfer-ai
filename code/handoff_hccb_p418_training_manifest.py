#!/usr/bin/env python3
"""Hand a running P418 model chain to a revised manifest after its child exits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_state(process_id: int) -> str | None:
    stat_path = Path(f"/proc/{process_id}/stat")
    if not stat_path.is_file():
        return None
    fields = stat_path.read_text(encoding="utf-8").split()
    return fields[2] if len(fields) > 2 else None


def parent_process_id(process_id: int) -> int | None:
    status_path = Path(f"/proc/{process_id}/status")
    if not status_path.is_file():
        return None
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("PPid:"):
            return int(line.split()[1])
    return None


def process_holds_execution_resources(state: str | None) -> bool:
    """Return whether a process state can still hold locks or execute code."""
    return state not in {None, "Z"}


def valid_completion(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    status = str(payload.get("status", "")).lower()
    return not any(
        token in status
        for token in (
            "failed",
            "failure",
            "incomplete",
            "not_started",
            "in_progress",
            "running",
            "blocked",
        )
    )


def validate_revised_manifest(path: Path, expected_sha256: str) -> dict[str, object]:
    actual_sha256 = sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"revised manifest SHA mismatch: {actual_sha256} != {expected_sha256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("revised manifest contains no jobs")
    physics_jobs = [
        job
        for job in jobs
        if "--physics-mode energy_and_flux" in str(job.get("command", ""))
    ]
    if not physics_jobs:
        raise ValueError("revised manifest contains no energy-and-flux jobs")
    missing_cuda = [
        str(job.get("job_id"))
        for job in physics_jobs
        if "--physics-device cuda" not in str(job.get("command", ""))
    ]
    if missing_cuda:
        raise ValueError(
            "physics jobs without explicit CUDA residuals: " + ", ".join(missing_cuda)
        )
    return payload


def write_record(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-executor-pid", type=int, required=True)
    parser.add_argument("--current-child-pid", type=int, required=True)
    parser.add_argument("--current-completion-file", type=Path, required=True)
    parser.add_argument("--new-manifest", type=Path, required=True)
    parser.add_argument("--new-manifest-sha256", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--required-ready-file", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--cpu-list", required=True)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    manifest = args.new_manifest.resolve()
    payload = validate_revised_manifest(manifest, args.new_manifest_sha256)
    if args.poll_seconds <= 0:
        raise ValueError("poll interval must be positive")
    if parent_process_id(args.current_child_pid) != args.old_executor_pid:
        raise RuntimeError("declared training process is not a child of the old executor")

    record = {
        "status": "p418_training_manifest_handoff_preflight_passed",
        "old_executor_pid": args.old_executor_pid,
        "current_child_pid": args.current_child_pid,
        "current_completion_file": str(args.current_completion_file.resolve()),
        "new_manifest": str(manifest),
        "new_manifest_sha256": sha256(manifest),
        "registered_jobs": len(payload["jobs"]),
        "physics_jobs_with_explicit_cuda": sum(
            "--physics-mode energy_and_flux" in str(job.get("command", ""))
            for job in payload["jobs"]
        ),
        "new_physical_parameters": [],
        "updated_unix_s": time.time(),
    }
    write_record(args.record.resolve(), record)
    if args.preflight:
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0

    os.kill(args.old_executor_pid, signal.SIGSTOP)
    record["status"] = "p418_training_manifest_handoff_waiting_for_current_model"
    record["old_executor_state"] = process_state(args.old_executor_pid)
    write_record(args.record.resolve(), record)

    completion = args.current_completion_file.resolve()
    while True:
        child_state = process_state(args.current_child_pid)
        if valid_completion(completion) and child_state in {None, "Z"}:
            break
        if child_state is None and not valid_completion(completion):
            os.kill(args.old_executor_pid, signal.SIGCONT)
            raise RuntimeError("current training exited without a valid completion file")
        time.sleep(args.poll_seconds)

    os.kill(args.old_executor_pid, signal.SIGKILL)
    for _ in range(100):
        if not process_holds_execution_resources(process_state(args.old_executor_pid)):
            break
        time.sleep(0.1)
    if process_holds_execution_resources(process_state(args.old_executor_pid)):
        raise RuntimeError("old executor did not exit after current training completed")

    command = [
        sys.executable,
        str(root / "code/execute_hccb_p418_formal_training_manifest.py"),
        "--manifest",
        str(manifest),
        "--root",
        str(root),
        "--state-file",
        str(args.state_file.resolve()),
        "--log-dir",
        str(args.log_dir.resolve()),
        "--lock-file",
        str(args.lock_file.resolve()),
        "--wait-interval-s",
        "60",
        "--cuda-visible-devices",
        args.cuda_visible_devices,
        "--cpu-list",
        args.cpu_list,
        "--required-ready-file",
        str(args.required_ready_file.resolve()),
        "--execute",
    ]
    args.stdout.parent.mkdir(parents=True, exist_ok=True)
    args.stderr.parent.mkdir(parents=True, exist_ok=True)
    with args.stdout.open("ab") as stdout, args.stderr.open("ab") as stderr:
        child = subprocess.Popen(
            command,
            cwd=root,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    record.update(
        {
            "status": "p418_training_manifest_handoff_complete",
            "new_executor_pid": child.pid,
            "new_executor_command": command,
            "updated_unix_s": time.time(),
        }
    )
    write_record(args.record.resolve(), record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
