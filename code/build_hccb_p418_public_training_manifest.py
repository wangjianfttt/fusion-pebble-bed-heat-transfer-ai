#!/usr/bin/env python3
"""Create a machine-independent copy of the formal P418 training manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PRIVATE_TOKENS = (
    "/" + "data2/",
    "/" + "Users/",
    "/" + "n96pfs/",
    "192" + ".168.",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_roots(value: Any, source_roots: tuple[str, ...], public_root: str) -> Any:
    if isinstance(value, str):
        for source_root in source_roots:
            value = value.replace(source_root, public_root)
        return value
    if isinstance(value, list):
        return [replace_roots(item, source_roots, public_root) for item in value]
    if isinstance(value, dict):
        return {
            key: replace_roots(item, source_roots, public_root)
            for key, item in value.items()
        }
    return value


def refresh_completion_state(
    payload: dict[str, Any], source_roots: tuple[str, ...], local_root: Path
) -> None:
    """Refresh the public progress snapshot from locally available result files."""
    registered = {
        job.get("job_id") for job in payload.get("jobs", []) if job.get("job_id")
    }
    completed_set = {
        job_id
        for job_id in payload.get("completed_job_ids", [])
        if job_id in registered
    }
    for job in payload.get("jobs", []):
        completion_file = job.get("completion_file")
        if not isinstance(completion_file, str) or not completion_file:
            continue
        local_completion = completion_file
        for source_root in source_roots:
            if local_completion.startswith(source_root + "/"):
                local_completion = str(local_root) + local_completion[len(source_root) :]
                break
        if Path(local_completion).is_file():
            completed_set.add(job["job_id"])

    completed = [
        job["job_id"]
        for job in payload.get("jobs", [])
        if job.get("job_id") in completed_set
    ]
    payload["completed_job_ids"] = completed
    payload["completed_job_count"] = len(completed)
    payload["remaining_job_count"] = payload.get("job_count", 0) - len(completed)
    payload["completion_state_basis"] = (
        "Recomputed from completion files available under ${PROJECT_ROOT} "
        "when this public manifest was built."
    )


def build(input_path: Path, output_path: Path, source_root: str) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    source_roots = {source_root.rstrip("/")}
    runner_suffix = "/code/run_hccb_p418_step_responses.sh"
    runner = payload.get("source_runner", "")
    if isinstance(runner, str) and runner.endswith(runner_suffix):
        source_roots.add(runner[: -len(runner_suffix)])
    ordered_roots = tuple(sorted(source_roots, key=len, reverse=True))
    refresh_completion_state(payload, ordered_roots, Path(source_root).resolve())
    public = replace_roots(payload, ordered_roots, "${PROJECT_ROOT}")
    public["path_policy"] = (
        "Machine paths are replaced by ${PROJECT_ROOT}; commands and numerical "
        "settings are otherwise unchanged."
    )
    public["source_manifest_sha256"] = sha256(input_path)
    text = json.dumps(public, ensure_ascii=False, indent=2) + "\n"
    leaked = [token for token in PRIVATE_TOKENS if token in text]
    if leaked:
        raise ValueError(f"private machine path remains in public manifest: {leaked}")
    if public.get("job_count") != len(public.get("jobs", [])):
        raise ValueError("job_count does not match the number of jobs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return public


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args()
    result = build(args.input.resolve(), args.output.resolve(), args.source_root)
    print(
        json.dumps(
            {
                "status": "p418_public_training_manifest_ready",
                "job_count": result["job_count"],
                "completed_job_count": result.get("completed_job_count"),
                "remaining_job_count": result.get("remaining_job_count"),
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
