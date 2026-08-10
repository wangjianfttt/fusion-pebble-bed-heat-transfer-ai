#!/usr/bin/env python3
"""Check formal P418 training commands and their declared input producers."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


INPUT_PATH_FLAGS = {
    "--cost-summary",
    "--csv",
    "--data",
    "--dataset-index",
    "--metrics",
    "--metrics-csv",
    "--model-summary",
    "--prediction-dir",
    "--residual-geometry",
    "--result-dir",
    "--result-root",
    "--seed-robustness-summary",
    "--speed-csv",
    "--splits",
    "--step-root",
    "--summary",
}


def load_jobs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("formal training manifest contains no jobs")
    return jobs


def is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def check_manifest(path: Path) -> dict[str, Any]:
    jobs = load_jobs(path)
    errors: list[str] = []
    identifiers = [str(job.get("job_id", "")) for job in jobs]
    if any(not identifier for identifier in identifiers):
        errors.append("one or more jobs have no job_id")
    if len(identifiers) != len(set(identifiers)):
        errors.append("job ids are not unique")

    completion_files = [str(job.get("completion_file", "")) for job in jobs]
    if any(not value for value in completion_files):
        errors.append("one or more jobs have no completion_file")
    if len(completion_files) != len(set(completion_files)):
        errors.append("completion files are not unique")

    ancestors: dict[str, set[str]] = {}
    producers: list[tuple[str, Path]] = []
    scripts_present = 0
    existing_input_count = 0
    deferred_inputs: list[dict[str, str]] = []

    for job in jobs:
        job_id = str(job.get("job_id", ""))
        dependencies = {str(value) for value in job.get("depends_on", [])}
        missing_dependencies = sorted(
            dependency for dependency in dependencies if dependency not in ancestors
        )
        if missing_dependencies:
            errors.append(
                f"{job_id}: dependencies appear after the job: "
                + ", ".join(missing_dependencies)
            )
        closure = set(dependencies)
        for dependency in dependencies:
            closure.update(ancestors.get(dependency, set()))
        ancestors[job_id] = closure

        tokens = shlex.split(str(job.get("command", "")))
        if len(tokens) < 2 or tokens[0] not in {"python3", "bash"}:
            errors.append(f"{job_id}: unsupported or empty command")
        elif not Path(tokens[1]).is_file():
            errors.append(f"{job_id}: command script is missing: {tokens[1]}")
        else:
            scripts_present += 1

        output_dir = Path(str(job.get("output_dir", "")))
        completion_file = Path(str(job.get("completion_file", "")))
        if (
            job.get("stage") != "paper_results"
            and output_dir
            and completion_file
            and not is_within(completion_file, output_dir)
        ):
            errors.append(f"{job_id}: completion file lies outside output_dir")

        index = 2
        while index < len(tokens):
            flag = tokens[index]
            if flag not in INPUT_PATH_FLAGS or index + 1 >= len(tokens):
                index += 1
                continue
            value = Path(tokens[index + 1])
            index += 2
            if not value.is_absolute():
                continue
            if value.exists():
                existing_input_count += 1
                continue
            candidates = [
                producer_id
                for producer_id, producer_dir in producers
                if is_within(value, producer_dir)
            ]
            declared = [
                producer_id for producer_id in candidates if producer_id in closure
            ]
            if declared:
                deferred_inputs.append(
                    {
                        "job_id": job_id,
                        "flag": flag,
                        "path": str(value),
                        "declared_producer": declared[-1],
                    }
                )
            elif candidates:
                errors.append(
                    f"{job_id}: input is produced earlier but its producer is not a "
                    f"declared dependency: {flag} {value}"
                )
            else:
                errors.append(
                    f"{job_id}: input is missing and has no earlier producer: "
                    f"{flag} {value}"
                )

        producers.append((job_id, output_dir))

    return {
        "status": (
            "p418_formal_training_manifest_static_readiness_passed"
            if not errors
            else "p418_formal_training_manifest_static_readiness_failed"
        ),
        "manifest": str(path.resolve()),
        "job_count": len(jobs),
        "scripts_present": scripts_present,
        "existing_input_path_count": existing_input_count,
        "deferred_input_path_count": len(deferred_inputs),
        "deferred_inputs": deferred_inputs,
        "errors": errors,
        "new_physical_parameters": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check_manifest(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
