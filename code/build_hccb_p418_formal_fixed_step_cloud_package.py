#!/usr/bin/env python3
"""Build the self-contained cloud package for the 12 formal P418 thermal steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "parameters/hccb_p418_transient_step_plan.json"
PACKAGE_NAME = "p418_formal_fixed_steps_300s"

PACKAGE_SCRIPTS = (
    "run_hccb_p418_formal_fixed_step_sequence.sh",
    "run_hccb_p418_formal_fixed_step_array_n96p.sbatch",
    "collect_hccb_p418_formal_fixed_step_n96p.sbatch",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def link_or_copy(source: Path | str, destination: Path | str) -> None:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def link_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(
        source,
        destination,
        symlinks=True,
        copy_function=link_or_copy,
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_endpoint_ready(case: Path) -> tuple[dict, str]:
    marker_path = case / "formal_sample_complete.json"
    metadata_path = case / "cht_smoke_metadata.json"
    marker = read_json(marker_path)
    if marker.get("solver_finished") is not True:
        raise ValueError(f"unfinished steady endpoint: {case.name}")
    iteration = str(marker.get("time", marker.get("reported_iteration", "")))
    if int(float(iteration)) != 200:
        raise ValueError(f"endpoint is not steady iteration 200: {case.name}")
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    return marker, iteration


def copy_endpoint(source: Path, destination: Path) -> dict:
    marker, iteration = ensure_endpoint_ready(source)
    destination.mkdir(parents=True)
    for filename in ("formal_sample_complete.json", "cht_smoke_metadata.json"):
        link_or_copy(source / filename, destination / filename)

    for dirname in ("0", "system", "constant"):
        link_tree(source / dirname, destination / dirname)

    final_fields = (
        "fluid/T",
        "solid/T",
        "fluid/U",
        "fluid/p",
        "fluid/p_rgh",
        "fluid/phi",
    )
    copied_fields = []
    for relative in final_fields:
        field = source / iteration / relative
        if not field.is_file():
            raise FileNotFoundError(field)
        link_or_copy(field, destination / iteration / relative)
        copied_fields.append(f"{iteration}/{relative}")

    return {
        "condition_id": source.name,
        "steady_iteration": int(float(iteration)),
        "solver_finished": True,
        "relative_mass_difference": marker["relative_mass_difference"],
        "relative_energy_difference": marker["relative_energy_difference"],
        "copied_final_fields": copied_fields,
        "formal_marker_sha256": sha256(source / "formal_sample_complete.json"),
        "metadata_sha256": sha256(source / "cht_smoke_metadata.json"),
    }


def validate_links(root: Path) -> list[dict[str, str]]:
    links = []
    for path in root.rglob("*"):
        if not path.is_symlink():
            continue
        target = os.readlink(path)
        if os.path.isabs(target):
            raise ValueError(f"absolute symbolic link is not allowed: {path} -> {target}")
        resolved = (path.parent / target).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"symbolic link escapes package: {path} -> {target}") from exc
        links.append({"path": str(path.relative_to(root)), "target": target})
    return links


def write_checksums(root: Path) -> tuple[int, int]:
    checksum_path = root / "PACKAGE_SHA256SUMS"
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path != checksum_path
    )
    total_bytes = sum(path.stat().st_size for path in files)
    with checksum_path.open("w", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256(path)}  {path.relative_to(root)}\n")
    return len(files), total_bytes


def build_package(
    project_root: Path,
    matrix_root: Path,
    runtime_template: Path,
    plan_path: Path,
    output_parent: Path,
    overwrite: bool,
) -> dict:
    package_root = output_parent / PACKAGE_NAME
    if package_root.exists():
        if not overwrite:
            raise FileExistsError(package_root)
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True)

    plan = read_json(plan_path)
    sequences = plan["sequences"]
    if len(sequences) != 12:
        raise ValueError(f"expected 12 formal sequences, found {len(sequences)}")
    endpoints = sorted(
        {
            str(sequence[key])
            for sequence in sequences
            for key in ("source_condition_id", "target_condition_id")
        }
    )
    if len(endpoints) != 11:
        raise ValueError(f"expected 11 unique endpoints, found {len(endpoints)}")

    link_tree(project_root / "code", package_root / "code")
    link_tree(project_root / "parameters", package_root / "parameters")
    link_tree(runtime_template / "python_vendor", package_root / "python_vendor")
    if (runtime_template / "python_wheels").is_dir():
        link_tree(runtime_template / "python_wheels", package_root / "python_wheels")

    scripts_dir = package_root / "scripts"
    scripts_dir.mkdir()
    for filename in PACKAGE_SCRIPTS:
        source = project_root / "scripts" / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        link_or_copy(source, scripts_dir / filename)
        (scripts_dir / filename).chmod(0o755)

    matrix_destination = package_root / "matrix"
    matrix_destination.mkdir()
    link_or_copy(
        matrix_root / "matrix_manifest.json",
        matrix_destination / "matrix_manifest.json",
    )
    endpoint_rows = []
    for endpoint in endpoints:
        endpoint_rows.append(
            copy_endpoint(
                matrix_root / endpoint,
                matrix_destination / endpoint,
            )
        )

    plan_destination = package_root / "parameters/hccb_p418_transient_step_plan.json"
    if (
        not plan_destination.is_file()
        or sha256(plan_path) != sha256(plan_destination)
    ):
        plan_destination.unlink(missing_ok=True)
        shutil.copy2(plan_path, plan_destination)

    endpoint_roles: dict[str, list[str]] = defaultdict(list)
    for sequence in sequences:
        endpoint_roles[str(sequence["source_condition_id"])].append(
            f"source:{sequence['sequence_id']}"
        )
        endpoint_roles[str(sequence["target_condition_id"])].append(
            f"target:{sequence['sequence_id']}"
        )

    links = validate_links(package_root)
    manifest = {
        "status": "p418_formal_fixed_step_cloud_package_built_not_submitted",
        "package_root": PACKAGE_NAME,
        "physical_response_duration_s": float(
            plan["numerical_time_design"]["duration_s"]
        ),
        "sequence_count": len(sequences),
        "unique_endpoint_count": len(endpoints),
        "sequence_order": [
            {
                "array_index": index,
                "sequence_id": row["sequence_id"],
                "family": row["family"],
                "source_condition_id": row["source_condition_id"],
                "target_condition_id": row["target_condition_id"],
            }
            for index, row in enumerate(sequences)
        ],
        "endpoint_rows": endpoint_rows,
        "endpoint_roles": dict(endpoint_roles),
        "resource_per_sequence": {
            "mpi_tasks": 32,
            "memory_GiB": 64,
            "walltime": "12:00:00",
        },
        "array_concurrency_limit": 6,
        "estimated_cleaned_output_bytes_per_sequence": 10_553_000_000,
        "estimated_cleaned_output_bytes_all_sequences": 126_636_000_000,
        "time_step_schedule": plan["numerical_time_design"]["time_step_schedule"],
        "field_write_schedule": plan["numerical_time_design"][
            "field_write_schedule"
        ],
        "new_physical_parameters": [],
        "formal_solver_submitted": False,
        "symbolic_links": links,
        "input_sha256": {
            "formal_plan": sha256(plan_destination),
            "matrix_manifest": sha256(matrix_destination / "matrix_manifest.json"),
            "runtime_numpy": sha256(
                next((package_root / "python_vendor/numpy").rglob("__init__.py"))
            ),
        },
    }
    manifest_path = package_root / "PACKAGE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    file_count, total_bytes = write_checksums(package_root)
    ready = {
        "status": "p418_formal_fixed_step_package_ready_not_submitted",
        "package_root": str(package_root),
        "regular_file_count": file_count,
        "regular_file_bytes": total_bytes,
        "checksum_manifest_sha256": sha256(package_root / "PACKAGE_SHA256SUMS"),
        "package_manifest_sha256": sha256(manifest_path),
        "sequence_count": 12,
        "unique_endpoint_count": 11,
        "duration_s": 300.0,
        "formal_solver_submitted": False,
    }
    (package_root / "PACKAGE_READY.json").write_text(
        json.dumps(ready, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # PACKAGE_READY is intentionally outside PACKAGE_SHA256SUMS because it records
    # that checksum file's digest.
    return ready


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--runtime-template", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    payload = build_package(
        args.project_root.resolve(),
        args.matrix_root.resolve(),
        args.runtime_template.resolve(),
        args.plan.resolve(),
        args.output_parent.resolve(),
        args.overwrite,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
