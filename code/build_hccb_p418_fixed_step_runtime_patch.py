#!/usr/bin/env python3
"""Build the small runtime patch for repaired P418 fixed-flow step calculations."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "code/add_hccb_transient_temperature_outputs.py",
    "code/build_hccb_p418_step_response_cases.py",
    "code/import_hccb_p418_parallel_history.py",
    "code/run_hccb_p418_step_responses.sh",
    "scripts/run_hccb_p418_formal_fixed_step_sequence.sh",
    "scripts/run_hccb_p418_high_re_fixed_step_sequence.sh",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(output_root: Path, overwrite: bool) -> dict:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    rows = []
    for relative in FILES:
        source = ROOT / relative
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if destination.suffix in {".sh", ".sbatch"} or destination.name.endswith(".py"):
            destination.chmod(0o755)
        rows.append(
            {
                "path": relative,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    manifest = {
        "status": "p418_fixed_step_runtime_patch_ready_not_submitted",
        "purpose": (
            "Use absolute OpenFOAM time indices for staged full-field writes, "
            "preserve wall heat-flow integration, and optionally import a verified "
            "0-1 s parallel history into a fresh formal work root."
        ),
        "scientific_model_changed": False,
        "physical_parameters_changed": False,
        "required_preflight": [
            "verify all patch file SHA256 values",
            "verify 32 MPI partitions and common OpenFOAM time index",
            "verify wallHeatFlux only follows requested full-field write times",
            "run one zero-solver check and one short representative recovery",
        ],
        "files": rows,
    }
    manifest_path = output_root / "PATCH_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksums = [
        f"{row['sha256']}  {row['path']}" for row in rows
    ] + [f"{sha256(manifest_path)}  PATCH_MANIFEST.json"]
    (output_root / "PATCH_SHA256SUMS").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )
    ready = {
        "status": "ready_for_cloud_zero_solver_preflight",
        "patch_manifest_sha256": sha256(manifest_path),
        "patch_checksums_sha256": sha256(output_root / "PATCH_SHA256SUMS"),
        "formal_solver_submission_authorized_by_this_file": False,
    }
    (output_root / "PATCH_READY.json").write_text(
        json.dumps(ready, indent=2) + "\n", encoding="utf-8"
    )
    return ready


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.output_root.resolve(), args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
