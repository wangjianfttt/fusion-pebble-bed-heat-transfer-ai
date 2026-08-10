#!/usr/bin/env python3
"""Assemble the no-solver/direct-transport matched-initial cloud package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "cloud_build"
    / "p418_matched_initial_direct_transport_smoke_20260809"
)

PACKAGE_FILES = (
    "compile_prepare_direct_no_solver.sbatch",
    "run_direct_matched_initial_smoke.sbatch",
    "submit_direct_matched_initial_smoke.sh",
    "smoke_plan.json",
    "README_CN.md",
)

MODULE_FILES = (
    "hccbHeliumTransport.H",
    "hccbHeliumTransportI.H",
    "hccbHeliumTransport.C",
    "hccbHeliumThermos.C",
    "physicalProperties.example",
    "Make/files",
    "Make/options",
    "check/hccbHeliumTransportCheck.C",
    "check/Make/files",
    "check/Make/options",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output.resolve()
    module_source = ROOT / "solver_extensions" / "hccbHeliumTransport"
    support_sources = (
        ROOT / "code" / "validate_hccb_p418_helium_transport_check.py",
        ROOT / "code" / "export_hccb_p418_transient_observables.py",
        ROOT / "code" / "export_hccb_p418_matched_initial_short_observables.py",
    )

    missing = [name for name in PACKAGE_FILES if not (output / name).is_file()]
    missing += [name for name in MODULE_FILES if not (module_source / name).is_file()]
    missing += [str(path) for path in support_sources if not path.is_file()]
    if missing:
        raise SystemExit(f"missing package inputs: {missing}")

    copied_module = output / "solver_extensions" / "hccbHeliumTransport"
    if copied_module.exists():
        shutil.rmtree(copied_module)
    for relative in MODULE_FILES:
        destination = copied_module / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(module_source / relative, destination)

    for source in support_sources:
        destination = output / "code" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    executable_files = (
        "compile_prepare_direct_no_solver.sbatch",
        "run_direct_matched_initial_smoke.sbatch",
        "submit_direct_matched_initial_smoke.sh",
        *(f"code/{source.name}" for source in support_sources),
    )
    for relative in executable_files:
        (output / relative).chmod(0o755)

    checksummed = sorted(
        path
        for path in output.rglob("*")
        if path.is_file()
        and path.name not in {"PACKAGE_SHA256SUMS", "PACKAGE_MANIFEST.json", "READY"}
        and "__pycache__" not in path.parts
        and not path.name.startswith("._")
    )
    checksums = {
        path.relative_to(output).as_posix(): sha256(path) for path in checksummed
    }
    checksum_path = output / "PACKAGE_SHA256SUMS"
    checksum_path.write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in checksums.items()),
        encoding="ascii",
    )

    plan = json.loads((output / "smoke_plan.json").read_text(encoding="utf-8"))
    manifest = {
        "status": "matched_initial_direct_transport_package_ready_no_solver_started",
        "package_root": output.name,
        "file_count": len(checksums),
        "checksums_sha256": sha256(checksum_path),
        "sequence_id": plan["sequence_id"],
        "direct_transport_type": "hccbHelium",
        "registered_parameter_ids": ["P070", "P071"],
        "physical_correlations_changed": False,
        "operating_conditions_changed": False,
        "openfoam_solver_started": False,
        "post_solver_observable_export_included": True,
        "observable_signal_count": 15,
        "solver_submission_requires_exact_phrase": "批准短算",
    }
    manifest_path = output / "PACKAGE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    ready = {
        "status": "ready_for_cloud_no_solver_preflight",
        "manifest_sha256": sha256(manifest_path),
        "checksums_sha256": sha256(checksum_path),
        "solver_approved": False,
    }
    (output / "READY").write_text(
        json.dumps(ready, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**manifest, "ready_sha256": sha256(output / "READY")}, indent=2))


if __name__ == "__main__":
    main()
