#!/usr/bin/env python3
"""Finalize a cross-packing mesh after basic and strict checkMesh diagnostics.

The pore-resolved reference mesh passes the ordinary OpenFOAM checkMesh checks
but retains documented failures under ``-allTopology -allGeometry``.  This
finalizer therefore requires both fluid and solid ordinary checks to report
``Mesh OK`` while preserving every strict diagnostic in the completion record.
It never changes the mesh and never starts a flow or heat-transfer solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def basic_mesh_ok(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    return "Mesh OK." in text and "FOAM FATAL" not in text


def directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(202, 303), required=True)
    parser.add_argument("--expected-packing-sha256", required=True)
    parser.add_argument(
        "--failure-record",
        type=Path,
        help="failure record path; defaults to STAGE/mesh_preprocess_failure.json",
    )
    args = parser.parse_args()

    stage = args.stage.resolve()
    run_root = args.run_root.resolve()
    if not stage.is_dir():
        raise SystemExit(f"stage directory is missing: {stage}")
    if run_root.exists():
        raise SystemExit(f"run root already exists: {run_root}")
    if stage.parent != run_root.parent:
        raise SystemExit("stage and run root must share one parent for atomic rename")
    if "/home/" in f"{run_root}/" or str(run_root) == "/home":
        raise SystemExit("run root must not be under /home")

    required = {
        "case_manifest": stage / "case_manifest.json",
        "strict_summary": stage / "mesh_check_summary.json",
        "strict_fluid_log": stage / "log.checkMesh.fluid",
        "strict_solid_log": stage / "log.checkMesh.solid",
        "basic_fluid_log": stage / "log.checkMesh.fluid.basic_diagnostic_20260725",
        "basic_solid_log": stage / "log.checkMesh.solid.basic_diagnostic_20260725",
        "allmesh_resource": stage / "resource.Allmesh.json",
        "fluid_resource": stage / "resource.checkMesh.fluid.json",
        "solid_resource": stage / "resource.checkMesh.solid.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise SystemExit("required mesh result files are missing:\n" + "\n".join(missing))

    case_manifest = read_json(required["case_manifest"])
    strict_summary = read_json(required["strict_summary"])
    if case_manifest.get("source_packing_sha256") != args.expected_packing_sha256:
        raise SystemExit("mesh packing checksum does not match the frozen input")
    if case_manifest.get("new_physical_parameters") != []:
        raise SystemExit("mesh manifest unexpectedly contains new physical parameters")

    phase_resources = {
        name: read_json(required[key])
        for name, key in (
            ("Allmesh", "allmesh_resource"),
            ("checkMesh.fluid.strict", "fluid_resource"),
            ("checkMesh.solid.strict", "solid_resource"),
        )
    }
    bad_phases = [
        name
        for name, payload in phase_resources.items()
        if int(payload.get("return_code", -1)) != 0
    ]
    if bad_phases:
        raise SystemExit("mesh command failed: " + ", ".join(bad_phases))

    basic = {
        "fluid": basic_mesh_ok(required["basic_fluid_log"]),
        "solid": basic_mesh_ok(required["basic_solid_log"]),
    }
    strict = {
        region: {
            "failed_check_count": int(strict_summary[region]["failed_check_count"]),
            "maximum_non_orthogonality_deg": float(
                strict_summary[region]["maximum_non_orthogonality_deg"]
            ),
            "maximum_skewness": float(strict_summary[region]["maximum_skewness"]),
            "small_determinant_cells": int(
                strict_summary[region]["small_determinant_cells"]
            ),
            "concave_cells": int(strict_summary[region]["concave_cells"]),
        }
        for region in ("fluid", "solid")
    }
    common = {
        "seed": args.seed,
        "source_packing_sha256": args.expected_packing_sha256,
        "case_manifest_sha256": sha256(required["case_manifest"]),
        "strict_summary_sha256": sha256(required["strict_summary"]),
        "basic_check": {
            region: {
                "mesh_ok": basic[region],
                "log_sha256": sha256(required[f"basic_{region}_log"]),
            }
            for region in ("fluid", "solid")
        },
        "strict_check": strict,
        "strict_diagnostics_retained": True,
        "fluid_is_one_connected_region": bool(
            strict_summary["checks"]["fluid_is_one_connected_region"]
        ),
        "cell_volume_porosity": float(strict_summary["cell_volume_porosity"]),
        "triangulated_porosity": float(strict_summary["triangulated_porosity"]),
        "phases": phase_resources,
        "heat_transfer_solver_started": False,
        "mesh_parameters_changed": False,
        "physical_parameters_changed": False,
    }

    if not all(basic.values()) or not common["fluid_is_one_connected_region"]:
        failure_path = args.failure_record or stage / "mesh_preprocess_failure.json"
        failure = {
            **common,
            "status": "mesh_preprocess_not_accepted",
            "reason": (
                "ordinary checkMesh did not report Mesh OK for both regions"
                if not all(basic.values())
                else "fluid region is not singly connected"
            ),
            "stage_preserved": str(stage),
        }
        failure_path.write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(failure, indent=2))
        return 3

    completion = {
        **common,
        "status": "independent_packing_mesh_preprocess_complete",
        "acceptance_basis": (
            "ordinary fluid and solid checkMesh report Mesh OK; strict "
            "-allTopology/-allGeometry diagnostics are retained quantitatively"
        ),
        "output_bytes_before_completion_record": directory_bytes(stage),
        "run_root": str(run_root),
    }
    completion_path = stage / "cloud_mesh_completion.json"
    completion_path.write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )

    os.replace(stage, run_root)
    sidecar = Path(f"{stage}.build_manifest.stdout.json")
    if sidecar.is_file():
        os.replace(sidecar, run_root / "build_manifest.stdout.json")
    print(json.dumps(completion, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
