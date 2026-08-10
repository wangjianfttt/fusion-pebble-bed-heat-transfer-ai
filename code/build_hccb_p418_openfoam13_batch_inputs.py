#!/usr/bin/env python3
"""Prepare the pending P418 OpenFOAM inputs without running a solver."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path


MESH_REGIONS = ("fluid", "solid")
REQUIRED_CASE_FILES = (
    "0/fluid/U",
    "0/fluid/T",
    "0/fluid/p",
    "0/fluid/p_rgh",
    "0/solid/T",
    "constant/fluid/physicalProperties",
    "constant/solid/physicalProperties",
    "constant/solid/fvModels",
    "system/controlDict",
    "system/decomposeParDict",
    "cht_smoke_metadata.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1.0e-10, abs_tol=1.0e-12):
        raise ValueError(f"{name}: {actual:.12g} != {expected:.12g}")


def read_cloud_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    pending = [row for row in rows if row["submit_to_cloud"] == "yes"]
    if not pending:
        raise ValueError("cloud case table contains no pending P418 condition")
    if len({row["condition_id"] for row in pending}) != len(pending):
        raise ValueError("cloud case table contains duplicate pending condition IDs")
    return pending


def verified_cases(path: Path) -> dict[str, dict[str, object]]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "hccb_p418_60_actual_case_inputs_verified":
        raise ValueError("the P418 OpenFOAM input check has not completed")
    if summary.get("all_openfoam_dictionary_values_match_registered_sources") is not True:
        raise ValueError("the P418 OpenFOAM dictionaries are not verified")
    cases = summary.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("the P418 input-check summary contains no cases")
    return {str(row["condition_id"]): row for row in cases}


def validate_case(
    case: Path,
    row: dict[str, str],
    verified: dict[str, object],
) -> dict[str, object]:
    for relative in REQUIRED_CASE_FILES:
        path = case / relative
        if not path.is_file():
            raise FileNotFoundError(path)
    for region in MESH_REGIONS:
        mesh = case / "constant" / region / "polyMesh"
        if not mesh.is_dir() or not any(mesh.iterdir()):
            raise FileNotFoundError(mesh)

    condition_id = row["condition_id"]
    metadata = json.loads(
        (case / "cht_smoke_metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("operating_condition_id") != condition_id:
        raise ValueError(f"{condition_id}: metadata contains another condition ID")
    if metadata.get("source_channel_volume_flow_preserved") is not True:
        raise ValueError(f"{condition_id}: inlet-channel volume flow is not preserved")
    if metadata.get("new_fitted_physical_parameters") != []:
        raise ValueError(f"{condition_id}: unexpected fitted physical parameter")

    expected_velocity = float(row["inlet_velocity_m_s"])
    expected_temperature = float(row["inlet_temperature_K"])
    expected_source = float(row["solid_heat_source_MW_m3"])
    close(float(metadata["inlet_velocity_m_s"]), expected_velocity, condition_id)
    close(float(metadata["inlet_temperature_K"]), expected_temperature, condition_id)
    close(
        float(metadata["solid_heat_source_W_m3"]) / 1.0e6,
        expected_source,
        condition_id,
    )
    close(float(verified["inlet_velocity_m_s"]), expected_velocity, condition_id)
    close(float(verified["inlet_temperature_K"]), expected_temperature, condition_id)
    close(float(verified["solid_heat_source_MW_m3"]), expected_source, condition_id)
    return {
        "condition_id": condition_id,
        "inlet_velocity_m_s": expected_velocity,
        "inlet_temperature_K": expected_temperature,
        "solid_heat_source_MW_m3": expected_source,
        "pore_opening_boundary_velocity_m_s": float(
            metadata["pore_opening_boundary_velocity_m_s"]
        ),
        "inlet_open_area_fraction": float(metadata["inlet_open_area_fraction"]),
    }


def same_mesh(reference: Path, candidate: Path) -> None:
    reference_files = {
        path.relative_to(reference): path for path in reference.rglob("*") if path.is_file()
    }
    candidate_files = {
        path.relative_to(candidate): path for path in candidate.rglob("*") if path.is_file()
    }
    if set(reference_files) != set(candidate_files):
        raise ValueError(f"mesh file set differs: {candidate}")
    for relative, reference_file in reference_files.items():
        candidate_file = candidate_files[relative]
        if reference_file.stat().st_size != candidate_file.stat().st_size:
            raise ValueError(f"mesh file size differs: {candidate_file}")
        if not reference_file.samefile(candidate_file) and sha256(reference_file) != sha256(
            candidate_file
        ):
            raise ValueError(f"mesh file content differs: {candidate_file}")


def copy_case_without_mesh(source: Path, destination: Path) -> None:
    shutil.copytree(source / "0", destination / "0", copy_function=shutil.copy2)
    shutil.copytree(source / "system", destination / "system", copy_function=shutil.copy2)

    def ignore_mesh(directory: str, names: list[str]) -> set[str]:
        path = Path(directory)
        if path.name in MESH_REGIONS and "polyMesh" in names:
            return {"polyMesh"}
        return set()

    shutil.copytree(
        source / "constant",
        destination / "constant",
        copy_function=shutil.copy2,
        ignore=ignore_mesh,
    )
    shutil.copy2(
        source / "cht_smoke_metadata.json",
        destination / "cht_smoke_metadata.json",
    )


def write_checksums(root: Path, output: Path) -> dict[str, str]:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path != output)
    hashes = {str(path.relative_to(root)): sha256(path) for path in files}
    output.write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in hashes.items()),
        encoding="utf-8",
    )
    return hashes


def build(
    matrix_root: Path,
    cloud_table: Path,
    input_check: Path,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir == Path("/home") or Path("/home") in output_dir.parents:
        raise ValueError(f"batch package output must not be under /home: {output_dir}")
    rows = read_cloud_rows(cloud_table)
    verified = verified_cases(input_check)
    missing = sorted({row["condition_id"] for row in rows}.difference(verified))
    if missing:
        raise ValueError(f"pending cases are missing from the input check: {missing}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    case_root = output_dir / "case_inputs"
    shared_root = output_dir / "shared_mesh"
    case_root.mkdir(parents=True)

    reference_case = matrix_root / rows[0]["condition_id"]
    validated: list[dict[str, object]] = []
    for row in rows:
        condition_id = row["condition_id"]
        case = matrix_root / condition_id
        if not case.is_dir():
            raise FileNotFoundError(case)
        validated.append(validate_case(case, row, verified[condition_id]))
        for region in MESH_REGIONS:
            same_mesh(
                reference_case / "constant" / region / "polyMesh",
                case / "constant" / region / "polyMesh",
            )
        destination = case_root / condition_id
        copy_case_without_mesh(case, destination)
        write_checksums(destination, destination / "INPUT_SHA256SUMS")

    for region in MESH_REGIONS:
        shutil.copytree(
            reference_case / "constant" / region / "polyMesh",
            shared_root / region / "polyMesh",
            copy_function=shutil.copy2,
        )
    shared_hashes = write_checksums(shared_root, shared_root / "SHA256SUMS")

    manifest = {
        "status": "p418_openfoam13_pending_batch_inputs_ready",
        "case_count": len(validated),
        "condition_ids": [row["condition_id"] for row in validated],
        "case_inputs": validated,
        "shared_mesh_file_count": len(shared_hashes),
        "physical_parameter_source": "P418 registered literature condition matrix",
        "new_physical_parameters": [],
        "solver_started": False,
    }
    (output_dir / "batch_input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--cloud-table", type=Path, required=True)
    parser.add_argument("--input-check", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(
        args.matrix_root.resolve(),
        args.cloud_table.resolve(),
        args.input_check.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
