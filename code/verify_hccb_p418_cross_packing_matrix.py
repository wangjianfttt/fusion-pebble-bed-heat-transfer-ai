#!/usr/bin/env python3
"""Verify one nine-condition P418 independent-packing matrix before solving."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def condition_map(records: list[dict], source: str) -> dict[str, dict]:
    mapped: dict[str, dict] = {}
    for record in records:
        identifier = record.get("condition_id")
        if not identifier:
            raise ValueError(f"{source} contains a condition without condition_id")
        if identifier in mapped:
            raise ValueError(f"{source} contains duplicate condition {identifier}")
        mapped[identifier] = record
    return mapped


def same_number(left: object, right: object) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-12)


def verify(
    *,
    seed: int,
    plan_path: Path,
    matrix_manifest_path: Path,
    matrix_root: Path,
    mesh_manifest_path: Path,
) -> dict:
    if seed not in (202, 303):
        raise ValueError("seed must be 202 or 303")

    plan = load_json(plan_path)
    matrix = load_json(matrix_manifest_path)
    mesh = load_json(mesh_manifest_path)

    expected = condition_map(
        plan["screening_design"]["conditions"], "cross-packing plan"
    )
    actual = condition_map(matrix.get("cases", []), "matrix manifest")
    directories = {
        path.name
        for path in matrix_root.glob("u*_T*_q*")
        if path.is_dir()
    }

    expected_ids = set(expected)
    actual_ids = set(actual)
    if len(expected_ids) != 9:
        raise ValueError(f"plan must define nine unique conditions; found {len(expected_ids)}")
    if actual_ids != expected_ids:
        raise ValueError(
            "matrix conditions differ from the declared nine-condition plan: "
            f"missing={sorted(expected_ids - actual_ids)}, "
            f"extra={sorted(actual_ids - expected_ids)}"
        )
    if directories != expected_ids:
        raise ValueError(
            "case directories differ from the declared nine-condition plan: "
            f"missing={sorted(expected_ids - directories)}, "
            f"extra={sorted(directories - expected_ids)}"
        )

    numeric_fields = (
        "inlet_velocity_m_s",
        "inlet_temperature_K",
        "solid_heat_source_MW_m3",
    )
    for identifier in sorted(expected_ids):
        for field in numeric_fields:
            if field not in actual[identifier]:
                raise ValueError(f"{identifier} is missing {field} in the matrix manifest")
            if not same_number(actual[identifier][field], expected[identifier][field]):
                raise ValueError(
                    f"{identifier} {field} differs from the declared plan: "
                    f"{actual[identifier][field]} != {expected[identifier][field]}"
                )

    packing_records = {
        int(record["seed"]): record for record in plan["packing_realisations"]
    }
    if seed not in packing_records:
        raise ValueError(f"seed{seed} is absent from the cross-packing plan")
    expected_packing_hash = packing_records[seed]["packing_npz_sha256"]
    actual_packing_hash = mesh.get("source_packing_sha256")
    if actual_packing_hash != expected_packing_hash:
        raise ValueError(
            f"seed{seed} mesh was not built from the declared packing: "
            f"{actual_packing_hash} != {expected_packing_hash}"
        )

    case_hashes = {record.get("mesh_source_packing_sha256") for record in actual.values()}
    if case_hashes != {expected_packing_hash}:
        raise ValueError(
            f"seed{seed} cases do not all use the declared packing hash: "
            f"{sorted(str(value) for value in case_hashes)}"
        )

    if matrix.get("selected_case_count") != 9:
        raise ValueError("matrix manifest selected_case_count must equal 9")
    if matrix.get("mode") != "selected":
        raise ValueError("independent-packing matrix must be built in selected mode")
    if matrix.get("new_physical_parameter_values_added", []) not in ([], None):
        raise ValueError("matrix unexpectedly records new physical parameter values")

    return {
        "status": "p418_cross_packing_matrix_matches_declared_plan",
        "packing_seed": seed,
        "condition_count": 9,
        "condition_ids": sorted(expected_ids),
        "packing_npz_sha256": expected_packing_hash,
        "new_physical_parameter_values_added": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--matrix-manifest", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--mesh-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = verify(
        seed=args.seed,
        plan_path=args.plan.resolve(),
        matrix_manifest_path=args.matrix_manifest.resolve(),
        matrix_root=args.matrix_root.resolve(),
        mesh_manifest_path=args.mesh_manifest.resolve(),
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
