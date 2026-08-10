#!/usr/bin/env python3
"""Complete the physical-parameter record of an existing P418 matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from hccb_p418_source_contract import (
    ALL_STEADY_PHYSICAL_PARAMETER_IDS,
    CASE_PHYSICS_PARAMETER_IDS,
    MESH_GEOMETRY_SOURCE_PARAMETER_IDS,
    OPERATING_PARAMETER_IDS,
)


def parameter_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["parameter_id"]: row for row in csv.DictReader(stream)}


def update_matrix(matrix_root: Path, manifest_path: Path) -> dict[str, object]:
    rows = parameter_rows(manifest_path)
    for parameter_id in ALL_STEADY_PHYSICAL_PARAMETER_IDS:
        row = rows.get(parameter_id)
        if row is None or row.get("status") != "extracted":
            raise ValueError(f"{parameter_id} is not an extracted literature value")

    matrix_path = matrix_root / "matrix_manifest.json"
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    case_records = payload.get("cases", [])
    if len(case_records) != 60:
        raise ValueError(f"expected 60 P418 cases, found {len(case_records)}")

    checked_cases: list[str] = []
    for record in case_records:
        condition_id = str(record["condition_id"])
        metadata_path = matrix_root / condition_id / "cht_smoke_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        actual = tuple(metadata.get("parameter_ids", ()))
        if actual != CASE_PHYSICS_PARAMETER_IDS:
            raise ValueError(
                f"{condition_id} calculation inputs differ from the P418 contract: {actual}"
            )
        checked_cases.append(condition_id)

    payload["parameter_ids"] = list(ALL_STEADY_PHYSICAL_PARAMETER_IDS)
    payload["operating_parameter_ids"] = list(OPERATING_PARAMETER_IDS)
    payload["case_physics_parameter_ids"] = list(CASE_PHYSICS_PARAMETER_IDS)
    payload["mesh_geometry_source_parameter_ids"] = list(
        MESH_GEOMETRY_SOURCE_PARAMETER_IDS
    )
    payload["parameter_source_statement"] = (
        "P048, P049, P050, P390, P404 and P423 define the published source-packing "
        "construction, crop and meshing sequence. The later fine local crop, retained "
        "particle fragments and realized triangulated porosity are computed mesh properties. "
        "All case-level "
        "flow, heat-transfer, material and boundary inputs are listed in "
        "case_physics_parameter_ids and are extracted literature values."
    )

    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=matrix_root, delete=False
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, matrix_path)

    return {
        "status": "p418_matrix_physical_parameter_sources_complete",
        "case_count": len(checked_cases),
        "physical_parameter_count": len(ALL_STEADY_PHYSICAL_PARAMETER_IDS),
        "operating_parameter_ids": list(OPERATING_PARAMETER_IDS),
        "case_physics_parameter_ids": list(CASE_PHYSICS_PARAMETER_IDS),
        "mesh_geometry_source_parameter_ids": list(
            MESH_GEOMETRY_SOURCE_PARAMETER_IDS
        ),
        "all_parameters_are_extracted_literature_values": True,
        "realized_mesh_porosity_is_a_computed_geometry_result": True,
        "new_physical_parameters": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--parameter-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = update_matrix(args.matrix_root.resolve(), args.parameter_manifest.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
