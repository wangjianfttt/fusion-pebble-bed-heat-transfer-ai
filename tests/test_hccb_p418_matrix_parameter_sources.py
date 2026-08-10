#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from hccb_p418_source_contract import (  # noqa: E402
    ALL_STEADY_PHYSICAL_PARAMETER_IDS,
    CASE_PHYSICS_PARAMETER_IDS,
    MESH_GEOMETRY_SOURCE_PARAMETER_IDS,
)
from update_hccb_p418_matrix_parameter_sources import update_matrix  # noqa: E402


class P418MatrixParameterSourcesTest(unittest.TestCase):
    def build_fixture(self, root: Path, *, bad_case: bool = False) -> tuple[Path, Path]:
        matrix = root / "matrix"
        matrix.mkdir()
        records = []
        for index in range(60):
            condition = f"case_{index:02d}"
            records.append({"condition_id": condition})
            case = matrix / condition
            case.mkdir()
            ids = list(CASE_PHYSICS_PARAMETER_IDS)
            if bad_case and index == 59:
                ids.pop()
            (case / "cht_smoke_metadata.json").write_text(
                json.dumps({"parameter_ids": ids}), encoding="utf-8"
            )
        (matrix / "matrix_manifest.json").write_text(
            json.dumps({"cases": records, "parameter_ids": ["P418"]}),
            encoding="utf-8",
        )
        manifest = root / "parameters.csv"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["parameter_id", "status"])
            writer.writeheader()
            for parameter_id in ALL_STEADY_PHYSICAL_PARAMETER_IDS:
                writer.writerow({"parameter_id": parameter_id, "status": "extracted"})
        return matrix, manifest

    def test_existing_matrix_receives_complete_parameter_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            matrix, manifest = self.build_fixture(Path(directory))
            result = update_matrix(matrix, manifest)
            payload = json.loads((matrix / "matrix_manifest.json").read_text())
            self.assertEqual(tuple(payload["parameter_ids"]), ALL_STEADY_PHYSICAL_PARAMETER_IDS)
            self.assertEqual(
                tuple(payload["case_physics_parameter_ids"]), CASE_PHYSICS_PARAMETER_IDS
            )
            self.assertEqual(
                tuple(payload["mesh_geometry_source_parameter_ids"]),
                MESH_GEOMETRY_SOURCE_PARAMETER_IDS,
            )
            self.assertEqual(result["case_count"], 60)

    def test_changed_case_parameter_list_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            matrix, manifest = self.build_fixture(Path(directory), bad_case=True)
            with self.assertRaisesRegex(ValueError, "calculation inputs differ"):
                update_matrix(matrix, manifest)


if __name__ == "__main__":
    unittest.main()
