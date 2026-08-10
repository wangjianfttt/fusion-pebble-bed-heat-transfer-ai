#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_completed_smoke_splits.py"


class P418CompletedSmokeSplitsTest(unittest.TestCase):
    def test_completed_conditions_are_partitioned_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            matrix = temporary_path / "matrix"
            matrix.mkdir()
            conditions = []
            for index in range(5):
                condition_id = f"case_{index}"
                conditions.append({"condition_id": condition_id})
                case = matrix / condition_id
                case.mkdir()
                (case / "formal_sample_complete.json").write_text("{}\n")
            source = temporary_path / "source.json"
            source.write_text(
                json.dumps(
                    {
                        "source_parameter_id": "P418",
                        "source_title": "published condition matrix",
                        "source_doi": "https://doi.org/10.1016/j.ijheatmasstransfer.2023.124325",
                        "conditions": conditions,
                    }
                )
            )
            output = temporary_path / "smoke.json"
            subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--matrix-root",
                    str(matrix),
                    "--source-splits",
                    str(source),
                    "--output",
                    str(output),
                    "--expected-case-count",
                    "5",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output.read_text())
            split = payload["splits"]["completed_smoke"]
            combined = split["train"] + split["validation"] + split["test"]
            self.assertEqual(combined, [f"case_{index}" for index in range(5)])
            self.assertEqual(len(split["train"]), 3)
            self.assertEqual(payload["scope"], "completed_case_software_smoke_only")

    def test_expected_count_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            matrix = temporary_path / "matrix"
            matrix.mkdir()
            source = temporary_path / "source.json"
            source.write_text(
                json.dumps(
                    {
                        "source_parameter_id": "P418",
                        "source_title": "published condition matrix",
                        "source_doi": "doi",
                        "conditions": [],
                    }
                )
            )
            process = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--matrix-root",
                    str(matrix),
                    "--source-splits",
                    str(source),
                    "--output",
                    str(temporary_path / "smoke.json"),
                    "--expected-case-count",
                    "3",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("completed case count 0 != expected 3", process.stderr)


if __name__ == "__main__":
    unittest.main()
