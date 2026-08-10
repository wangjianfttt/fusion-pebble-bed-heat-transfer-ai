#!/usr/bin/env python3

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_physical_parameter_sources.py"
MANIFEST = ROOT / "parameters/literature_parameter_manifest.csv"
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_p418_physical_parameter_sources import SELECTION  # noqa: E402


class PhysicalParameterSourcesTest(unittest.TestCase):
    def test_focused_source_list_contains_only_extracted_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            csv_output = base / "sources.csv"
            md_output = base / "sources.md"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--manifest",
                    str(MANIFEST),
                    "--csv-output",
                    str(csv_output),
                    "--markdown-output",
                    str(md_output),
                ],
                check=True,
            )
            with csv_output.open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            expected_ids = [parameter_id for parameter_id, _, _ in SELECTION]
            self.assertEqual([row["parameter_id"] for row in rows], expected_ids)
            self.assertEqual(len({row["parameter_id"] for row in rows}), len(expected_ids))
            self.assertNotIn("P391", {row["parameter_id"] for row in rows})
            self.assertNotIn("P392", {row["parameter_id"] for row in rows})
            self.assertNotIn("local_pdf:", md_output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
