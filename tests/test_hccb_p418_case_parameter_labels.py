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

from update_hccb_p418_case_parameter_labels import (  # noqa: E402
    ACTUAL_INPUT_IDS,
    REFERENCE_ONLY,
    parameter_rows,
    update_case,
)


class P418CaseParameterLabelsTest(unittest.TestCase):
    def test_reference_values_are_not_listed_as_numerical_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "parameters.csv"
            fields = ["parameter_id", "value"]
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for parameter_id in ACTUAL_INPUT_IDS + tuple(REFERENCE_ONLY):
                    writer.writerow({"parameter_id": parameter_id, "value": "source"})
            case = root / "u0p20_T700_q6p85"
            sample = case / "training_sample_300_schema3"
            sample.mkdir(parents=True)
            (case / "cht_smoke_metadata.json").write_text(
                json.dumps(
                    {
                        "operating_condition_id": case.name,
                        "parameter_ids": list(ACTUAL_INPUT_IDS) + list(REFERENCE_ONLY),
                    }
                ),
                encoding="utf-8",
            )
            (sample / "metadata.json").write_text(
                json.dumps({"literature_parameters": []}), encoding="utf-8"
            )
            result = update_case(
                case,
                sample_directory="training_sample_300_schema3",
                rows=parameter_rows(manifest),
            )
            updated = json.loads((case / "cht_smoke_metadata.json").read_text())
            self.assertEqual(tuple(updated["parameter_ids"]), ACTUAL_INPUT_IDS)
            self.assertEqual(set(updated["literature_comparison_parameter_ids"]), set(REFERENCE_ONLY))
            self.assertTrue(result["sample_metadata_updated"])


if __name__ == "__main__":
    unittest.main()
