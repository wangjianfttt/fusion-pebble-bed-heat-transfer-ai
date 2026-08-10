#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_p418_data_split_table import (  # noqa: E402
    collect_steady_records,
    collect_transient_records,
    render_table,
)


class DataSplitTableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.steady = json.loads(
            (ROOT / "parameters/hccb_p418_model_splits.json").read_text(encoding="utf-8")
        )
        cls.transient = json.loads(
            (ROOT / "parameters/hccb_p418_step_response_splits.json").read_text(
                encoding="utf-8"
            )
        )
        cls.plan = json.loads(
            (ROOT / "parameters/hccb_p418_transient_step_plan.json").read_text(
                encoding="utf-8"
            )
        )

    def test_registered_splits_cover_complete_conditions_and_trajectories(self) -> None:
        steady_records = collect_steady_records(self.steady)
        transient_records = collect_transient_records(self.transient, self.plan)
        self.assertEqual(len(steady_records), 5)
        self.assertEqual(len(transient_records), 3)
        self.assertEqual(
            (steady_records[0]["train_count"], steady_records[0]["validation_count"], steady_records[0]["test_count"]),
            (36, 12, 12),
        )
        self.assertEqual(
            (transient_records[-1]["train_count"], transient_records[-1]["validation_count"], transient_records[-1]["test_count"]),
            (6, 2, 4),
        )
        latex = render_table(steady_records, transient_records)
        self.assertIn("Temperature extrapolation", latex)
        self.assertIn("$T_{\\rm in}=900$ K (15)", latex)
        self.assertIn("two unseen endpoint pairs (4)", latex)
        self.assertIn("no time point", latex)

    def test_overlapping_steady_roles_are_rejected(self) -> None:
        payload = copy.deepcopy(self.steady)
        split = payload["splits"]["interleaved_all_ranges"]
        split["validation"][0] = split["train"][0]
        with self.assertRaisesRegex(ValueError, "roles overlap"):
            collect_steady_records(payload)

    def test_reverse_endpoint_pair_crossing_roles_is_rejected(self) -> None:
        payload = copy.deepcopy(self.transient)
        split = payload["splits"]["pair_disjoint_stress_test"]
        split["validation"][0], split["test"][0] = split["test"][0], split["validation"][0]
        with self.assertRaisesRegex(ValueError, "forward/reverse endpoint pairs cross roles"):
            collect_transient_records(payload, self.plan)


if __name__ == "__main__":
    unittest.main()
