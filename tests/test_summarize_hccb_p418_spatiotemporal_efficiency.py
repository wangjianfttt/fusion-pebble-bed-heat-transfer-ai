#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/summarize_hccb_p418_spatiotemporal_efficiency.py"


class P418SpatiotemporalEfficiencySummaryTest(unittest.TestCase):
    def test_compares_only_matched_full_graph_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = {
                "nodes": 46089,
                "edges": 245848,
                "time_points": 37,
                "torch_num_threads": 8,
                "physical_parameter_ids": ["P418", "P429"],
                "new_physical_parameters": [],
                "initial_maximum_absolute_error": 0.0,
                "hydrodynamic_maximum_absolute_error": 0.0,
                "loss_finite": True,
                "all_gradients_present": True,
                "all_gradients_finite": True,
            }
            repeated = root / "repeated.json"
            factorized = root / "factorized.json"
            repeated.write_text(
                json.dumps(
                    {
                        **common,
                        "spatial_temporal_mode": "repeated_query_spatial",
                        "elapsed_seconds": 100.0,
                    }
                ),
                encoding="utf-8",
            )
            factorized.write_text(
                json.dumps(
                    {
                        **common,
                        "spatial_temporal_mode": "factorized_static_spatial",
                        "elapsed_seconds": 40.0,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "output"
            subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--repeated-summary",
                    str(repeated),
                    "--factorized-summary",
                    str(factorized),
                    "--output-dir",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(result["factorized_update_speedup"], 2.5)
            self.assertTrue(result["initial_state_and_hydrodynamics_exact"])

            changed = json.loads(factorized.read_text(encoding="utf-8"))
            changed["time_points"] = 13
            factorized.write_text(json.dumps(changed), encoding="utf-8")
            rejected = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--repeated-summary",
                    str(repeated),
                    "--factorized-summary",
                    str(factorized),
                    "--output-dir",
                    str(root / "bad"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("timing inputs differ", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
