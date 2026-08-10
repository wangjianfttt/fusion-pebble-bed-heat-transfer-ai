#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/verify_hccb_p418_experimental_observation_sources.py"


class P418ExperimentalObservationSourcesTest(unittest.TestCase):
    def test_literature_sources_and_p418_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--sources",
                    str(
                        ROOT
                        / "parameters/hccb_p418_experimental_observation_sources.json"
                    ),
                    "--manifest",
                    str(ROOT / "parameters/literature_parameter_manifest.csv"),
                    "--output-dir",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(
                payload["status"], "experimental_observation_sources_verified"
            )
            self.assertGreaterEqual(payload["family_count"], 5)
            self.assertGreaterEqual(payload["unique_literature_parameter_count"], 15)
            self.assertEqual(payload["p418_assigned_numeric_values"], [])
            self.assertEqual(payload["new_physical_parameters"], [])
            self.assertTrue(
                (output / "P418_实验观测量与模型对应_CN.md").is_file()
            )

    def test_sensor_response_and_uncertainty_rules_are_explicit(self) -> None:
        payload = json.loads(
            (
                ROOT
                / "parameters/hccb_p418_experimental_observation_sources.json"
            ).read_text(encoding="utf-8")
        )
        rules = "\n".join(payload["usage_rules_cn"])
        self.assertIn("最近同相区域近似", rules)
        self.assertIn("标准不确定度", rules)
        self.assertIn("不作精确固定条件", rules)


if __name__ == "__main__":
    unittest.main()
