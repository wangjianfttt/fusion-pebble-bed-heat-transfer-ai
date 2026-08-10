#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "code/run_hccb_p418_mixed_endpoint_model_smoke.sh"


class P418MixedEndpointModelSmokeRunnerTest(unittest.TestCase):
    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)

    def test_scope_and_inputs_are_explicit(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        for required in (
            "hccb_p418_mixed_endpoint_smoke_model",
            "hccb_p418_mixed_endpoint_smoke_splits.json",
            "hccb_p418_mixed_endpoint_smoke_model_comparison_",
            'SPLIT_NAMES="completed_smoke"',
            'EPOCHS=${EPOCHS:-1}',
            'EXPECTED_CASES=8',
            "not a formal train/test sample",
            "pinn_data_only pinn graph transolver",
        ):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
