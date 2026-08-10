#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/run_hccb_p418_mixed_endpoint_smoke.sh"


class P418MixedEndpointSmokeRunnerTest(unittest.TestCase):
    def test_shell_syntax_and_required_steps(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("build_hccb_p418_completed_smoke_splits.py", text)
        self.assertIn("run_hccb_p418_60_postprocess.sh", text)
        self.assertIn("cp -al", text)
        self.assertIn("mixed_endpoint_postprocess_ready", text)
        self.assertIn("software-path check only", text)


if __name__ == "__main__":
    unittest.main()
