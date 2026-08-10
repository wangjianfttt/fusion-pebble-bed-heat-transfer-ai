#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "code/run_hccb_p418_60_transient_models.sh"


class P418TransientRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = RUNNER.read_text(encoding="utf-8")

    def test_formal_runner_requires_sixty_direct_curves(self) -> None:
        self.assertIn("completed} -ne 60", self.text)
        self.assertIn("--run-role formal", self.text)
        self.assertIn("export_hccb_p418_transient_observables.py", self.text)

    def test_source_sized_transformer_defaults_remain_in_trainer(self) -> None:
        trainer = (ROOT / "code/train_hccb_p418_transient_observable_transformer.py").read_text(
            encoding="utf-8"
        )
        for text in [
            '"d_model": 256',
            '"heads": 8',
            '"layers": 5',
            '"epochs": 500',
            '"batch_size": 8',
            '"learning_rate": 1.0e-3',
            '"weight_decay": 1.0e-5',
            'default=FORMAL_ARCHITECTURE["d_model"]',
            'default=FORMAL_TRAINING["epochs"]',
            "validate_numerical_settings(args)",
        ]:
            self.assertIn(text, trainer)

    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)


if __name__ == "__main__":
    unittest.main()
