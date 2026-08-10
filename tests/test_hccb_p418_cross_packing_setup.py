#!/usr/bin/env python3
"""Dry-run test for the cross-packing OpenFOAM setup wrapper."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/run_hccb_p418_cross_packing_setup.sh"


class HCCBP418CrossPackingSetupTest(unittest.TestCase):
    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_default_run_creates_nothing(self) -> None:
        environment = os.environ.copy()
        environment.update({"ROOT": str(ROOT), "EXECUTE": "0"})
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertIn("seed 202", result.stdout)
        self.assertIn("seed 303", result.stdout)
        self.assertIn("reference seed101 local mesh", result.stdout)
        self.assertIn("crop box (dp): 1.234 5.157 3.921 8.163 2.906 6.396", result.stdout)
        self.assertIn("9 exact P418 conditions", result.stdout)
        self.assertIn("dry run only", result.stdout)
        self.assertFalse(
            (ROOT / "hccb_dense_snappy_g2_nativezone_r2_seed202").exists()
        )
        self.assertFalse(
            (ROOT / "hccb_dense_snappy_g2_nativezone_r2_seed303").exists()
        )

    def test_mesh_settings_are_read_from_seed101_record(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("BASE_MESH_MANIFEST", text)
        self.assertIn('base["crop_box_dp"]', text)
        self.assertIn('base["numerical_controls"][key]', text)
        self.assertNotIn("--crop-box-dp 1.234", text)
        self.assertNotIn("--cells-per-diameter 10.101010101", text)
        self.assertIn("mesh_stage=", text)
        self.assertIn("--resume-existing", text)
        self.assertIn("reuse completed seed${seed} mesh", text)
        self.assertNotIn("refusing to replace an existing seed${seed} mesh or matrix", text)


if __name__ == "__main__":
    unittest.main()
