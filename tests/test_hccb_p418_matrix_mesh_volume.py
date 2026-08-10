#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_dense_cht_p418_matrix import source_solid_volume  # noqa: E402


class MatrixMeshVolumeTest(unittest.TestCase):
    def test_unsolved_mesh_uses_checkmesh_solid_volume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            mesh = base / "mesh"
            mesh.mkdir()
            summary = base / "summary.json"
            summary.write_text(
                json.dumps({"solid": {"volume_m3": 1.234e-6}}),
                encoding="utf-8",
            )
            self.assertAlmostEqual(source_solid_volume(mesh, summary), 1.234e-6)


if __name__ == "__main__":
    unittest.main()
