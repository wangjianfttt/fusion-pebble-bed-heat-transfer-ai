#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_regional_topology.py"


class P418RegionalTopologyTest(unittest.TestCase):
    def test_connected_typed_hierarchy_preserves_volume_and_interface(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            topology = base / "topology.npz"
            native = base / "native.npz"
            output = base / "regional"
            np.savez_compressed(
                topology,
                fluid_cell_centroid_m=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0.]], dtype=float),
                fluid_cell_volume_m3=np.array([1., 2., 1., 2.]),
                solid_cell_centroid_m=np.array([[0, 1, 0], [1, 1, 0], [2, 1, 0], [3, 1, 0.]], dtype=float),
                solid_cell_volume_m3=np.array([2., 1., 2., 1.]),
            )
            pairs = [(0, 1, 0), (1, 2, 0), (2, 3, 0), (4, 5, 1), (5, 6, 1), (6, 7, 1), (1, 5, 2), (2, 6, 2)]
            src = np.array([a for a, b, k in pairs] + [b for a, b, k in pairs])
            dst = np.array([b for a, b, k in pairs] + [a for a, b, k in pairs])
            kind = np.array([k for a, b, k in pairs] * 2, dtype=np.int8)
            sign = np.r_[np.ones(len(pairs)), -np.ones(len(pairs))]
            np.savez_compressed(
                native,
                node_region_type=np.array([0] * 4 + [1] * 4, dtype=np.int8),
                edge_source_global=src,
                edge_target_global=dst,
                edge_kind=kind,
                edge_area_m2=np.ones(len(src)),
                edge_area_vector_m2=np.column_stack((sign, np.zeros((len(src), 2)))),
                edge_face_centroid_m=np.zeros((len(src), 3)),
            )
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--shared-topology",
                    str(topology),
                    "--native-graph",
                    str(native),
                    "--levels",
                    "2",
                    "--subsample-factor",
                    "2",
                    "--output-dir",
                    str(output),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "p418_multiregion_regional_topology_ready")
            self.assertTrue(all(summary["checks"].values()))
            with np.load(output / "regional_topology.npz") as hierarchy:
                self.assertEqual(len(hierarchy["level_0_node_type"]), 4)
                self.assertEqual(len(hierarchy["level_1_node_type"]), 2)
                self.assertIn(2, hierarchy["level_0_edge_kind"])


if __name__ == "__main__":
    unittest.main()
