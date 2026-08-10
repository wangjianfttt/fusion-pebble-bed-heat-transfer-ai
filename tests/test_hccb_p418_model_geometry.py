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
SCRIPT = ROOT / "code/build_hccb_p418_model_geometry.py"


class P418ModelGeometryTest(unittest.TestCase):
    def test_boundary_roles_reach_regional_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            np.savez_compressed(
                base / "topology.npz",
                fluid_cell_volume_m3=np.ones(2),
                solid_cell_volume_m3=np.ones(2),
                fluid_boundary_face_owner=np.array([0, 1]),
                fluid_boundary_face_patch=np.array([0, 1]),
                solid_boundary_face_owner=np.array([0, 1]),
                solid_boundary_face_patch=np.array([0, 1]),
                interface_fluid_cell=np.array([1]),
                interface_solid_cell=np.array([0]),
            )
            (base / "dataset.json").write_text(
                json.dumps(
                    {
                        "shared_topology_file": "topology.npz",
                        "boundary_patch_names": {
                            "fluid": ["inlet", "fluid_to_solid"],
                            "solid": ["solid_to_fluid", "outlet"],
                        },
                    }
                )
            )
            np.savez_compressed(
                base / "regional.npz",
                fine_node_centroid_m=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1.]], dtype=float),
                level_0_parent_from_finer=np.array([0, 0, 1, 1]),
                level_0_node_type=np.array([0, 1]),
                level_0_volume_m3=np.array([2., 2.]),
            )
            roles = {
                "role_order": ["inlet", "outlet", "fluid_solid_interface"],
                "regions": {
                    "fluid": {
                        "inlet": {"role": "inlet"},
                        "fluid_to_solid": {"role": "fluid_solid_interface"},
                    },
                    "solid": {
                        "solid_to_fluid": {"role": "fluid_solid_interface"},
                        "outlet": {"role": "outlet"},
                    },
                },
            }
            (base / "roles.json").write_text(json.dumps(roles))
            output = base / "output"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--dataset-index",
                    str(base / "dataset.json"),
                    "--regional-topology",
                    str(base / "regional.npz"),
                    "--boundary-roles",
                    str(base / "roles.json"),
                    "--output-dir",
                    str(output),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "p418_regional_model_geometry_ready")
            with np.load(output / "model_geometry.npz") as geometry:
                self.assertEqual(geometry["fine_boundary_role"].shape, (4, 3))
                self.assertEqual(geometry["level_0_boundary_volume_fraction"].shape, (2, 3))
                self.assertEqual(
                    geometry["boundary_role_names"].tolist(),
                    ["inlet", "outlet", "fluid_solid_interface"],
                )


if __name__ == "__main__":
    unittest.main()
