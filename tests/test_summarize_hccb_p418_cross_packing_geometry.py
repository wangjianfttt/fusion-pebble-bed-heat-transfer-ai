#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from summarize_hccb_p418_cross_packing_geometry import (  # noqa: E402
    render_latex_table,
    summarize_packings,
    write_outputs,
)


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CrossPackingGeometryTest(unittest.TestCase):
    def test_three_packings_and_screening_directions_are_summarized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packing_root = root / "data/apd006_hccb_source_sequence_target_packings"
            packing_root.mkdir(parents=True)
            records = []
            manifests = {}
            for index, seed in enumerate((101, 202, 303)):
                folder = packing_root / f"seed{seed}_s80_xlo_ycentre"
                folder.mkdir()
                packing = folder / "packing.npz"
                centres = np.asarray(
                    [[0.0006 + 0.0001 * index, 0.001, 0.001],
                     [0.0017, 0.001, 0.001],
                     [0.0028, 0.001, 0.001]],
                    dtype=float,
                )
                np.savez(
                    packing,
                    centres_m=centres,
                    physical_radius_m=np.asarray(0.0005),
                    meshing_radius_m=np.asarray(0.000495),
                )
                records.append(
                    {
                        "seed": seed,
                        "particle_count": 3,
                        "crop_porosity_geometric": 0.40 + 0.001 * index,
                        "packing_npz_sha256": checksum(packing),
                        "checks": {"geometry": True},
                    }
                )
                manifest = root / f"manifest_{seed}.json"
                manifest.write_text(
                    json.dumps(
                        {
                            "source_packing_sha256": checksum(packing),
                            "crop_lower_m": [0.0, 0.0, 0.0],
                            "crop_upper_m": [0.0035, 0.0020, 0.0020],
                            "intersecting_particle_count": 3,
                            "retained_particle_fragment_count": 3,
                            "triangulated_porosity": 0.39 + 0.002 * index,
                            "omitted_solid_volume_m3": 0.0,
                            "porosity_change_from_omission": 0.0,
                            "new_physical_parameters": [],
                        }
                    ),
                    encoding="utf-8",
                )
                manifests[seed] = manifest
            packing_set = root / "packing_set.json"
            packing_set.write_text(json.dumps(records), encoding="utf-8")
            plan = root / "plan.json"
            condition_ids = [f"case_{index}" for index in range(9)]
            plan.write_text(
                json.dumps(
                    {
                        "screening_design": {
                            "conditions": [
                                {"condition_id": identifier}
                                for identifier in condition_ids
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            physical = root / "physical.csv"
            with physical.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["condition_id", "cooling_wall_heat_direction"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "condition_id": "case_0",
                        "cooling_wall_heat_direction": "wall_to_fluid",
                    }
                )
                writer.writerow(
                    {
                        "condition_id": "case_1",
                        "cooling_wall_heat_direction": "fluid_to_wall",
                    }
                )
            rows, summary = summarize_packings(
                root, packing_set, manifests, plan, physical
            )
            output = root / "output"
            write_outputs(output, rows, summary)
            latex = render_latex_table(rows)
            chinese = (output / "P418_三套颗粒排列几何差异_CN.md").read_text(
                encoding="utf-8"
            )
        self.assertEqual(len(rows), 3)
        self.assertEqual(summary["screening_known_physical_direction_count"], 2)
        self.assertEqual(summary["screening_wall_to_fluid_count"], 1)
        self.assertEqual(summary["screening_fluid_to_wall_count"], 1)
        self.assertEqual(len(summary["screening_unknown_condition_ids"]), 7)
        self.assertIn("局部流道", chinese)
        self.assertIn("Geometric comparison", latex)
        self.assertIn(r"$N_{\mathrm{wall}}$", latex)
        self.assertIn("101 & 3", latex)


if __name__ == "__main__":
    unittest.main()
