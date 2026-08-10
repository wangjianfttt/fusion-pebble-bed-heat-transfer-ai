#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from verify_hccb_p418_actual_case_inputs import (  # noqa: E402
    DEFAULT_MESH_MANIFEST,
    dp_dimensions,
    published_conditions,
    scalar_from_foam_value,
    source_rows,
    vector_from_foam_value,
    verify_geometry_sources,
)


class VerifyP418ActualCaseInputsTest(unittest.TestCase):
    def test_published_matrix_has_exactly_sixty_cases(self) -> None:
        value = (
            "Re_p_AVE<1.8;87 percent within +/-30 percent;60 cases from "
            "u_in=0.05,0.10,0.15,0.20,0.25 m/s x "
            "T_in=300,500,700,900 K x phi=4.85,6.85,8.85 MW/m3"
        )
        conditions = published_conditions(value)
        self.assertEqual(len(conditions), 60)
        self.assertEqual(conditions["u0p25_T900_q8p85"], (0.25, 900.0, 8.85))

    def test_openfoam_scalar_parser_rejects_multiple_values(self) -> None:
        self.assertEqual(scalar_from_foam_value("uniform 120000"), 120000.0)
        with self.assertRaises(ValueError):
            scalar_from_foam_value("uniform (0 0 0.05)")

    def test_openfoam_vector_parser(self) -> None:
        self.assertEqual(vector_from_foam_value("uniform ( 0 0 0.05 )"), (0.0, 0.0, 0.05))

    def test_dp_dimensions_ignore_extension_lengths(self) -> None:
        self.assertEqual(
            dp_dimensions("12.5dp x 12.5dp x 10dp; inlet channel=10dp"),
            (12.5, 12.5, 10.0),
        )

    def test_seed101_geometry_matches_all_registered_sources(self) -> None:
        rows = source_rows(ROOT / "parameters/hccb_p418_physical_parameter_sources.csv")
        result = verify_geometry_sources(
            rows,
            ROOT / "results/apd006_hccb_source_sequence_lammps/sweep/seed101_s80/packing_input_manifest.json",
            ROOT / "data/apd006_hccb_source_sequence_target_packings/seed101_s80_xlo_ycentre/summary.json",
            DEFAULT_MESH_MANIFEST,
        )
        self.assertTrue(result["all_published_geometry_and_meshing_inputs_match"])
        self.assertAlmostEqual(result["meshing_particle_diameter_m"], 0.00099)
        self.assertEqual(result["fine_local_retained_particle_fragments"], 125)
        self.assertEqual(len(result["fine_local_crop_bounds_dp"]), 6)
        self.assertEqual(len(result["fine_local_crop_lengths_dp"]), 3)
        self.assertAlmostEqual(result["fine_local_crop_lengths_dp"][0], 3.923)


if __name__ == "__main__":
    unittest.main()
