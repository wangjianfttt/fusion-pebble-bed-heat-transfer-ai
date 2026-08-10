#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_p418_regional_energy_flux_targets import (  # noqa: E402
    INTERFACE_L1_RELATIVE_TOLERANCE,
    INTERFACE_MAX_RELATIVE_TOLERANCE,
    aggregate_crossing_flux,
    conservative_internal_projection,
    interface_reciprocity_metrics,
    regional_balance,
)


class RegionalEnergyFluxTargetsTest(unittest.TestCase):
    def test_interface_consistency_limits_remain_sub_per_mille(self) -> None:
        self.assertLessEqual(INTERFACE_MAX_RELATIVE_TOLERANCE, 1.0e-6)
        self.assertLessEqual(INTERFACE_L1_RELATIVE_TOLERANCE, 2.0e-5)

    def test_interface_reciprocity_metrics_are_scale_invariant(self) -> None:
        fluid = np.array([1.0e-3, 2.0e-3, -5.0e-4])
        solid = fluid + np.array([2.0e-10, -3.0e-10, 1.0e-10])
        reference = interface_reciprocity_metrics(fluid, solid)
        scaled = interface_reciprocity_metrics(100.0 * fluid, 100.0 * solid)
        self.assertAlmostEqual(
            reference["maximum_difference_over_maximum_interface_face_power"],
            scaled["maximum_difference_over_maximum_interface_face_power"],
        )
        self.assertAlmostEqual(
            reference["l1_difference_over_l1_interface_face_power"],
            scaled["l1_difference_over_l1_interface_face_power"],
        )

    def test_interface_reciprocity_metrics_keep_absolute_difference(self) -> None:
        result = interface_reciprocity_metrics(
            np.array([1.0, 2.0]), np.array([1.0 + 4.0e-10, 2.0])
        )
        self.assertAlmostEqual(result["maximum_interface_pair_difference_W"], 4.0e-10)

    def test_crossing_flux_orientation_is_preserved(self) -> None:
        key, flux = aggregate_crossing_flux(
            fine_owner_global=np.array([0, 1, 2]),
            fine_neighbour_global=np.array([1, 2, 3]),
            fine_flux_owner_to_neighbour=np.array([2.0, 3.0, -4.0]),
            fine_to_region=np.array([1, 0, 2, 1]),
            region_count=3,
        )
        result = dict(zip(key.tolist(), flux.tolist()))
        self.assertAlmostEqual(result[1], -2.0)
        self.assertAlmostEqual(result[2], 3.0)
        self.assertAlmostEqual(result[5], 4.0)

    def test_projection_recovers_local_balance_without_changing_global_remainder(self) -> None:
        owner = np.array([0, 1], dtype=np.int64)
        neighbour = np.array([1, 2], dtype=np.int64)
        boundary_owner = np.array([0, 2], dtype=np.int64)
        boundary = np.array([-1.0, 1.2])
        source = np.array([0.0, 0.2, 0.0])
        initial = np.array([0.7, 0.5])
        projected, metrics = conservative_internal_projection(
            initial_internal_flux=initial,
            owner=owner,
            neighbour=neighbour,
            conductance=np.ones(2),
            boundary_flux=boundary,
            boundary_owner=boundary_owner,
            source=source,
            cell_volume=np.ones(3),
        )
        final = regional_balance(
            projected, owner, neighbour, boundary, boundary_owner, source
        )
        np.testing.assert_allclose(final, np.full(3, final.sum() / 3), atol=1.0e-11)
        self.assertLess(
            metrics["final_local_balance_l1_W"],
            metrics["initial_local_balance_l1_W"],
        )


if __name__ == "__main__":
    unittest.main()
