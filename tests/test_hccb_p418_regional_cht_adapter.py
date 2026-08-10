#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from hccb_p418_regional_cht_adapter import (  # noqa: E402
    P418SubfaceGeometry,
    _volume_mean,
    p418_boundary_conditions,
)
from hccb_multiregion_steady_cht_residual import CoupledInterfaceMap, RegionMesh  # noqa: E402


def empty_mesh(cells: int, boundary_owners: list[int]) -> RegionMesh:
    return RegionMesh(
        cell_centroid=torch.zeros((cells, 3), dtype=torch.float64),
        cell_volume=torch.ones(cells, dtype=torch.float64),
        internal_face_centroid=torch.zeros((0, 3), dtype=torch.float64),
        internal_area_vector=torch.zeros((0, 3), dtype=torch.float64),
        internal_owner=torch.zeros(0, dtype=torch.long),
        internal_neighbour=torch.zeros(0, dtype=torch.long),
        boundary_face_centroid=torch.zeros((len(boundary_owners), 3), dtype=torch.float64),
        boundary_area_vector=torch.ones((len(boundary_owners), 3), dtype=torch.float64),
        boundary_owner=torch.tensor(boundary_owners, dtype=torch.long),
    )


class P418RegionalChtAdapterTest(unittest.TestCase):
    def test_volume_average_uses_cell_volume(self) -> None:
        result = _volume_mean(
            values=np.array([1.0, 3.0, 10.0]),
            volume=np.array([1.0, 3.0, 2.0]),
            regional_global=np.array([2, 2, 4]),
            selected_global=np.array([2, 4]),
        )
        self.assertTrue(np.allclose(result, [2.5, 10.0]))

    def test_boundary_conditions_match_p418_case_family(self) -> None:
        names_f = ("inlet", "outlet", "coolingWall", "symmetryWalls", "fluid_to_solid")
        names_s = ("inlet", "outlet", "coolingWall", "symmetryWalls", "solid_to_fluid")
        geometry = P418SubfaceGeometry(
            fluid_mesh=empty_mesh(2, [0, 0, 1, 1, 1]),
            solid_mesh=empty_mesh(1, [0, 0, 0, 0, 0]),
            interface=CoupledInterfaceMap(torch.tensor([4]), torch.tensor([4])),
            fluid_boundary_patch=torch.arange(5),
            solid_boundary_patch=torch.arange(5),
            fluid_patch_names=names_f,
            solid_patch_names=names_s,
            fine_to_regional_global=np.array([0, 1, 2]),
            fluid_global_region=np.array([0, 1]),
            solid_global_region=np.array([2]),
        )
        condition = torch.tensor([[0.2, 700.0, 6.85e6, 1.2e5, 635.0]])
        velocity, fluid_t, solid_t = p418_boundary_conditions(geometry, condition)
        self.assertTrue(torch.equal(velocity.fixed_value_mask, torch.tensor([True, False, True, False, True])))
        self.assertAlmostEqual(float(velocity.fixed_reference_value[0, 0, 2]), 0.2)
        self.assertTrue(torch.equal(fluid_t.fixed_value_mask, torch.tensor([True, False, True, False, False])))
        self.assertTrue(torch.equal(fluid_t.inlet_outlet_mask, torch.tensor([False, True, False, False, False])))
        self.assertTrue(torch.equal(solid_t.fixed_value_mask, torch.tensor([False, False, True, False, False])))
        self.assertTrue(bool(solid_t.coupled_temperature_mask[4]))


if __name__ == "__main__":
    unittest.main()
