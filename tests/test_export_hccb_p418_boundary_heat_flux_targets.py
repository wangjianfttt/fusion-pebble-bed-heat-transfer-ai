#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from export_hccb_p418_boundary_heat_flux_targets import field_boundary_array  # noqa: E402


class BoundaryHeatFluxTargetTest(unittest.TestCase):
    def test_ascii_field_is_concatenated_in_declared_patch_order(self) -> None:
        fixture = ROOT / "tests/fixtures/openfoam_scalar_boundary_field"
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(
            """FoamFile { format ascii; class volScalarField; object wallHeatFlux; }
dimensions [1 0 -3 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet { type calculated; value nonuniform List<scalar> 2 (1 2); }
    outlet { type calculated; value uniform -3; }
    symmetryWalls { type symmetry; }
}
""",
            encoding="utf-8",
        )
        try:
            values = field_boundary_array(
                fixture,
                cell_count=3,
                patch_names=["inlet", "outlet", "symmetryWalls"],
                patch_sizes={"inlet": 2, "outlet": 1, "symmetryWalls": 2},
            )
            np.testing.assert_allclose(values, [1.0, 2.0, -3.0, 0.0, 0.0])
        finally:
            fixture.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
