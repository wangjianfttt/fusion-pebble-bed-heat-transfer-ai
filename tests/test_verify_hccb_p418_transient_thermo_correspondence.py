#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from verify_hccb_p418_transient_thermo_correspondence import (  # noqa: E402
    build_summary,
    piecewise_linear_integral,
)


class P418TransientThermoCorrespondenceTest(unittest.TestCase):
    def test_piecewise_linear_integral_is_exact_for_linear_table(self) -> None:
        nodes = np.asarray([0.0, 2.0, 5.0])
        values = 3.0 * nodes + 4.0
        query = np.asarray([0.0, 1.0, 2.0, 3.0, 5.0])
        reference = 1.0
        expected = 1.5 * (query**2 - reference**2) + 4.0 * (query - reference)
        np.testing.assert_allclose(
            piecewise_linear_integral(nodes, values, query, reference=reference),
            expected,
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_registered_openfoam_and_python_thermo_correspond(self) -> None:
        payload = build_summary()
        self.assertTrue(payload["status"].startswith("passed"))
        self.assertTrue(all(payload["checks"].values()))
        self.assertEqual(payload["new_fitted_physical_parameters"], [])


if __name__ == "__main__":
    unittest.main()
