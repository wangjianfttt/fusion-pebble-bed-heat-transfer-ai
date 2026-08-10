#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from verify_hccb_p418_step_initialization import internal_field_values  # noqa: E402


class StepInitializationFieldReaderTests(unittest.TestCase):
    def write_field(self, body: str) -> Path:
        self.temporary = tempfile.TemporaryDirectory()
        path = Path(self.temporary.name) / "field"
        path.write_text(
            "FoamFile {}\ninternalField " + body + ";\nboundaryField {}\n",
            encoding="utf-8",
        )
        return path

    def tearDown(self) -> None:
        temporary = getattr(self, "temporary", None)
        if temporary is not None:
            temporary.cleanup()

    def test_nonuniform_scalar_count_is_not_read_as_temperature(self) -> None:
        path = self.write_field("nonuniform List<scalar> 3\n(300 450 900)")
        np.testing.assert_allclose(internal_field_values(path), [300.0, 450.0, 900.0])

    def test_nonuniform_vector_count_is_not_read_as_field_value(self) -> None:
        path = self.write_field("nonuniform List<vector> 2\n((0 0 0.05) (0 0 0.10))")
        np.testing.assert_allclose(
            internal_field_values(path), [0.0, 0.0, 0.05, 0.0, 0.0, 0.10]
        )

    def test_uniform_scalar_is_supported(self) -> None:
        path = self.write_field("uniform 635")
        np.testing.assert_allclose(internal_field_values(path), [635.0])


if __name__ == "__main__":
    unittest.main()
