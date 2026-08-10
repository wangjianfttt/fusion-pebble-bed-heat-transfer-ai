import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from check_hccb_p418_real_physics_device_equivalence import (
    RELATIVE_LINF_TOLERANCE,
    sliced_mass_flux,
    tensor_record,
)


def test_sliced_mass_flux_preserves_static_faces_and_slices_time_series():
    static = np.arange(8, dtype=np.float32)
    transient = np.arange(40, dtype=np.float32).reshape(5, 8)

    assert sliced_mass_flux(static, 1, 4) is static
    np.testing.assert_array_equal(sliced_mass_flux(transient, 1, 4), transient[1:4])


def test_tensor_record_reports_declared_relative_tolerance():
    reference = torch.tensor([1.0, 2.0], dtype=torch.float32)
    candidate = reference * (1.0 + 0.5 * RELATIVE_LINF_TOLERANCE)

    record = tensor_record(reference, candidate)

    assert record["all_finite"]
    assert record["relative_linf_below_declared_tolerance"]
