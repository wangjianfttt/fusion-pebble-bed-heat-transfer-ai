from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
SPEC = importlib.util.spec_from_file_location(
    "train_hccb_p418_regional_operator",
    ROOT / "code" / "train_hccb_p418_regional_operator.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_volume_weighted_rmse_uses_cell_volume():
    predicted = np.array([1.0, 3.0])
    reference = np.array([1.0, 1.0])
    volume = np.array([3.0, 1.0])
    assert MODULE.volume_weighted_rmse(predicted, reference, volume) == pytest.approx(1.0)


def test_volume_weighted_rmse_rejects_nonphysical_volume():
    with pytest.raises(ValueError, match="positive"):
        MODULE.volume_weighted_rmse(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array([1.0, 0.0]),
        )


def test_cross_packing_output_contains_field_and_hotspot_metrics():
    text = (ROOT / "code" / "train_hccb_p418_regional_operator.py").read_text(
        encoding="utf-8"
    )
    assert "fluid_temperature_volume_weighted_rmse_K" in text
    assert "solid_temperature_volume_weighted_rmse_K" in text
    assert "solid_hotspot_location_error_m" in text
    assert "Not inferred from state values alone" in text
