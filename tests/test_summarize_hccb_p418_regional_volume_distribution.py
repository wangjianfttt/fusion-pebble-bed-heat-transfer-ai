#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from summarize_hccb_p418_regional_volume_distribution import volume_summary  # noqa: E402


def test_volume_summary_keeps_finite_volume_measure() -> None:
    result = volume_summary(np.asarray([1.0, 2.0, 4.0]))
    assert result["region_count"] == 3
    assert result["maximum_over_minimum"] == 4.0
    assert result["total_volume_m3"] == 7.0


def test_volume_summary_rejects_nonphysical_volume() -> None:
    with pytest.raises(ValueError):
        volume_summary(np.asarray([1.0, 0.0]))
