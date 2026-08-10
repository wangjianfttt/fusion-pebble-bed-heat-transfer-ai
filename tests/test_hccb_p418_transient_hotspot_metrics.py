from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from hccb_p418_transient_hotspot_metrics import solid_transient_hotspot_metrics


def test_known_dynamic_maximum_and_hotspot_displacement() -> None:
    node_type = np.asarray([0, 1, 1], dtype=np.int8)
    centroid = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    target = np.asarray([[[300.0, 500.0, 500.0], [300.0, 510.0, 505.0], [300.0, 508.0, 515.0]]])
    prediction = np.asarray([[[300.0, 500.0, 500.0], [300.0, 509.0, 511.0], [300.0, 509.0, 514.0]]])
    metrics = solid_transient_hotspot_metrics(prediction, target, node_type, centroid)
    np.testing.assert_allclose(metrics["solid_maximum_temperature_history_RMSE_K"], 1.0)
    assert metrics["solid_maximum_temperature_history_maximum_absolute_error_K"] == 1.0
    assert metrics["solid_regional_hotspot_location_mean_error_m"] == 2.5
    assert metrics["solid_regional_hotspot_location_p95_error_m"] == 4.75
    assert metrics["solid_regional_hotspot_location_maximum_error_m"] == 5.0
    assert metrics["solid_regional_hotspot_exact_match_fraction"] == 0.5
    assert metrics["solid_hotspot_target_temperature_deficit_mean_K"] == 2.5
    assert metrics["solid_hotspot_target_temperature_deficit_p95_K"] == 4.75
    assert metrics["solid_hotspot_target_temperature_deficit_maximum_K"] == 5.0
    assert metrics["solid_hotspot_prediction_temperature_deficit_mean_K"] == 1.0
    assert metrics["solid_hotspot_prediction_temperature_deficit_p95_K"] == 1.9
    assert metrics["solid_hotspot_prediction_temperature_deficit_maximum_K"] == 2.0
    assert metrics["solid_hotspot_dynamic_sample_count"] == 2


def test_rejects_missing_dynamic_time_or_solid_nodes() -> None:
    with pytest.raises(ValueError, match="initial and later"):
        solid_transient_hotspot_metrics(
            np.zeros((1, 1, 2)), np.zeros((1, 1, 2)), np.asarray([0, 1]), np.zeros((2, 3))
        )
    with pytest.raises(ValueError, match="no solid"):
        solid_transient_hotspot_metrics(
            np.zeros((1, 2, 2)), np.zeros((1, 2, 2)), np.asarray([0, 0]), np.zeros((2, 3))
        )
