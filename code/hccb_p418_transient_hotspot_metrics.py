#!/usr/bin/env python3
"""Common solid-temperature maximum and regional-hotspot trajectory metrics."""

from __future__ import annotations

import numpy as np


def solid_transient_hotspot_metrics(
    prediction_temperature_k: np.ndarray,
    target_temperature_k: np.ndarray,
    node_type: np.ndarray,
    node_centroid_m: np.ndarray,
) -> dict[str, float]:
    """Evaluate dynamic solid maxima and hottest regional-node locations.

    The first saved time is excluded because the prescribed initial temperature
    can be spatially uniform, in which case a hotspot location is undefined.
    Location errors refer to regional-node centroids, not pebble-scale maxima.
    """

    prediction = np.asarray(prediction_temperature_k, dtype=np.float64)
    target = np.asarray(target_temperature_k, dtype=np.float64)
    material = np.asarray(node_type)
    centroid = np.asarray(node_centroid_m, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("temperature histories must share [curve,time,node] shape")
    if prediction.shape[1] < 2:
        raise ValueError("dynamic hotspot metrics require an initial and later time")
    if material.shape != (prediction.shape[2],):
        raise ValueError("node_type does not match the temperature node axis")
    if centroid.shape != (prediction.shape[2], 3) or not np.all(np.isfinite(centroid)):
        raise ValueError("node centroids must have finite [node,3] coordinates")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(target)):
        raise ValueError("temperature histories contain non-finite values")
    solid_nodes = np.flatnonzero(material == 1)
    if len(solid_nodes) == 0:
        raise ValueError("temperature histories contain no solid regional nodes")

    predicted_solid = prediction[:, 1:, solid_nodes]
    target_solid = target[:, 1:, solid_nodes]
    predicted_maximum = predicted_solid.max(axis=-1)
    target_maximum = target_solid.max(axis=-1)
    maximum_error = predicted_maximum - target_maximum

    predicted_hotspot_local = np.argmax(predicted_solid, axis=-1)
    target_hotspot_local = np.argmax(target_solid, axis=-1)
    predicted_hotspot = solid_nodes[predicted_hotspot_local]
    target_hotspot = solid_nodes[target_hotspot_local]
    displacement = np.linalg.norm(
        centroid[predicted_hotspot] - centroid[target_hotspot], axis=-1
    )
    target_at_predicted_hotspot = np.take_along_axis(
        target_solid, predicted_hotspot_local[..., None], axis=-1
    )[..., 0]
    prediction_at_target_hotspot = np.take_along_axis(
        predicted_solid, target_hotspot_local[..., None], axis=-1
    )[..., 0]
    target_temperature_deficit = np.maximum(
        target_maximum - target_at_predicted_hotspot, 0.0
    )
    prediction_temperature_deficit = np.maximum(
        predicted_maximum - prediction_at_target_hotspot, 0.0
    )

    return {
        "solid_maximum_temperature_history_RMSE_K": float(
            np.sqrt(np.mean(np.square(maximum_error)))
        ),
        "solid_maximum_temperature_history_maximum_absolute_error_K": float(
            np.max(np.abs(maximum_error))
        ),
        "solid_regional_hotspot_location_mean_error_m": float(np.mean(displacement)),
        "solid_regional_hotspot_location_p95_error_m": float(
            np.quantile(displacement, 0.95)
        ),
        "solid_regional_hotspot_location_maximum_error_m": float(
            np.max(displacement)
        ),
        "solid_regional_hotspot_exact_match_fraction": float(
            np.mean(predicted_hotspot == target_hotspot)
        ),
        "solid_hotspot_target_temperature_deficit_mean_K": float(
            np.mean(target_temperature_deficit)
        ),
        "solid_hotspot_target_temperature_deficit_p95_K": float(
            np.quantile(target_temperature_deficit, 0.95)
        ),
        "solid_hotspot_target_temperature_deficit_maximum_K": float(
            np.max(target_temperature_deficit)
        ),
        "solid_hotspot_prediction_temperature_deficit_mean_K": float(
            np.mean(prediction_temperature_deficit)
        ),
        "solid_hotspot_prediction_temperature_deficit_p95_K": float(
            np.quantile(prediction_temperature_deficit, 0.95)
        ),
        "solid_hotspot_prediction_temperature_deficit_maximum_K": float(
            np.max(prediction_temperature_deficit)
        ),
        "solid_hotspot_dynamic_sample_count": int(displacement.size),
    }
