from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch runs remotely")
def test_temperature_formats_and_energy_components() -> None:
    import torch

    from evaluate_hccb_p418_temporal_energy_balance import (
        physical_state,
        prediction_files,
        residual_components,
        residual_field_components,
        registered_temperature_range_diagnostics,
        summarize_endpoint_energy_groups,
        temperature_fields,
    )

    node_type = np.asarray([0, 1], dtype=np.int64)
    normalized = np.asarray([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32)
    target = np.zeros_like(normalized)
    prediction, reference = temperature_fields(
        {
            "baseline_temperature_normalized": normalized,
            "target_temperature_normalized": target,
            "node_type": node_type,
            "temperature_mean_K_by_node_type": np.asarray([300.0, 600.0]),
            "temperature_std_K_by_node_type": np.asarray([10.0, 20.0]),
        }
    )
    np.testing.assert_allclose(prediction[0], [[310.0, 640.0], [330.0, 680.0]])
    np.testing.assert_allclose(reference[0], [[300.0, 600.0], [300.0, 600.0]])
    state = physical_state(prediction[0], np.ones((2, 4), dtype=np.float32))
    assert state.shape == (2, 2, 5)
    np.testing.assert_allclose(state[..., 4], prediction[0])

    class Residual:
        fluid_energy_w_m3 = torch.tensor([[[2.0, 4.0]]])
        solid_energy_w_m3 = torch.tensor([[[6.0]]])

    components = residual_components(
        Residual(),
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 2.0e-6, 0.0, 0.0]]),
        torch.tensor([1.0, 3.0]),
        torch.tensor([2.0]),
    )
    assert components["fluid_mse"] == pytest.approx(2.5)
    assert components["solid_mse"] == pytest.approx(9.0)
    assert components["combined_mse"] == pytest.approx(5.75)
    assert components["fluid_volume_weighted_mse"] == pytest.approx(3.25)
    assert components["solid_volume_weighted_mse"] == pytest.approx(9.0)
    assert components["volume_weighted_mse"] == pytest.approx(31.0 / 6.0)
    assert components["global_closure_mse"] == pytest.approx(42.25)
    assert components["local_l1_mse"] == pytest.approx(42.25)
    projection_difference = residual_field_components(
        Residual.fluid_energy_w_m3 - Residual.fluid_energy_w_m3,
        Residual.solid_energy_w_m3 - Residual.solid_energy_w_m3,
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 2.0e-6, 0.0, 0.0]]),
        torch.tensor([1.0, 3.0]),
        torch.tensor([2.0]),
    )
    assert projection_difference["combined_mse"] == 0.0
    assert projection_difference["global_closure_mse"] == 0.0
    assert prediction_files(
        {"prediction_files": {role: f"{role}.npz" for role in ("train", "validation", "test")}}
    )["test"] == "test.npz"
    assert prediction_files(
        {"prediction_files": {"test": "chained-test.npz"}}, ("test",)
    ) == {"test": "chained-test.npz"}

    diagnostics = registered_temperature_range_diagnostics(
        {
            "temperature_prediction_K": np.asarray(
                [[[300.0, 700.0], [301.0, 1301.0]]], dtype=np.float32
            ),
            "temperature_target_K": np.asarray(
                [[[300.0, 700.0], [301.0, 900.0]]], dtype=np.float32
            ),
            "node_type": node_type,
        }
    )
    assert diagnostics["prediction_within_registered_thermophysical_range"] is False
    assert diagnostics["reference_within_registered_thermophysical_range"] is True
    assert diagnostics["prediction_solid_out_of_range_value_count"] == 1
    assert diagnostics["prediction_solid_temperature_max_K"] == pytest.approx(1301.0)
    grouped = summarize_endpoint_energy_groups(
        [
            {
                "endpoint_novelty_class": "both_steady_endpoints_unseen",
                "projection_aware_volume_weighted_energy_RMSE": 3.0,
            },
            {
                "endpoint_novelty_class": "both_steady_endpoints_unseen",
                "projection_aware_volume_weighted_energy_RMSE": 4.0,
            },
        ]
    )["both_steady_endpoints_unseen"]
    assert grouped["curve_count"] == 2
    assert grouped[
        "projection_aware_volume_weighted_energy_equation_normalized_RMSE"
    ] == pytest.approx(np.sqrt(12.5))
