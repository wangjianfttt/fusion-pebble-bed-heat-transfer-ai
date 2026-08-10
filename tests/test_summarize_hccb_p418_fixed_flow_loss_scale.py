from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "code/summarize_hccb_p418_fixed_flow_loss_scale.py"
SPEC = importlib.util.spec_from_file_location("fixed_flow_loss_scale", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summarize_row_reconstructs_declared_objective() -> None:
    row = {
        "epoch": 4,
        "temperature_data_loss": 2.0,
        "reference_edge_flux_loss": 20.0,
        "projection_aware_energy_loss": 70.0,
        "total_loss": 100.0,
        "validation_selection_score": 5.0,
        "validation_fluid_temperature_RMSE_K": 10.0,
        "validation_solid_temperature_RMSE_K": 8.0,
    }
    candidate = {
        "temperature_data_weight": 5.0,
        "reference_edge_energy_flux_weight": 1.0,
        "projection_aware_transient_energy_weight": 1.0,
    }
    result = MODULE.summarize_row(row, candidate)
    assert result["weighted_total_recomputed"] == 100.0
    assert result["relative_total_difference"] == 0.0
    assert result["objective_fraction"]["temperature_data"] == 0.1
    assert result["objective_fraction"]["projection_aware_transient_energy"] == 0.7


def test_summarize_row_rejects_non_finite_loss() -> None:
    row = {
        "epoch": 1,
        "temperature_data_loss": float("nan"),
        "reference_edge_flux_loss": 1.0,
        "projection_aware_energy_loss": 1.0,
        "total_loss": 2.0,
        "validation_selection_score": 1.0,
        "validation_fluid_temperature_RMSE_K": 1.0,
        "validation_solid_temperature_RMSE_K": 1.0,
    }
    candidate = {
        "temperature_data_weight": 1.0,
        "reference_edge_energy_flux_weight": 1.0,
        "projection_aware_transient_energy_weight": 1.0,
    }
    try:
        MODULE.summarize_row(row, candidate)
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("non-finite loss was accepted")


def test_missing_optional_fluid_validation_rmse_is_not_invented() -> None:
    row = {
        "epoch": 1,
        "temperature_data_loss": 1.0,
        "reference_edge_flux_loss": 1.0,
        "projection_aware_energy_loss": 1.0,
        "total_loss": 3.0,
        "validation_selection_score": 1.0,
        "validation_solid_temperature_RMSE_K": 4.0,
    }
    candidate = {
        "temperature_data_weight": 1.0,
        "reference_edge_energy_flux_weight": 1.0,
        "projection_aware_transient_energy_weight": 1.0,
    }
    result = MODULE.summarize_row(row, candidate)
    assert result["validation_fluid_temperature_RMSE_K"] is None
