#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_formal_step_runner_uses_no_invented_temperature_sensor_mask() -> None:
    runner = (ROOT / "code/run_hccb_p418_step_responses.sh").read_text(
        encoding="utf-8"
    )
    assert "--run-role computed_residual_benchmark" in runner
    assert "--observation-mask" not in runner
    assert "--observation-source" not in runner


def test_sparse_reconstruction_remains_unavailable_without_source_data() -> None:
    contract = (ROOT / "parameters/apd006_tdem_diffusion_route_contract.yaml").read_text(
        encoding="utf-8"
    )
    assert "status: diffusion_training_locked_until_sparse_observation_contract_is_complete" in contract
    for missing_source in (
        "no_machine_readable_TESOMEX_3d_sensor_coordinate_table",
        "no_source_backed_thermocouple_noise_and_dynamic_response_model",
        "no_repeated_experimental_transient_ensemble_for_posterior_calibration",
        "no_calibrated_observation_operator_between_TESOMEX_sensors_and_particle_nodes",
    ):
        assert missing_source in contract
    assert "computed_sparse_mask_contract:" in contract
    assert "source_kind: computed_openfoam_target" in contract
    assert "external_experiment_exact_conditioning_allowed: false" in contract
