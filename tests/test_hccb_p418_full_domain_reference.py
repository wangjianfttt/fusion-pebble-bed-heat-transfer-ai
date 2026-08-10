import csv
import json
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from compare_hccb_p418_full_domain_reference import compare  # noqa: E402
from build_hccb_p418_full_domain_reference_table import build_table  # noqa: E402


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_full_domain_contract_uses_only_registered_physics() -> None:
    contract = yaml.safe_load(
        (ROOT / "parameters/hccb_p418_full_domain_reference_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert contract["geometry"]["packed_region"] == "12.5dp_x_12.5dp_x_10dp"
    assert contract["geometry"]["packing_realization"]["particle_count"] == 2039
    assert contract["reference_condition"]["inlet_velocity_m_s"] == 0.20
    assert contract["reference_condition"]["solid_heat_source_MW_m3"] == 6.85
    assert contract["mesh"]["primary_level"] == "G2"
    assert contract["mesh"]["primary_target_total_cells"] == 6670000
    assert contract["new_physical_parameters"] == []


def test_full_domain_runner_waits_for_physical_data_and_mesh_check() -> None:
    text = (ROOT / "code/run_hccb_p418_full_domain_reference.sh").read_text(
        encoding="utf-8"
    )
    assert "steady_completed} -ne 60" in text
    assert "step_completed} -ne 12" in text
    assert "--mesh-level G2" in text
    assert "no physical result was run" in text
    assert "compare_hccb_p418_full_domain_reference.py" in text
    assert "completion.json" in text
    assert "reuse completed full-domain reference" in text
    assert "skip completed full-domain reference" not in text


def test_full_domain_watcher_is_idle_until_both_physical_sets_are_ready() -> None:
    text = (
        ROOT / "code/run_hccb_p418_full_domain_reference_when_ready.sh"
    ).read_text(encoding="utf-8")
    assert "formal_sample_complete.json" in text
    assert "step_response_complete.json" in text
    assert "steady_count} -eq 60" in text
    assert "step_count} -eq 12" in text
    assert "sleep \"${POLL_SECONDS}\"" in text
    assert "flock -n 9" in text
    assert "run_hccb_p418_full_domain_reference.sh" in text


def test_final_route_waits_for_full_domain_result_before_manuscript() -> None:
    route = (ROOT / "code/run_hccb_p418_formal_calculations.sh").read_text(
        encoding="utf-8"
    )
    refresh = (ROOT / "code/run_hccb_p418_manuscript_refresh.sh").read_text(
        encoding="utf-8"
    )
    manuscript = (ROOT / "manuscript/main.tex").read_text(encoding="utf-8")
    assert "run_hccb_p418_full_domain_reference.sh" in route
    assert "FULL_DOMAIN_COMPLETION" in route
    assert "FULL_DOMAIN_COMPARISON" in route
    assert "generated_full_domain_reference.tex" in route
    assert "hccb_p418_full_domain_reference/completion.json" in refresh
    assert "generated_full_domain_reference.tex" in refresh
    assert "\\input{generated_full_domain_reference}" in manuscript


def test_full_and_local_domain_comparison(tmp_path: Path) -> None:
    parameters = tmp_path / "parameters.csv"
    fields = [
        "parameter_id",
        "parameter_name",
        "material_or_system",
        "value",
        "status",
        "unit",
        "source_title",
        "source_url_or_doi",
        "evidence_type",
        "notes",
    ]
    with parameters.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for parameter_id, value in (("P048", "1"), ("P055", "635"), ("P092", "2.5")):
            writer.writerow({"parameter_id": parameter_id, "value": value})

    full = write_json(
        tmp_path / "full.json",
        {
            "status": "steady_CHT_result_computed_pending_mesh_and_seed_sensitivity",
            "observables": {
                "inlet_temperature_K": 700.0,
                "outlet_temperature_K": 680.0,
                "pressure_drop_Pa": 87.0,
                "reference_pressure_drop_Pa": 87.0,
                "pressure_drop_relative_error": 0.0,
                "maximum_solid_temperature_K": 897.0,
                "reference_maximum_temperature_K": 897.0,
                "maximum_temperature_relative_error": 0.0,
                "relative_mass_imbalance": 1.0e-7,
            },
            "conservation": {
                "generated_power_W": 5.0,
                "fluid_wall_heat_flux_integrals_W": {"coolingWall": -4.0},
                "combined_energy_residual_relative": 2.0e-5,
            },
        },
    )
    local = write_json(
        tmp_path / "local.json",
        {
            "solver_finished": True,
            "all_reported_values_are_finite": True,
            "physical_conditions": {
                "inlet_velocity_m_s": 0.20,
                "inlet_temperature_K": 700.0,
                "solid_heat_source_W_m3": 6.85e6,
            },
            "flow": {"pressure_drop_Pa": 30.363, "relative_mass_difference": 2.0e-7},
            "temperature": {"outlet_average_K": 680.0, "solid_maximum_K": 800.0},
            "heat_balance": {
                "solid_generated_power_W": 1.0,
                "cooling_wall_heat_flow_W": -0.8,
                "relative_energy_difference": 3.0e-5,
            },
        },
    )
    mesh = write_json(
        tmp_path / "mesh.json",
        {"crop_box_dp": [1.234, 5.157, 3.921, 8.163, 2.906, 6.396]},
    )

    result = compare(full, local, mesh, parameters)

    assert result["status"] == "hccb_p418_full_and_local_domain_compared"
    assert result["geometry"]["local_flow_length_dp"] == pytest.approx(3.49)
    assert result["full_domain"]["domain_average_pressure_gradient_Pa_m"] == pytest.approx(2900.0)
    assert result["local_domain"]["domain_average_pressure_gradient_Pa_m"] == pytest.approx(8700.0)
    assert result["full_domain"]["cooling_wall_heat_over_generated_power"] == pytest.approx(-0.8)
    assert result["new_physical_parameters"] == []

    table = build_table(result)
    assert "One-condition comparison" in table
    assert "Source-sized domain" in table
    assert "\\cite{wang2023pore}" in table
    assert "87" in table
    assert "897" in table
    assert "not used to tune either calculation" in table
