import csv
import json
from pathlib import Path

import numpy as np
import pytest

from analyze_hccb_p418_velocity_step_time_scales import (
    VALUE_COLUMN,
    analyze,
    cell_outflow_turnover_rate,
)


def test_cell_outflow_turnover_rate_uses_face_orientation():
    rate = cell_outflow_turnover_rate(
        np.array([1.0, 2.0]),
        np.array([2.0, 2.0]),
        np.array([0]),
        np.array([1]),
        np.array([4.0]),
        np.array([1]),
        np.array([2.0]),
    )
    np.testing.assert_allclose(rate, [2.0, 0.5])


def test_analyze_separates_flow_and_thermal_time_scales(tmp_path: Path):
    topology_path = tmp_path / "topology.npz"
    field_path = tmp_path / "field.npz"
    source_path = tmp_path / "parameters.csv"
    particle_path = tmp_path / "particle.json"
    input_summary_path = tmp_path / "formal_inputs.json"
    np.savez(
        topology_path,
        fluid_cell_centroid_m=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]]),
        fluid_cell_volume_m3=np.array([1.0, 1.0]),
        fluid_internal_face_owner=np.array([0]),
        fluid_internal_face_neighbour=np.array([1]),
        fluid_boundary_face_owner=np.array([1]),
    )
    np.savez(
        field_path,
        fluid_density_kg_m3=np.array([1.0, 1.0]),
        fluid_internal_face_mass_flow_kg_s=np.array([2.0]),
        fluid_boundary_face_mass_flow_kg_s=np.array([2.0]),
    )
    with source_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["parameter_id", VALUE_COLUMN])
        writer.writeheader()
        writer.writerow(
            {
                "parameter_id": "P048",
                VALUE_COLUMN: "1",
            }
        )
        writer.writerow(
            {
                "parameter_id": "P427",
                VALUE_COLUMN: (
                    "bed=12.5dp x 12.5dp x 10dp;inlet_extension=10dp;"
                    "outlet_extension=10dp"
                ),
            }
        )
        writer.writerow(
            {
                "parameter_id": "P418",
                VALUE_COLUMN: "u_in=0.05,0.25 m/s x T_in=300 K",
            }
        )
    particle_path.write_text(
        json.dumps(
            {
                "parameter_ids": ["P048", "P092"],
                "values": [
                    {"particle_radial_diffusion_scale_s": 0.4},
                    {"particle_radial_diffusion_scale_s": 0.6},
                ],
            }
        ),
        encoding="utf-8",
    )
    input_summary_path.write_text(
        json.dumps(
            {
                "status": "hccb_p418_60_actual_case_inputs_verified",
                "cases": [
                    {
                        "inlet_velocity_m_s": velocity,
                        "pore_opening_boundary_velocity_m_s": velocity / 0.4,
                        "inlet_open_area_fraction": 0.4,
                        "source_channel_volume_flow_preserved": True,
                    }
                    for velocity in (0.05, 0.25)
                ],
            }
        ),
        encoding="utf-8",
    )
    result = analyze(
        topology_path,
        field_path,
        source_path,
        particle_path,
        2,
        1.0,
        input_summary_path,
    )
    assert result["nominal_domain_crossing_times_s"] == pytest.approx([0.08, 0.016])
    assert result["pore_opening_boundary_velocities_m_s"] == pytest.approx(
        [0.125, 0.625]
    )
    assert result["inlet_open_area_fraction"] == pytest.approx(0.4)
    assert result["velocity_basis"] == (
        "source_channel_area_preserving_pore_boundary_velocity"
    )
    assert result["published_10dp_bed_crossing_times_s"] == pytest.approx([0.2, 0.04])
    assert result["published_30dp_full_domain_crossing_times_s"] == pytest.approx(
        [0.6, 0.12]
    )
    assert result["time_scale_ratios"][
        "particle_conduction_to_published_10dp_bed_crossing_global_min"
    ] == pytest.approx(2.0)
    assert result["time_scale_ratios"][
        "particle_conduction_to_published_10dp_bed_crossing_global_max"
    ] == pytest.approx(15.0)
    assert "temperature-to-flow feedback" in " ".join(
        result["fixed_flow_scope"]["not_represented"]
    )
    assert result["new_physical_parameters"] == []
    assert result["implied_p95_Courant_if_flow_were_active"] == 2.0
