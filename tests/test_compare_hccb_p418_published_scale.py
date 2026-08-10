import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from compare_hccb_p418_published_scale import compare  # noqa: E402


def test_different_geometry_is_not_direct_validation(tmp_path: Path) -> None:
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
    rows = [
        {"parameter_id": "P048", "value": "1"},
        {"parameter_id": "P053", "value": "6.85"},
        {"parameter_id": "P390", "value": "12.5dp x 12.5dp x 10dp; inlet channel=10dp; outlet extension=10dp"},
        {"parameter_id": "P391", "value": "u_in=0.20;T_in=700;deltaP=87;Tmax=897", "source_title": "source", "source_url_or_doi": "doi"},
    ]
    with parameters.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "physical_conditions": {
                    "inlet_velocity_m_s": 0.2,
                    "inlet_temperature_K": 700.0,
                    "solid_heat_source_W_m3": 6.85e6,
                },
                "flow": {"pressure_drop_Pa": 11.2},
                "temperature": {"solid_maximum_K": 700.0},
            }
        ),
        encoding="utf-8",
    )
    sample = tmp_path / "sample.npz"
    np.savez_compressed(
        sample,
        fluid_cell_centroid_m=np.array([[0.0, 0.0, 0.001], [0.004, 0.004, 0.003]]),
        solid_cell_centroid_m=np.array([[0.001, 0.001, 0.002]]),
        fluid_boundary_face_centroid_m=np.array([[0.002, 0.002, 0.0], [0.002, 0.002, 0.0035]]),
        fluid_boundary_face_patch=np.array([0, 1]),
    )

    result = compare(parameters, summary, sample)

    assert all(result["operating_conditions_match"].values())
    assert result["status"] == "same_operating_point_different_geometry"
    assert result["direct_pressure_or_temperature_error_is_valid"] is False
    assert np.isclose(result["local_crop"]["flow_length_dp"], 3.5)
