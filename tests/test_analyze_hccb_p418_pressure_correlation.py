import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/analyze_hccb_p418_pressure_correlation.py"


def test_pressure_correlation_uses_superficial_not_pore_velocity(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix"
    case = matrix / "u0p05_T300_q4p85"
    sample_dir = case / "training_sample_200_schema3"
    sample_dir.mkdir(parents=True)
    sample = sample_dir / "fields_and_topology.npz"
    rho = 480.19 * 0.12 / 300.0
    mu = 0.4646 * 300.0**0.66 * 1.0e-6
    porosity = 0.4
    length = 0.003
    source_channel_velocity = 0.05
    pore_opening_velocity = source_channel_velocity / porosity
    superficial_velocity = source_channel_velocity
    pressure_drop = (
        180.0
        * (1.0 - porosity) ** 2
        / porosity**3
        * mu
        / 0.001**2
        * superficial_velocity
        * length
    )
    np.savez_compressed(
        sample,
        fluid_boundary_face_patch=np.asarray([0, 1]),
        solid_boundary_face_patch=np.asarray([0]),
        fluid_boundary_face_area_m2=np.asarray([0.4, 0.4]),
        solid_boundary_face_area_m2=np.asarray([0.6]),
        fluid_boundary_temperature_K=np.asarray([300.0, 300.0]),
        fluid_boundary_pressure_Pa=np.asarray([120000.0 + pressure_drop, 120000.0]),
        fluid_boundary_density_kg_m3=np.asarray([rho, rho]),
        fluid_boundary_face_mass_flow_kg_s=np.asarray(
            [
                -rho * pore_opening_velocity * 0.4,
                rho * pore_opening_velocity * 0.4,
            ]
        ),
        fluid_boundary_face_centroid_m=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, length]]),
        fluid_cell_volume_m3=np.asarray([0.4]),
        solid_cell_volume_m3=np.asarray([0.6]),
    )
    (sample_dir / "metadata.json").write_text(
        json.dumps(
            {
                "fluid_patch_names": ["inlet", "outlet"],
                "solid_patch_names": ["inlet"],
                "physical_conditions": {
                    "inlet_velocity_m_s": source_channel_velocity,
                    "pore_opening_boundary_velocity_m_s": pore_opening_velocity,
                },
            }
        ),
        encoding="utf-8",
    )
    (case / "formal_sample_complete.json").write_text(
        json.dumps(
            {
                "condition_id": "u0p05_T300_q4p85",
                "training_sample": str(sample),
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.csv"
    rows = [
        {"parameter_id": "P048", "value": "1", "status": "extracted", "notes": ""},
        {
            "parameter_id": "P420",
            "value": "rho^-9.181*mu^-12.238*rin^-5.320*muin^-8.062",
            "status": "extracted",
            "notes": "reference state is T=300 K and p=0.12 MPa",
        },
        {
            "parameter_id": "P421",
            "value": "180*xi_f*mu_IN/d_p^2*u_IN",
            "status": "extracted",
            "notes": "",
        },
        {"parameter_id": "P422", "value": "4.6", "status": "extracted", "notes": ""},
        {"parameter_id": "P426", "value": "0.12", "status": "extracted", "notes": ""},
    ]
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    physical = tmp_path / "physical.csv"
    with physical.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["condition_id", "pressure_drop_Pa"])
        writer.writeheader()
        writer.writerow(
            {"condition_id": "u0p05_T300_q4p85", "pressure_drop_Pa": pressure_drop}
        )
    output = tmp_path / "output"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--matrix-root",
            str(matrix),
            "--parameter-manifest",
            str(manifest),
            "--physical-csv",
            str(physical),
            "--output-dir",
            str(output),
            "--expected-case-count",
            "1",
        ],
        check=True,
    )
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert result["maximum_absolute_difference_percent"] < 1.0e-10
    assert result["superficial_to_pore_velocity_ratio_range"] == [0.4, 0.4]
    assert result["maximum_superficial_vs_source_channel_velocity_difference_fraction"] < 1.0e-12
    assert result["new_physical_parameters"] == []
