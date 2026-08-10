from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/verify_hccb_p418_mesh_fine_reference.py"


def write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    metadata = tmp_path / "metadata.json"
    completion = tmp_path / "completion.json"
    result = tmp_path / "result.json"
    manifest = tmp_path / "manifest.json"
    metadata.write_text(
        json.dumps(
            {
                "operating_condition_id": "u0p20_T700_q6p85",
                "mesh_resolution_label": "fine",
                "mesh_source_packing_sha256": "packing-a",
                "source_channel_volume_flow_preserved": True,
                "source_inlet_channel_velocity_m_s": 0.2,
                "pore_opening_boundary_velocity_m_s": 0.5,
                "inlet_open_area_fraction": 0.4,
                "inlet_temperature_K": 700.0,
                "solid_heat_source_W_m3": 6.85e6,
                "end_time": 200,
            }
        ),
        encoding="utf-8",
    )
    completion.write_text(
        json.dumps(
            {
                "condition_id": "u0p20_T700_q6p85",
                "time": "200",
                "solver_finished": True,
            }
        ),
        encoding="utf-8",
    )
    result.write_text(
        json.dumps(
            {
                "physical_conditions": {
                    "inlet_velocity_m_s": 0.2,
                    "inlet_temperature_K": 700.0,
                    "solid_heat_source_W_m3": 6.85e6,
                }
            }
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps({"source_packing_sha256": "packing-a"}), encoding="utf-8"
    )
    return metadata, completion, result, manifest


def command(paths: tuple[Path, Path, Path, Path]) -> list[str]:
    metadata, completion, result, manifest = paths
    return [
        sys.executable,
        str(SCRIPT),
        "--metadata",
        str(metadata),
        "--completion",
        str(completion),
        "--result",
        str(result),
        "--mesh-manifest",
        str(manifest),
        "--condition-id",
        "u0p20_T700_q6p85",
        "--end-time",
        "200",
    ]


def test_verified_fine_reference_reports_all_checks(tmp_path: Path) -> None:
    completed = subprocess.run(
        command(write_inputs(tmp_path)), check=True, capture_output=True, text=True
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "verified_p418_mesh_fine_reference"
    assert all(payload["checks"].values())
    assert payload["new_physical_parameters"] == []


def test_wrong_packing_is_rejected(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    paths[-1].write_text(
        json.dumps({"source_packing_sha256": "packing-b"}), encoding="utf-8"
    )
    completed = subprocess.run(command(paths), capture_output=True, text=True)
    assert completed.returncode != 0
    assert "packing" in completed.stderr


def test_result_condition_mismatch_is_rejected(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    paths[2].write_text(
        json.dumps(
            {
                "physical_conditions": {
                    "inlet_velocity_m_s": 0.1,
                    "inlet_temperature_K": 700.0,
                    "solid_heat_source_W_m3": 6.85e6,
                }
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(command(paths), capture_output=True, text=True)
    assert completed.returncode != 0
    assert "result_velocity" in completed.stderr
