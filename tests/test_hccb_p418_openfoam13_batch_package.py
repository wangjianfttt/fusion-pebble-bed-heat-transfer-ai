from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "code/build_hccb_p418_openfoam13_batch_inputs.py"
SHELL_BUILDER = ROOT / "code/build_hccb_p418_openfoam13_batch_package.sh"


def make_case(
    matrix: Path,
    condition_id: str,
    velocity: float,
    temperature: float,
    source: float,
) -> None:
    case = matrix / condition_id
    files = {
        "0/fluid/U": f"velocity {velocity}",
        "0/fluid/T": f"temperature {temperature}",
        "0/fluid/p": "pressure",
        "0/fluid/p_rgh": "pressure_rgh",
        "0/solid/T": f"solid temperature {temperature}",
        "constant/fluid/physicalProperties": "helium",
        "constant/solid/physicalProperties": f"solid at {temperature}",
        "constant/solid/fvModels": f"source {source}",
        "constant/fluid/polyMesh/points": "shared fluid mesh",
        "constant/solid/polyMesh/points": "shared solid mesh",
        "system/controlDict": "control",
        "system/decomposeParDict": "decompose",
    }
    for relative, content in files.items():
        path = case / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    metadata = {
        "operating_condition_id": condition_id,
        "inlet_velocity_m_s": velocity,
        "inlet_temperature_K": temperature,
        "solid_heat_source_W_m3": source * 1.0e6,
        "source_channel_volume_flow_preserved": True,
        "new_fitted_physical_parameters": [],
        "pore_opening_boundary_velocity_m_s": velocity / 0.4,
        "inlet_open_area_fraction": 0.4,
    }
    (case / "cht_smoke_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def make_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    conditions = [
        ("u0p10_T500_q4p85", 0.10, 500.0, 4.85),
        ("u0p20_T900_q8p85", 0.20, 900.0, 8.85),
    ]
    for condition in conditions:
        make_case(matrix, *condition)

    cloud_table = tmp_path / "cloud_case_matrix.csv"
    with cloud_table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "order",
                "condition_id",
                "inlet_velocity_m_s",
                "inlet_temperature_K",
                "solid_heat_source_MW_m3",
                "workstation_status",
                "submit_to_cloud",
            ),
        )
        writer.writeheader()
        for order, (condition_id, velocity, temperature, source) in enumerate(
            conditions, start=1
        ):
            writer.writerow(
                {
                    "order": order,
                    "condition_id": condition_id,
                    "inlet_velocity_m_s": velocity,
                    "inlet_temperature_K": temperature,
                    "solid_heat_source_MW_m3": source,
                    "workstation_status": "pending_cloud",
                    "submit_to_cloud": "yes",
                }
            )

    input_check = tmp_path / "input_check.json"
    input_check.write_text(
        json.dumps(
            {
                "status": "hccb_p418_60_actual_case_inputs_verified",
                "all_openfoam_dictionary_values_match_registered_sources": True,
                "cases": [
                    {
                        "condition_id": condition_id,
                        "inlet_velocity_m_s": velocity,
                        "inlet_temperature_K": temperature,
                        "solid_heat_source_MW_m3": source,
                    }
                    for condition_id, velocity, temperature, source in conditions
                ],
            }
        ),
        encoding="utf-8",
    )
    return matrix, cloud_table, input_check


def test_batch_inputs_keep_each_condition_and_one_shared_mesh(tmp_path: Path) -> None:
    matrix, cloud_table, input_check = make_inputs(tmp_path)
    output = tmp_path / "batch"
    subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--matrix-root",
            str(matrix),
            "--cloud-table",
            str(cloud_table),
            "--input-check",
            str(input_check),
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads(
        (output / "batch_input_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["case_count"] == 2
    assert manifest["solver_started"] is False
    assert manifest["new_physical_parameters"] == []
    first = output / "case_inputs/u0p10_T500_q4p85"
    second = output / "case_inputs/u0p20_T900_q8p85"
    assert "500.0" in (first / "0/fluid/T").read_text(encoding="utf-8")
    assert "900.0" in (second / "0/fluid/T").read_text(encoding="utf-8")
    assert "4.85" in (first / "constant/solid/fvModels").read_text(
        encoding="utf-8"
    )
    assert "8.85" in (second / "constant/solid/fvModels").read_text(
        encoding="utf-8"
    )
    assert not list((output / "case_inputs").rglob("polyMesh"))
    assert (output / "shared_mesh/fluid/polyMesh/points").is_file()
    assert (output / "shared_mesh/solid/polyMesh/points").is_file()
    assert (first / "INPUT_SHA256SUMS").is_file()
    assert (first / "cht_smoke_metadata.json").is_file()


def test_batch_inputs_reject_condition_metadata_mismatch(tmp_path: Path) -> None:
    matrix, cloud_table, input_check = make_inputs(tmp_path)
    metadata_path = matrix / "u0p10_T500_q4p85/cht_smoke_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["inlet_temperature_K"] = 900.0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--matrix-root",
            str(matrix),
            "--cloud-table",
            str(cloud_table),
            "--input-check",
            str(input_check),
            "--output-dir",
            str(tmp_path / "batch"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "u0p10_T500_q4p85" in result.stderr


def test_batch_shell_scripts_have_valid_syntax() -> None:
    for script in (
        SHELL_BUILDER,
        ROOT / "cloud_migration/run_openfoam13_batch_case.sh",
        ROOT / "cloud_migration/submit_slurm_array_example.sh",
        ROOT / "cloud_migration/postprocess_openfoam13_case.sh",
    ):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_batch_runner_checks_input_files_before_solver() -> None:
    text = (ROOT / "cloud_migration/run_openfoam13_batch_case.sh").read_text(
        encoding="utf-8"
    )
    assert "sha256sum -c INPUT_SHA256SUMS" in text
    assert "sha256sum -c SHA256SUMS" in text
    assert "pending_case_ids.txt" in text
    assert "run_openfoam13_case.sh" in text
