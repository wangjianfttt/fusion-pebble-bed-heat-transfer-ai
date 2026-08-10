from __future__ import annotations

import json
import subprocess
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/compare_hccb_p418_parallel_partitions.py"


def reference_log() -> str:
    return """
Time = 1s
surfaceFieldValue inletMassFlow write:
    sum(inlet) of phi = -2.0e-7
surfaceFieldValue outletMassFlow write:
    sum(outlet) of phi = 1.98e-7
surfaceFieldValue outletTemperature write:
    areaAverage(outlet) of T = 350
surfaceFieldValue inletPressure write:
    areaAverage(inlet) of p = 120100
surfaceFieldValue outletPressure write:
    areaAverage(outlet) of p = 120000
surfaceFieldValue coolingWallPower write:
    areaIntegrate(coolingWall) of wallHeatFlux = 0.5
volFieldValue solidTemperatureMaximum write:
    max(all) of T = 630 at location (0 0 0)
Time = 2s
"""


def candidate() -> dict:
    return {
        "solver_finished": True,
        "reported_iteration": 1.0,
        "physical_conditions": {
            "inlet_temperature_K": 300.0,
            "inlet_velocity_m_s": 0.1,
            "cooling_wall_temperature_K": 635.0,
            "solid_heat_source_W_m3": 8.85e6,
        },
        "flow": {
            "outlet_mass_flow_kg_s": 1.99e-7,
            "relative_mass_difference": 0.005,
            "pressure_drop_Pa": 100.5,
        },
        "temperature": {
            "outlet_average_K": 350.2,
            "solid_maximum_K": 630.1,
        },
        "heat_balance": {
            "cooling_wall_heat_flow_W": 0.501,
            "relative_energy_difference": 3.0,
        },
    }


def build_archive(tmp_path: Path) -> Path:
    root = tmp_path / "case"
    root.mkdir()
    (root / "log.foamMultiRun").write_text(reference_log(), encoding="utf-8")
    (root / "cht_result_summary_200.json").write_text(
        json.dumps(
            {
                "solver_finished": True,
                "physical_conditions": candidate()["physical_conditions"],
            }
        ),
        encoding="utf-8",
    )
    archive = tmp_path / "reference.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(root, arcname="case")
    return archive


def test_comparison_reads_archived_log_and_reports_partition_difference(
    tmp_path: Path,
) -> None:
    archive = build_archive(tmp_path)
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate()), encoding="utf-8")
    output = tmp_path / "output"
    subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--reference-archive",
            str(archive),
            "--reference-log-member",
            "case/log.foamMultiRun",
            "--reference-summary-member",
            "case/cht_result_summary_200.json",
            "--reference-np",
            "32",
            "--candidate-summary",
            str(candidate_path),
            "--candidate-np",
            "15",
            "--time",
            "1",
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(
        (output / "parallel_partition_comparison.json").read_text(encoding="utf-8")
    )
    assert result["same_physical_inputs"] is True
    assert result["reference"]["mpi_process_count"] == 32
    assert result["candidate"]["mpi_process_count"] == 15
    assert result["largest_direct_relative_difference"]["quantity"] == (
        "outlet_mass_flow_kg_s"
    )


def test_comparison_rejects_different_physical_conditions(tmp_path: Path) -> None:
    archive = build_archive(tmp_path)
    changed = candidate()
    changed["physical_conditions"]["inlet_velocity_m_s"] = 0.2
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(changed), encoding="utf-8")
    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--reference-archive",
            str(archive),
            "--reference-log-member",
            "case/log.foamMultiRun",
            "--reference-summary-member",
            "case/cht_result_summary_200.json",
            "--reference-np",
            "32",
            "--candidate-summary",
            str(candidate_path),
            "--candidate-np",
            "15",
            "--time",
            "1",
            "--output-dir",
            str(tmp_path / "output"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "physical conditions differ" in result.stderr
