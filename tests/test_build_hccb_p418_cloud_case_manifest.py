import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_cloud_case_manifest.py"


def test_cloud_manifest_separates_completed_and_pending(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    conditions = [
        {
            "condition_id": "u0p05_T300_q4p85",
            "inlet_velocity_m_s": 0.05,
            "inlet_temperature_K": 300.0,
            "solid_heat_source_MW_m3": 4.85,
        },
        {
            "condition_id": "u0p10_T500_q6p85",
            "inlet_velocity_m_s": 0.10,
            "inlet_temperature_K": 500.0,
            "solid_heat_source_MW_m3": 6.85,
        },
    ]
    (matrix / "matrix_manifest.json").write_text(
        json.dumps({"published_conditions": conditions}), encoding="utf-8"
    )
    for condition in conditions:
        (matrix / condition["condition_id"]).mkdir()
    (matrix / conditions[0]["condition_id"] / "formal_sample_complete.json").write_text(
        "{}", encoding="utf-8"
    )

    output = tmp_path / "output"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--matrix-root",
            str(matrix),
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(
        (output / "cloud_case_matrix_summary.json").read_text(encoding="utf-8")
    )
    assert summary["completed_on_workstation"] == 1
    assert summary["pending_for_cloud"] == 1
    assert summary["pending_condition_ids"] == ["u0p10_T500_q6p85"]
    with (output / "cloud_case_matrix.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["submit_to_cloud"] == "no"
    assert rows[1]["submit_to_cloud"] == "yes"
    assert (output / "pending_case_ids.txt").read_text(encoding="utf-8") == (
        "u0p10_T500_q6p85\n"
    )
