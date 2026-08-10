from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/export_hccb_p418_matched_initial_short_observables.py"
DIRECT_SIGNALS = {
    "fluid/inletTemperature": 700.0,
    "fluid/outletTemperature": 670.0,
    "fluid/inletPressure": 120020.0,
    "fluid/outletPressure": 120000.0,
    "fluid/inletMassFlow": -2.0e-7,
    "fluid/outletMassFlow": 2.0e-7,
    "fluid/inletEnthalpyFlow": -0.42,
    "fluid/outletEnthalpyFlow": 0.40,
    "fluid/coolingWallPower": -0.07,
    "solid/solidTemperatureMaximum": 699.0,
    "fluid/fluidTemperatureVolumeAverage": 675.0,
    "solid/solidTemperatureVolumeAverage": 674.0,
}


def write_case(case: Path) -> None:
    for relative, base in DIRECT_SIGNALS.items():
        filename = "volFieldValue.dat" if "Temperature" in relative and "inlet" not in relative and "outlet" not in relative else "surfaceFieldValue.dat"
        path = case / "postProcessing" / relative / "0" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# Time value\n0  {0}\n0.005 {1}\n0.01 {2}\n".format(base, base + 0.1, base + 0.2),
            encoding="utf-8",
        )


def write_completion(path: Path, *, final_time: float = 0.01) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "matched_initial_direct_transport_representative_smoke_complete",
                "sequence_id": "source_up_u0p15_T700",
                "final_common_complete_time_s": final_time,
                "mpi_ranks": 32,
                "error_scan": {
                    "foam_fatal": False,
                    "mpi_abort": False,
                    "nonpositive_transport_input": False,
                    "nan": False,
                    "segmentation_fault": False,
                },
            }
        ),
        encoding="utf-8",
    )


def test_exports_compatible_single_case_matrix(tmp_path: Path) -> None:
    case = tmp_path / "case"
    completion = tmp_path / "complete.json"
    output = tmp_path / "output"
    write_case(case)
    write_completion(completion)
    completed = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--case",
            str(case),
            "--completion",
            str(completion),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["time_point_count"] == 3
    assert summary["end_time_s"] == 0.01
    assert summary["all_values_finite"] is True
    assert summary["openfoam_solver_started_by_this_export"] is False
    data = np.load(output / "hccb_p418_transient_observables.npz", allow_pickle=True)
    assert data["values"].shape == (1, 3, 15)
    assert data["complete"].tolist() == [True]
    names = data["signal_names"].tolist()
    pressure = data["values"][0, :, names.index("pressure_drop_Pa")]
    assert np.allclose(pressure, 20.0)


def test_rejects_incomplete_smoke(tmp_path: Path) -> None:
    case = tmp_path / "case"
    completion = tmp_path / "complete.json"
    write_case(case)
    write_completion(completion, final_time=0.005)
    completed = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--case",
            str(case),
            "--completion",
            str(completion),
            "--output-dir",
            str(tmp_path / "output"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "0.01 s" in completed.stderr
