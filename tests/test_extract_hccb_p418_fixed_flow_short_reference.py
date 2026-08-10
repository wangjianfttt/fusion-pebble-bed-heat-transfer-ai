from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/extract_hccb_p418_fixed_flow_short_reference.py"


def write_fixture(path: Path, *, complete: bool = True) -> None:
    time = np.linspace(0.0, 0.02, 2001)
    names = np.array(["outlet_temperature_K", "pressure_drop_Pa"], dtype=object)
    values = np.column_stack((700.0 + time, 20.0 + 2.0 * time))[None, :, :]
    np.savez_compressed(
        path,
        case_id=np.array(["source_up_u0p15_T700"], dtype=object),
        complete=np.array([complete]),
        time_s=time[None, :],
        time_mask=np.ones((1, time.size), dtype=bool),
        values=values,
        signal_names=names,
    )


def test_extracts_exact_0p01s_window_without_interpolation(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    output = tmp_path / "output"
    write_fixture(source)
    completed = subprocess.run(
        ["python3", str(SCRIPT), "--input", str(source), "--output-dir", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["time_point_count"] == 1001
    assert summary["end_time_s"] == 0.01
    assert summary["interpolation_used"] is False
    assert summary["openfoam_solver_started_by_this_extraction"] is False
    assert summary["new_physical_parameters"] == []
    with (output / "fixed_flow_reference_0p01s.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1001
    assert float(rows[-1]["outlet_temperature_K"]) == 700.01


def test_rejects_incomplete_reference(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    write_fixture(source, complete=False)
    completed = subprocess.run(
        ["python3", str(SCRIPT), "--input", str(source), "--output-dir", str(tmp_path / "out")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "incomplete" in completed.stderr
