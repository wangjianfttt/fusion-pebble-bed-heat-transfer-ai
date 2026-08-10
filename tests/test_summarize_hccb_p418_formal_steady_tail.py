#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from summarize_hccb_p418_formal_steady_tail import (  # noqa: E402
    relative_change,
    summarize_case,
    verified_recorded_summary,
)


HISTORIES = {
    "fluid/outletTemperature/0/surfaceFieldValue.dat": (500.0, 500.1),
    "solid/solidTemperatureMaximum/0/volFieldValue.dat": (700.0, 700.2),
    "fluid/coolingWallPower/0/surfaceFieldValue.dat": (10.0, 10.1),
    "fluid/inletMassFlow/0/surfaceFieldValue.dat": (-2.0, -2.0),
    "fluid/outletMassFlow/0/surfaceFieldValue.dat": (2.0, 2.0),
    "fluid/outletEnthalpyFlow/0/surfaceFieldValue.dat": (100.0, 101.0),
    "fluid/inletPressure/0/surfaceFieldValue.dat": (10.0, 10.1),
    "fluid/outletPressure/0/surfaceFieldValue.dat": (5.0, 5.0),
}


def write_field(path: Path, values: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "FoamFile {}\ninternalField " + values + ";\nboundaryField {}\n",
        encoding="utf-8",
    )


def make_case(root: Path, include_fields: bool = True) -> Path:
    case = root / "u0p05_T500_q4p85"
    for relative, (start, end) in HISTORIES.items():
        path = case / "postProcessing" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"175 {start}\n200 {end}\n", encoding="utf-8")
    if include_fields:
        for time, shift in (("175", 0.0), ("200", 1.0)):
            write_field(case / "processor0" / time / "fluid/T", f"uniform {500 + shift}")
            write_field(case / "processor0" / time / "solid/T", f"uniform {700 + shift}")
            write_field(
                case / "processor0" / time / "fluid/U",
                f"uniform (0 0 {0.1 + 0.01 * shift})",
            )
            write_field(case / "processor0" / time / "fluid/p", f"uniform {10 + shift}")
    return case


def test_relative_change_uses_final_value_scale() -> None:
    assert np.isclose(relative_change(9.0, 10.0), 0.1)


def test_complete_final_window_contains_scalar_and_full_field_changes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        result = summarize_case(make_case(Path(directory)), 175.0, 200.0, False)
    assert result["full_field_available"] is True
    assert np.isclose(result["maximum_temperature_field_rms_change_K"], 1.0)
    assert result["engineering_changes"]["final_relative_mass_difference"] == 0.0


def test_early_case_can_retain_scalar_history_when_partitions_were_removed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        result = summarize_case(make_case(Path(directory), include_fields=False), 175.0, 200.0, True)
    assert result["full_field_available"] is False
    assert len(result["missing_full_fields"]) == 4


def test_recorded_summary_checksum_is_verified_before_training_use() -> None:
    with tempfile.TemporaryDirectory() as directory:
        case = make_case(Path(directory))
        path = case / "steady.json"
        document = summarize_case(case, 175.0, 200.0, False)
        path.write_text(json.dumps(document), encoding="utf-8")
        marker = {
            "time": "200",
            "steady_final_window_status": "formal_steady_final_window_measured",
            "steady_final_window_summary": str(path),
            "steady_final_window_summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "steady_final_window_full_field_available": True,
        }
        verified_recorded_summary(case, marker)
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="checksum differs"):
            verified_recorded_summary(case, marker)


def test_finalize_requires_full_field_tail_before_completion_marker() -> None:
    text = (ROOT / "code" / "finalize_hccb_cht_case.sh").read_text(encoding="utf-8")
    tail_call = text.index("summarize_hccb_p418_formal_steady_tail.py")
    marker_write = text.index('case / "formal_sample_complete.json"')
    assert tail_call < marker_write
    assert 'tail.get("full_field_available") is not True' in text
