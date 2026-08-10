#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from summarize_hccb_p418_formal_steady_tails import summarize  # noqa: E402


def make_case(matrix: Path, index: int, full: bool) -> None:
    case = matrix / f"case_{index}"
    case.mkdir(parents=True)
    document = {
        "status": "formal_steady_final_window_measured",
        "window_s": [175.0, 200.0],
        "engineering_changes": {
            name: {"relative_change": 0.01, "absolute_change": 0.1}
            for name in (
                "outlet_temperature_K",
                "solid_maximum_temperature_K",
                "cooling_wall_power_W",
                "outlet_enthalpy_flow_W",
                "pressure_drop_Pa",
            )
        },
        "full_field_available": full,
        "full_field_changes": (
            [
                {"field_name": "fluid_velocity_components", "relative_rms_change": 0.02},
                {"field_name": "fluid_pressure", "relative_rms_change": 0.03},
            ]
            if full
            else []
        ),
        "maximum_temperature_field_rms_change_K": 0.2 if full else None,
        "maximum_temperature_field_point_change_K": 0.3 if full else None,
    }
    document["engineering_changes"]["final_relative_mass_difference"] = 1.0e-8
    path = case / "steady.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    marker = {
        "steady_final_window_status": "formal_steady_final_window_measured",
        "steady_final_window_summary": str(path),
        "steady_final_window_summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (case / "formal_sample_complete.json").write_text(json.dumps(marker), encoding="utf-8")


def test_matrix_summary_requires_declared_full_field_coverage() -> None:
    with tempfile.TemporaryDirectory() as directory:
        matrix = Path(directory)
        for index in range(4):
            make_case(matrix, index, full=index > 0)
        rows, result = summarize(matrix, expected=4, minimum_full_fields=3)
        assert len(rows) == 4
        assert result["full_field_case_count"] == 3
        assert result["minimum_full_field_case_count"] == 3
        assert result["status"] == "formal_steady_final_windows_ready"


def test_matrix_summary_rejects_insufficient_full_fields() -> None:
    with tempfile.TemporaryDirectory() as directory:
        matrix = Path(directory)
        for index in range(4):
            make_case(matrix, index, full=index > 1)
        try:
            summarize(matrix, expected=4, minimum_full_fields=3)
        except ValueError as exc:
            assert "full-field final-window cases" in str(exc)
        else:
            raise AssertionError("insufficient full-field histories were accepted")


def test_partial_summary_reports_completed_subset_without_paper_text() -> None:
    with tempfile.TemporaryDirectory() as directory:
        matrix = Path(directory)
        for index in range(2):
            make_case(matrix, index, full=False)
        rows, result = summarize(
            matrix,
            expected=4,
            minimum_full_fields=0,
            allow_partial=True,
        )
        assert len(rows) == 2
        assert result["status"] == "partial_steady_final_windows_ready"
        assert result["case_count"] == 2
        assert result["expected_case_count"] == 4
        assert result["maximum_changes"]["temperature_field_rms_K"] is None
