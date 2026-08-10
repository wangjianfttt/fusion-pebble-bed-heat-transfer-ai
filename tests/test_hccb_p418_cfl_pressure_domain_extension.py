from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "code/build_hccb_openfoam_helium_property_table.py"
VERIFY = ROOT / "code/verify_hccb_p418_pressure_domain_extension.py"
CANONICAL = (
    ROOT
    / "results/apd006_hccb_openfoam_helium_property_table/helium_property_table.npz"
)


def test_cfl_pressure_extension_keeps_correlations_and_covers_both_excursions(
    tmp_path: Path,
) -> None:
    output = tmp_path / "table"
    build = subprocess.run(
        [
            sys.executable,
            str(BUILD),
            "--output-dir",
            str(output),
            "--pressure-guard-multiplier",
            "300",
            "--pressure-support-design-id",
            "ND079",
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["pressure_support_design_id"] == "ND079"
    assert "ND079" in summary["numerical_design_ids"]
    assert summary["pressure_nodes_pa"] == [93900.0, 120000.0, 146100.0]
    assert max(summary["maximum_relative_interpolation_errors"].values()) < 1.0e-4

    check_path = tmp_path / "domain_extension_check.json"
    verify = subprocess.run(
        [
            sys.executable,
            str(VERIFY),
            "--canonical-table",
            str(CANONICAL),
            "--extended-table",
            str(output / "helium_property_table.npz"),
            "--output",
            str(check_path),
            "--failed-job-id",
            "14718,14720",
            "--failed-case-id",
            "maxCo_0p8,maxCo_0p2",
            "--observed-pressure-min-pa",
            "97736.93757",
            "--observed-pressure-max-pa",
            "141956.6815",
        ],
        capture_output=True,
        text=True,
    )
    assert verify.returncode == 0, verify.stderr
    check = json.loads(check_path.read_text(encoding="utf-8"))
    assert check["status"] == "hccb_p418_pressure_domain_extension_passed"
    assert check["checks"]["observed_pressure_bounds_are_inside_extended_domain"]
    assert check["physical_correlations_changed"] is False
    assert check["operating_conditions_changed"] is False
    assert check["new_fitted_physical_parameters"] == []
    assert check["extended_margin_below_observed_min_pa"] > 0.0
    assert check["extended_margin_above_observed_max_pa"] > 0.0

    with np.load(output / "helium_property_table.npz", allow_pickle=False) as table:
        assert table["pressure_pa"].tolist() == [93900.0, 120000.0, 146100.0]
