from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "build_hccb_openfoam_helium_property_table.py"
CANONICAL = ROOT / "results" / "apd006_hccb_openfoam_helium_property_table"
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_openfoam_helium_property_table import (  # noqa: E402
    helium_kappa,
    helium_mu,
    helium_rho,
    p418_temperature_support,
)


def manifest():
    with (ROOT / "parameters" / "literature_parameter_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        return {row["parameter_id"]: row for row in csv.DictReader(stream)}


def test_formal_table_support_comes_from_p418_and_p426():
    rows = manifest()
    assert p418_temperature_support(rows["P418"]["value"]).tolist() == [
        300.0,
        500.0,
        700.0,
        900.0,
    ]
    assert float(rows["P426"]["value"]) == 0.12


def regenerate(output: Path) -> dict[str, object]:
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads((output / "summary.json").read_text(encoding="utf-8"))


def test_regenerated_table_matches_every_literature_formula_node(tmp_path: Path):
    output = tmp_path / "table"
    summary = regenerate(output)
    assert summary["status"] == "hccb_openfoam_helium_property_table_passed"
    assert summary["parameter_ids"] == [
        "P070",
        "P071",
        "P388",
        "P389",
        "P391",
        "P418",
        "P424",
        "P426",
    ]
    assert summary["checks"]["pressure_center_matches_P426"] is True
    with np.load(output / "helium_property_table.npz", allow_pickle=False) as table:
        pressure = table["pressure_pa"]
        temperature = table["temperature_k"]
        pp, tt = np.meshgrid(pressure, temperature, indexing="ij")
        np.testing.assert_allclose(table["rho_kg_m3"], helium_rho(pp, tt), rtol=0, atol=0)
        np.testing.assert_allclose(table["mu_pa_s"], helium_mu(pp, tt), rtol=0, atol=0)
        np.testing.assert_allclose(
            table["kappa_w_m_k"], helium_kappa(pp, tt), rtol=0, atol=0
        )


def test_current_openfoam_table_values_do_not_change_beyond_roundoff(tmp_path: Path):
    output = tmp_path / "table"
    regenerate(output)
    with np.load(CANONICAL / "helium_property_table.npz", allow_pickle=False) as old:
        with np.load(output / "helium_property_table.npz", allow_pickle=False) as new:
            assert set(old.files) == set(new.files)
            for name in old.files:
                np.testing.assert_allclose(old[name], new[name], rtol=2.0e-15, atol=0.0)
