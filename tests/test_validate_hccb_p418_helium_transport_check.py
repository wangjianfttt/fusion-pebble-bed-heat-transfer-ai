from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "code" / "validate_hccb_p418_helium_transport_check.py"
SPEC = importlib.util.spec_from_file_location("helium_transport_check", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_registered_correlations_are_positive() -> None:
    for pressure in (93413.29837, 120000.0, 145976.4733):
        for temperature in (299.0, 500.0, 700.0, 1001.0):
            assert MODULE.registered_mu(temperature) > 0
            assert MODULE.registered_kappa(pressure, temperature) > 0


def test_parse_openfoam_style_output(tmp_path: Path) -> None:
    output = tmp_path / "check.log"
    output.write_text(
        "OpenFOAM banner\n"
        "p_pa,T_k,mu_pa_s,kappa_w_m_k\n"
        "120000,500,2.808,0.2185\n",
        encoding="utf-8",
    )
    rows = MODULE.parse_rows(output)
    assert rows == [
        {
            "p_pa": 120000.0,
            "T_k": 500.0,
            "mu_pa_s": 2.808,
            "kappa_w_m_k": 0.2185,
        }
    ]
