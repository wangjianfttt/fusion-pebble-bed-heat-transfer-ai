from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_scope_limit_text.py"
SUMMARY = (
    ROOT
    / "results/hccb_p418_public_figure_data/scope_limits_public/scope_limits_summary.json"
)
TRANSPORT_CHECK = (
    ROOT
    / "results/hccb_p418_public_figure_data/openfoam13_direct_transport_build_public.json"
)
DIRECT_COUPLED_FAILURE = (
    ROOT
    / "results/hccb_p418_public_figure_data/direct_transport_scope_limit.json"
)
FAILURE_SCALE = (
    ROOT
    / "results/hccb_p418_fully_coupled_failure_scale_20260812/summary.json"
)


def test_scope_limit_text_uses_verified_failure_records(tmp_path: Path) -> None:
    output = tmp_path / "generated_scope_limits.tex"
    subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--summary",
            str(SUMMARY),
            "--transport-check",
            str(TRANSPORT_CHECK),
            "--direct-coupled-failure",
            str(DIRECT_COUPLED_FAILURE),
            "--failure-scale",
            str(FAILURE_SCALE),
            "--output",
            str(output),
        ],
        check=True,
    )
    text = output.read_text(encoding="utf-8")
    assert "accepted local three-grid sensitivity study" in text
    assert "no full-domain result is claimed" in text
    assert "no solver was run on that mesh" in text
    assert "medium-to-fine thermal-curve differences remained below 0.043" in text
    assert "Three fully coupled Courant-number tests stopped" in text
    assert r"\SI{93413.3}{Pa}" in text
    assert r"\SI{93900}{Pa}" in text
    assert "prescribed hydrodynamic field" in text
    assert "direct OpenFOAM implementation" in text
    assert r"\SI{0.001522}{s}" in text
    assert r"\SI{1308.8}{K}" in text
    assert r"\SI{1300}{K}" in text
    assert r"\cite{kleykamp1996enthalpy}" in text
    assert r"\SI{0.0059}{K}" in text
    assert r"\SI{551.7}{K}" in text
    assert "numerical startup instability" in text
    assert "not fully coupled response evidence" in text
