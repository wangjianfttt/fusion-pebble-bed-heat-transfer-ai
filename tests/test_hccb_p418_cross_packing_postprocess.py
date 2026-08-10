from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "run_hccb_p418_cross_packing_postprocess.sh"


def test_cross_packing_postprocess_defaults_to_dry_run(tmp_path: Path):
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env={**os.environ, "ROOT": str(tmp_path), "SEED": "202"},
        check=True,
        capture_output=True,
        text=True,
    )
    assert "dry run only" in result.stdout
    assert not (tmp_path / "hccb_dense_cht_p418_cross_packing_seed202_screen9_dataset").exists()
    assert not (tmp_path / "results").exists()


def test_cross_packing_postprocess_requires_declared_seed(tmp_path: Path):
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env={**os.environ, "ROOT": str(tmp_path), "SEED": "404"},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "SEED must be 202 or 303" in result.stderr


def test_cross_packing_postprocess_uses_own_graph_and_seed101_scales():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "native_multiregion_graph/native_multiregion_graph.npz" in text
    assert "build_hccb_p418_regional_topology.py" in text
    assert "build_hccb_p418_model_geometry.py" in text
    assert "build_hccb_p418_subface_residual_geometry.py" in text
    assert "export_hccb_p418_boundary_heat_flux_targets.py" in text
    assert "build_hccb_p418_regional_state_targets.py" in text
    assert "build_hccb_p418_regional_mass_flux_targets.py" in text
    assert "build_hccb_p418_regional_energy_flux_targets.py" in text
    assert "seed101 training statistics" in text
    assert "build_hccb_p418_training_statistics.py" not in text
    assert "--expected-case-count \"${EXPECTED_CASES}\"" in text
