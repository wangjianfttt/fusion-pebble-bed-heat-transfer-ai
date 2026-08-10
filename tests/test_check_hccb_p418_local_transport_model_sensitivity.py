from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/check_hccb_p418_local_transport_model_sensitivity.py"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch required")
def test_local_transport_input_paths_are_exercised(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "p418_local_transport_input_paths_confirmed"
    assert payload["new_physical_parameters"] == []
    assert all(payload["checks"].values())
    fixed = payload["fixed_flow_operator"]
    assert fixed["initial_state_exact_max_abs"] <= 1.0e-7
    assert fixed["fixed_hydrodynamics_max_abs"] <= 1.0e-7
    assert fixed["velocity_to_final_temperature_max_abs"] > 1.0e-8
    assert fixed["pressure_to_final_temperature_max_abs"] > 1.0e-8
    assert fixed["boundary_role_to_final_temperature_max_abs"] > 1.0e-8
    flux = payload["fixed_flow_physics"]
    assert flux["mass_flux_to_fluid_energy_residual_max_abs_W_m3"] > 0.0
    assert flux["mass_flux_to_internal_energy_flux_max_abs_W"] > 0.0
    coupled = payload["fully_coupled_operator"]
    assert coupled["initial_internal_flux_exact_max_abs"] <= 1.0e-7
    assert coupled["predicted_internal_flux_time_change_max_abs"] > 1.0e-8
    assert coupled["initial_flux_to_predicted_flux_max_abs"] > 1.0e-8
    assert coupled["initial_flux_to_state_direct_max_abs"] > 1.0e-8
    assert (tmp_path / "summary.json").is_file()
    note = tmp_path / "P418_局部流场输入敏感性_CN.md"
    assert note.is_file()
    text = note.read_text(encoding="utf-8")
    assert "不要求据此新增全耦合求解" in text
    assert "仍需等待完整60组稳态工况" not in text
