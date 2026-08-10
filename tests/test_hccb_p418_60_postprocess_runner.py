#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/run_hccb_p418_60_postprocess.sh"


class P418PostprocessRunnerTest(unittest.TestCase):
    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_runner_contains_full_physical_targets(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for required in (
            "update_hccb_p418_matrix_parameter_sources.py",
            "verify_hccb_p418_actual_case_inputs.py",
            "build_hccb_p418_source_summary.py",
            "build_hccb_p418_shared_mesh_dataset.py",
            "export_hccb_p418_boundary_heat_flux_targets.py",
            "export_hccb_p418_experimental_comparison_targets.py",
            "check_hccb_p418_pressure_density_consistency.py",
            "build_hccb_p418_regional_state_targets.py",
            "quantify_hccb_p418_regional_representation_fidelity.py",
            "compare_hccb_p418_native_reconstruction.py",
            "build_hccb_p418_regional_mass_flux_targets.py",
            "build_hccb_p418_regional_energy_flux_targets.py",
            "build_hccb_p418_training_statistics.py",
            "summarize_hccb_p418_steady_hotspots.py",
        ):
            self.assertIn(required, text)
        self.assertIn("EXPECTED_CASES=${EXPECTED_CASES:-60}", text)
        self.assertIn("--sample-paths-from-completion-markers", text)
        self.assertIn("--time-from-completion-marker", text)
        self.assertIn("--expected-case-count", text)
        self.assertIn("--require-completion-markers", text)
        self.assertIn("--require-sourceflow-mapping", text)
        self.assertIn("--require-steady-final-window", text)
        self.assertIn("STEADY_HOTSPOT_DIR", text)
        self.assertIn("/opt/openfoam13/etc/bashrc", text)
        self.assertIn("set +e\nsource \"${OPENFOAM_BASHRC}\"", text)
        self.assertIn("openfoam_source_status=$?", text)


if __name__ == "__main__":
    unittest.main()
