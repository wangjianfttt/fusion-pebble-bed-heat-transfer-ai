from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from verify_hccb_p418_end_to_end_model_interface import verify


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def inputs() -> tuple[dict, dict, dict, dict, dict, dict]:
    return (
        load("parameters/hccb_p418_fused_model_contract.json"),
        load(
            "results/hccb_p418_actual_spatiotemporal_operator_56time_gpu_factorized/summary.json"
        ),
        load(
            "results/hccb_p418_actual_temporal_diffusion_56time_gpu_batch1_bfloat16_chunk2048/summary.json"
        ),
        load("results/hccb_p418_diffusion_physical_state/summary.json"),
        load("results/hccb_p418_fully_coupled_training_interface/summary.json"),
        load("results/hccb_p418_fused_preflight/summary.json"),
    )


def test_actual_end_to_end_interface_verifies() -> None:
    result = verify(*inputs())
    assert result["status"] == "p418_end_to_end_model_interface_verified"
    assert result["actual_regional_graph"]["nodes"] == 46089
    assert result["actual_regional_graph"]["time_points"] == 56
    assert all(result["checks"].values())
    assert result["formal_accuracy_available"] is False
    assert result["new_physical_parameters"] == []


def test_graph_and_diffusion_must_use_same_mesh() -> None:
    values = list(inputs())
    values[2] = copy.deepcopy(values[2])
    values[2]["regional_topology_sha256"] = "different"
    with pytest.raises(ValueError, match="same_regional_graph"):
        verify(*values)


def test_diffusion_cannot_change_hydrodynamics() -> None:
    values = list(inputs())
    values[3] = copy.deepcopy(values[3])
    values[3]["maximum_absolute_fixed_hydrodynamic_change"] = 1.0e-6
    with pytest.raises(ValueError, match="diffusion_preserves_hydrodynamics"):
        verify(*values)


def test_command_writes_chinese_summary(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "code/verify_hccb_p418_end_to_end_model_interface.py"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    chinese = (tmp_path / "P418_融合模型接口检查_CN.md").read_text(
        encoding="utf-8"
    )
    assert result["status"] == "p418_end_to_end_model_interface_verified"
    assert "真实46,089节点网格" in chinese
