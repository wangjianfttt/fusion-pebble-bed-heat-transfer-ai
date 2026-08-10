from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from hccb_p418_selected_fixed_flow_chain import (
    STRICT_SPLIT,
    selected_model_directories,
)


def test_strict_split_requires_validation_selection_by_default(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        selected_model_directories(tmp_path, STRICT_SPLIT)


def test_preselection_fallback_is_explicit_and_uses_registered_directories(
    tmp_path: Path,
) -> None:
    directories = selected_model_directories(
        tmp_path,
        STRICT_SPLIT,
        allow_registered_preselection=True,
    )
    assert directories["graph_transformer_energy_flux"].name == (
        "regional_graph_transformer_bounded_physics_pair_disjoint_stress_test"
    )
    assert directories["diffusion_residual_correction"].name == (
        "temporal_diffusion_pair_disjoint_stress_test"
    )
