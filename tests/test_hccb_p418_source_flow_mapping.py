from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_gmsh_cht_smoke_case import source_channel_to_pore_velocity  # noqa: E402


def test_particle_cut_inlet_preserves_source_channel_volume_flow() -> None:
    pore_velocity, open_fraction = source_channel_to_pore_velocity(
        0.05, 0.4, 0.6
    )
    assert open_fraction == pytest.approx(0.4)
    assert pore_velocity == pytest.approx(0.125)
    assert pore_velocity * 0.4 == pytest.approx(0.05 * (0.4 + 0.6))


def test_invalid_inlet_area_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        source_channel_to_pore_velocity(0.05, 0.0, 1.0)
