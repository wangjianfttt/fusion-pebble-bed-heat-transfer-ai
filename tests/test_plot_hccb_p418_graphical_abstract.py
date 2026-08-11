from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from plot_hccb_p418_graphical_abstract import render  # noqa: E402


def test_graphical_abstract_requires_validated_final_field(tmp_path: Path) -> None:
    (tmp_path / "figures").mkdir()
    try:
        render(tmp_path, tmp_path / "out/graphical_abstract")
    except FileNotFoundError as error:
        assert "validated final inputs" in str(error)
    else:
        raise AssertionError("unvalidated graphical abstract was generated")


def test_graphical_abstract_is_deterministic_and_large_enough(tmp_path: Path) -> None:
    figures = tmp_path / "figures"
    manuscript = tmp_path / "manuscript"
    figures.mkdir()
    manuscript.mkdir()
    domain = np.zeros((900, 1200, 3), dtype=np.uint8)
    domain[:, :600] = (230, 150, 40)
    domain[:, 600:] = (30, 120, 180)
    field = np.zeros((1500, 1200, 3), dtype=np.uint8)
    field[:1000] = (20, 160, 110)
    Image.fromarray(domain).save(figures / "hccb_p418_physical_model_domain.png")
    Image.fromarray(field).save(figures / "hccb_p418_openfoam_model_field_comparison.png")
    (manuscript / "generated_openfoam_model_field_comparison_validated.tex").write_text(
        "validated\n", encoding="utf-8"
    )

    first = render(tmp_path, tmp_path / "first/graphical_abstract")
    second = render(tmp_path, tmp_path / "second/graphical_abstract")
    assert first["status"] == "p418_ijhmt_graphical_abstract_ready"
    assert first["generative_ai_used_for_image"] is False
    assert first["png_size_pixels"][0] >= 1328
    assert first["png_size_pixels"][1] >= 531
    assert first["outputs"]["png"]["sha256"] == second["outputs"]["png"]["sha256"]
