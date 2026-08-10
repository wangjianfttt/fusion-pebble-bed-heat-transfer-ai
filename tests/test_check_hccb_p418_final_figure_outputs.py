from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from check_hccb_p418_final_figure_outputs import (  # noqa: E402
    FIGURES,
    check_figure,
    parse_pdffonts_output,
)


def write_minimal_png(path: Path, width: int, height: int) -> None:
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x06\x00\x00\x00"
    )


def write_fixture(
    root: Path,
    *,
    embedded_image: bool = False,
    font_size: float = 7.2,
    panel_count: int = 6,
    figure_size_mm: list[float] | None = None,
    unequal_panel: bool = False,
) -> tuple[str, dict]:
    basename = "fixture"
    specification = {
        "status": "complete",
        "aspect_range": (0.8, 1.0),
        "layout": "two columns by three rows",
        "panel_count": panel_count,
    }
    if figure_size_mm is not None:
        specification["figure_size_mm"] = figure_size_mm
    (root / f"{basename}.pdf").write_bytes(b"%PDF-1.7\n" + b"0" * 2000)
    image = "<image href='x.png'/>" if embedded_image else ""
    labels = "".join(
        (
            f"<text style=\"font: 700 {font_size}px 'Arial', sans-serif\">"
            f"({chr(ord('a') + index)})</text>"
        )
        for index in range(panel_count)
    )
    clip_paths = "".join(
        (
            f"<clipPath id='clip{index}'><rect width='"
            f"{100.0 + (5.0 if unequal_panel and index == panel_count - 1 else 0.0)}' "
            "height='80.0'/></clipPath>"
        )
        for index in range(panel_count)
    )
    (root / f"{basename}.svg").write_text(
        f"<svg>{labels}{clip_paths}{image}</svg>", encoding="utf-8"
    )
    write_minimal_png(root / f"{basename}.png", 3600, 4000)
    (root / f"{basename}.json").write_text(
        json.dumps(
            {"status": "complete", "figure_size_mm": figure_size_mm}
        ),
        encoding="utf-8",
    )
    return basename, specification


def test_accepts_editable_vector_figure_outputs(tmp_path: Path) -> None:
    basename, specification = write_fixture(tmp_path)
    result = check_figure(tmp_path, basename, specification)
    assert result["png_aspect_ratio"] == 0.9
    assert result["svg"]["editable_text_element_count"] == 6
    assert result["pdf"]["embedded_raster_image_count"] == 0
    assert result["svg"]["font_size_px"]["median"] == 7.2
    assert result["svg"]["panel_labels"] == list("abcdef")
    assert result["svg"]["panel_plot_box_px"] == {
        "width": 100.0,
        "height": 80.0,
        "count": 6,
    }


def test_rejects_embedded_svg_raster(tmp_path: Path) -> None:
    basename, specification = write_fixture(tmp_path, embedded_image=True)
    with pytest.raises(ValueError, match="embedded raster"):
        check_figure(tmp_path, basename, specification)


def test_allows_declared_field_raster_layers(tmp_path: Path) -> None:
    basename, specification = write_fixture(tmp_path, embedded_image=True)
    specification["maximum_svg_raster_layers"] = 1
    result = check_figure(tmp_path, basename, specification)
    assert result["svg"]["embedded_raster_image_count"] == 1


def test_rejects_small_figure_text(tmp_path: Path) -> None:
    basename, specification = write_fixture(tmp_path, font_size=6.5)
    with pytest.raises(ValueError, match="median font size"):
        check_figure(tmp_path, basename, specification)


def test_allows_standard_mathematical_subscript_size(tmp_path: Path) -> None:
    basename, specification = write_fixture(tmp_path)
    svg = tmp_path / f"{basename}.svg"
    text = svg.read_text(encoding="utf-8")
    text = text.replace(
        "</svg>",
        "<text style=\"font: oblique 4.34px 'DejaVu Sans'\">s</text></svg>",
    )
    svg.write_text(text, encoding="utf-8")
    check_figure(tmp_path, basename, specification)


def test_rejects_equally_small_ordinary_annotation(tmp_path: Path) -> None:
    basename, specification = write_fixture(tmp_path)
    svg = tmp_path / f"{basename}.svg"
    text = svg.read_text(encoding="utf-8")
    text = text.replace(
        "</svg>",
        "<text style=\"font: 4.34px 'Arial', sans-serif\">note</text></svg>",
    )
    svg.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="auxiliary font size"):
        check_figure(tmp_path, basename, specification)


def test_rejects_unequal_panel_plot_boxes(tmp_path: Path) -> None:
    basename, specification = write_fixture(tmp_path, unequal_panel=True)
    with pytest.raises(ValueError, match="panel widths are inconsistent"):
        check_figure(tmp_path, basename, specification)


def test_checks_declared_physical_figure_size(tmp_path: Path) -> None:
    basename, specification = write_fixture(
        tmp_path,
        figure_size_mm=[183.0, 103.0],
    )
    result = check_figure(tmp_path, basename, specification)
    assert result["figure_size_mm"] == [183.0, 103.0]
    (tmp_path / f"{basename}.json").write_text(
        json.dumps(
            {"status": "complete", "figure_size_mm": [183.0, 110.0]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="figure_size_mm"):
        check_figure(tmp_path, basename, specification)


def test_field_figure_check_matches_two_by_three_layout() -> None:
    specification = FIGURES["hccb_p418_openfoam_model_field_comparison"]
    assert specification["layout"] == "two columns by three rows"
    assert specification["aspect_range"] == (0.78, 0.88)
    assert specification["figure_size_mm"] == [137.16, 170.18]


def test_transient_figure_matches_registered_two_by_three_canvas() -> None:
    specification = FIGURES["hccb_p418_transient_model_comparison"]
    assert specification["layout"] == "two columns by three rows"
    assert specification["panel_count"] == 6
    assert specification["figure_size_mm"] == [137.16, 170.18]
    assert specification["required_metadata_equal"] == {
        "strict_split_loss_balancing_stage": "validation_selected"
    }


def test_field_figure_requires_final_model_selection_metadata() -> None:
    specification = FIGURES["hccb_p418_openfoam_model_field_comparison"]
    assert specification["required_metadata_equal"] == {
        "strict_split_loss_balancing_stage": "validation_selected",
        "selection_data_role": "validation",
        "display_data_role": "test",
    }
    assert set(specification["required_metadata_nonempty"]) == {
        "selected_model",
        "model_selection_file",
        "model_selection_file_sha256",
    }


def test_rejects_preflight_metadata_for_formal_figure(tmp_path: Path) -> None:
    basename, specification = write_fixture(tmp_path)
    specification["required_metadata_equal"] = {
        "strict_split_loss_balancing_stage": "validation_selected"
    }
    specification["required_metadata_nonempty"] = ("selected_model",)
    with pytest.raises(ValueError, match="strict_split_loss_balancing_stage"):
        check_figure(tmp_path, basename, specification)

    (tmp_path / f"{basename}.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "strict_split_loss_balancing_stage": "validation_selected",
                "selected_model": None,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="selected_model"):
        check_figure(tmp_path, basename, specification)


def test_rejects_test_selected_field_figure(tmp_path: Path) -> None:
    basename, specification = write_fixture(tmp_path)
    specification["required_metadata_equal"] = {
        "selection_data_role": "validation",
        "display_data_role": "test",
    }
    metadata_path = tmp_path / f"{basename}.json"
    metadata_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "selection_data_role": "test",
                "display_data_role": "test",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="selection_data_role"):
        check_figure(tmp_path, basename, specification)


def test_physical_model_figure_matches_registered_canvas() -> None:
    specification = FIGURES["hccb_p418_physical_model_domain"]
    assert specification["layout"] == "two columns by two rows"
    assert specification["aspect_range"] == (1.10, 1.14)
    assert specification["figure_size_mm"] == [137.16, 122.428]


def test_accepts_embedded_non_type3_pdf_fonts() -> None:
    output = """name type encoding emb sub uni object ID
------------------------------------ ----------------- ---------------- --- --- --- ---------
ABCDEF+ArialMT TrueType WinAnsi yes yes yes 10 0
GHIJKL+DejaVuSans CID TrueType Identity-H yes yes yes 12 0
"""
    result = parse_pdffonts_output(output, Path("figure.pdf"))
    assert result == {
        "font_count": 2,
        "all_fonts_embedded": True,
        "type3_font_count": 0,
    }


@pytest.mark.parametrize(
    "row, message",
    [
        ("ArialMT TrueType WinAnsi no no yes 10 0", "unembedded"),
        ("ABCDEF+CMR10 Type 3 Custom yes yes no 12 0", "Type 3"),
    ],
)
def test_rejects_nonportable_pdf_fonts(row: str, message: str) -> None:
    output = f"""name type encoding emb sub uni object ID
------------------------------------ ----------------- ---------------- --- --- --- ---------
{row}
"""
    with pytest.raises(ValueError, match=message):
        parse_pdffonts_output(output, Path("figure.pdf"))


def test_supports_metadata_outside_figure_directory(tmp_path: Path) -> None:
    figure_dir = tmp_path / "figures"
    result_dir = tmp_path / "results" / "evidence"
    figure_dir.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    basename, specification = write_fixture(figure_dir, panel_count=4)
    external_metadata = result_dir / "summary.json"
    external_metadata.write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    (figure_dir / f"{basename}.json").unlink()
    specification["metadata_path"] = "../results/evidence/summary.json"
    result = check_figure(figure_dir, basename, specification)
    assert result["metadata"] == str(external_metadata.resolve())


def test_checker_covers_all_current_multiplot_figures() -> None:
    assert FIGURES["hccb_p418_physical_model_domain"]["panel_count"] == 4
    assert FIGURES["hccb_heat_ai_external_evidence"]["panel_count"] == 4
    assert (
        FIGURES["hccb_heat_ai_external_evidence"]["metadata_path"]
        == "../results/hccb_heat_ai_external_evidence/summary.json"
    )
