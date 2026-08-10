#!/usr/bin/env python3
"""Check that final manuscript figures remain editable, sharp and proportionate."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import struct
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


FIGURES = {
    "hccb_p418_physical_response": {
        "status": "complete_60_condition_physical_response_figure",
        "aspect_range": (0.78, 1.05),
        "layout": "two columns by three rows",
        "panel_count": 6,
        "figure_size_mm": [137.16, 170.18],
        "require_embedded_pdf_fonts": True,
    },
    "hccb_p418_steady_model_comparison": {
        "status": "complete_formal_p418_steady_model_comparison_figure",
        "aspect_range": (0.78, 1.05),
        "layout": "two columns by three rows",
        "panel_count": 6,
        "figure_size_mm": [137.16, 170.18],
        "require_embedded_pdf_fonts": True,
    },
    "hccb_p418_transient_model_comparison": {
        "status": "complete_formal_p418_transient_model_comparison_figure",
        "aspect_range": (0.78, 1.05),
        "layout": "two columns by three rows",
        "panel_count": 6,
        "figure_size_mm": [137.16, 170.18],
        "required_metadata_equal": {
            "strict_split_loss_balancing_stage": "validation_selected",
        },
        "require_embedded_pdf_fonts": True,
    },
    "hccb_p418_openfoam_model_field_comparison": {
        "status": "complete_same_scale_openfoam_model_field_comparison",
        "aspect_range": (0.78, 0.88),
        "layout": "two columns by three rows",
        "panel_count": 6,
        "figure_size_mm": [137.16, 170.18],
        "required_metadata_equal": {
            "strict_split_loss_balancing_stage": "validation_selected",
            "selection_data_role": "validation",
            "display_data_role": "test",
        },
        "required_metadata_nonempty": (
            "selected_model",
            "model_selection_file",
            "model_selection_file_sha256",
        ),
        "maximum_svg_raster_layers": 12,
        "maximum_pdf_raster_layers": 20,
        "require_embedded_pdf_fonts": True,
    },
    "hccb_p418_seed202_integral_9": {
        "status": "complete_p418_seed202_integral_9_figure",
        "aspect_range": (1.10, 1.50),
        "layout": "two columns by two rows",
        "panel_count": 4,
        "require_embedded_pdf_fonts": True,
    },
    "hccb_p418_physical_model_domain": {
        "status": "complete_actual_geometry_and_model_figure",
        "aspect_range": (1.10, 1.14),
        "layout": "two columns by two rows",
        "panel_count": 4,
        "figure_size_mm": [137.16, 122.428],
        "maximum_svg_raster_layers": 3,
        "maximum_pdf_raster_layers": 6,
        "require_embedded_pdf_fonts": True,
    },
    "hccb_heat_ai_external_evidence": {
        "status": "external_thermal_hydraulic_comparison_complete",
        "aspect_range": (1.15, 1.25),
        "layout": "two columns by two rows",
        "panel_count": 4,
        "metadata_path": "../results/hccb_heat_ai_external_evidence/summary.json",
        "require_embedded_pdf_fonts": True,
    },
}

MINIMUM_MEDIAN_FONT_SIZE_PX = 7.0
MINIMUM_AUXILIARY_FONT_SIZE_PX = 4.8
MINIMUM_MATH_SUBSCRIPT_FONT_SIZE_PX = 4.2
SANS_SERIF_FAMILIES = ("Arial", "Helvetica", "DejaVu Sans", "sans-serif")
MATHEMATICAL_FONT_FAMILIES = ("cmmi", "cmr", "cmsy", "STIX")


def png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"invalid PNG: {path}")
    width, height = struct.unpack(">II", data[16:24])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid PNG dimensions: {path}")
    return width, height


def svg_checks(
    path: Path,
    expected_panel_count: int,
    maximum_embedded_raster_images: int = 0,
) -> dict:
    text = path.read_text(encoding="utf-8")
    if "<svg" not in text:
        raise ValueError(f"invalid SVG: {path}")
    text_count = len(re.findall(r"<text(?:\s|>)", text))
    image_count = len(re.findall(r"<image(?:\s|>)", text))
    if text_count == 0:
        raise ValueError(f"SVG text was converted to paths: {path}")
    if image_count > maximum_embedded_raster_images:
        raise ValueError(
            f"SVG contains {image_count} embedded raster images, above the "
            f"allowed {maximum_embedded_raster_images}: {path}"
        )
    font_declarations = [
        style
        for style in re.findall(r'style="([^"]+)"', text)
        if re.search(r"(?:^|;)\s*font(?:-size|-family)?\s*:", style)
    ]
    font_records = [
        (float(match.group(1)), declaration)
        for declaration in font_declarations
        if (
            match := re.search(
                r"([0-9]+(?:\.[0-9]+)?)px", declaration
            )
        )
    ]
    font_sizes = [size for size, _ in font_records]
    if not font_sizes:
        raise ValueError(f"SVG has no inspectable font sizes: {path}")
    median_font_size = statistics.median(font_sizes)
    if median_font_size < MINIMUM_MEDIAN_FONT_SIZE_PX:
        raise ValueError(
            f"SVG median font size {median_font_size:.2f}px is below "
            f"{MINIMUM_MEDIAN_FONT_SIZE_PX:.2f}px: {path}"
        )
    undersized = []
    for size, declaration in font_records:
        is_mathematical = bool(
            re.search(r"\b(?:oblique|italic)\b", declaration)
            or any(
                family in declaration
                for family in MATHEMATICAL_FONT_FAMILIES
            )
        )
        minimum = (
            MINIMUM_MATH_SUBSCRIPT_FONT_SIZE_PX
            if is_mathematical
            else MINIMUM_AUXILIARY_FONT_SIZE_PX
        )
        if size < minimum:
            undersized.append((size, declaration, minimum))
    if undersized:
        size, declaration, minimum = min(undersized, key=lambda item: item[0])
        raise ValueError(
            f"SVG auxiliary font size {size:.2f}px is below "
            f"{minimum:.2f}px: {path}: {declaration}"
        )
    sans_serif_count = sum(
        any(family in declaration for family in SANS_SERIF_FAMILIES)
        for declaration in font_declarations
    )
    mathematical_font_count = sum(
        any(family in declaration for family in MATHEMATICAL_FONT_FAMILIES)
        for declaration in font_declarations
    )
    if sans_serif_count + mathematical_font_count != len(font_declarations):
        raise ValueError(f"SVG contains non-sans-serif figure text: {path}")
    sans_serif_fraction = sans_serif_count / len(font_declarations)
    if sans_serif_fraction < 0.90:
        raise ValueError(
            f"SVG sans-serif text fraction {sans_serif_fraction:.3f} is "
            f"below 0.90: {path}"
        )
    panel_labels = re.findall(
        r">[\s]*\(([a-z])\)[\s]*</(?:text|tspan)>", text
    )
    expected_labels = [
        chr(ord("a") + index) for index in range(expected_panel_count)
    ]
    if panel_labels != expected_labels:
        raise ValueError(
            f"SVG panel labels are {panel_labels}, expected "
            f"{expected_labels}: {path}"
        )
    clip_rectangles: list[tuple[float, float]] = []
    tree = ET.parse(path)
    for clip_path in tree.iter():
        if clip_path.tag.rsplit("}", 1)[-1] != "clipPath":
            continue
        for element in clip_path.iter():
            if element.tag.rsplit("}", 1)[-1] != "rect":
                continue
            try:
                width = float(element.attrib["width"])
                height = float(element.attrib["height"])
            except (KeyError, ValueError):
                continue
            if math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0:
                clip_rectangles.append((width, height))
    if len(clip_rectangles) < expected_panel_count:
        raise ValueError(
            f"SVG exposes only {len(clip_rectangles)} clipped plotting boxes, "
            f"expected at least {expected_panel_count}: {path}"
        )
    panel_rectangles = sorted(
        clip_rectangles, key=lambda size: size[0] * size[1], reverse=True
    )[:expected_panel_count]
    panel_widths = [size[0] for size in panel_rectangles]
    panel_heights = [size[1] for size in panel_rectangles]
    width_tolerance = max(panel_widths) * 1.0e-4
    height_tolerance = max(panel_heights) * 1.0e-4
    if max(panel_widths) - min(panel_widths) > width_tolerance:
        raise ValueError(f"SVG panel widths are inconsistent: {path}")
    if max(panel_heights) - min(panel_heights) > height_tolerance:
        raise ValueError(f"SVG panel heights are inconsistent: {path}")
    return {
        "editable_text_element_count": text_count,
        "embedded_raster_image_count": image_count,
        "font_size_px": {
            "minimum": min(font_sizes),
            "median": median_font_size,
            "maximum": max(font_sizes),
        },
        "sans_serif_font_declaration_count": sans_serif_count,
        "mathematical_font_declaration_count": mathematical_font_count,
        "sans_serif_font_fraction": sans_serif_fraction,
        "panel_labels": panel_labels,
        "panel_plot_box_px": {
            "width": statistics.median(panel_widths),
            "height": statistics.median(panel_heights),
            "count": expected_panel_count,
        },
    }


def parse_pdffonts_output(text: str, path: Path) -> dict:
    lines = text.splitlines()
    separator = next(
        (
            index
            for index, line in enumerate(lines)
            if "-" in line and re.fullmatch(r"[-\s]+", line)
        ),
        None,
    )
    if separator is None:
        raise ValueError(f"pdffonts returned no font table: {path}")
    font_lines = [line for line in lines[separator + 1 :] if line.strip()]
    if not font_lines:
        raise ValueError(f"PDF contains no inspectable fonts: {path}")
    records = []
    for line in font_lines:
        match = re.search(
            r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", line
        )
        if match is None:
            raise ValueError(f"cannot parse pdffonts row for {path}: {line}")
        records.append(
            {
                "line": line.strip(),
                "embedded": match.group(1) == "yes",
                "subset": match.group(2) == "yes",
                "unicode": match.group(3) == "yes",
                "type3": bool(re.search(r"\bType 3\b", line)),
            }
        )
    unembedded = [record["line"] for record in records if not record["embedded"]]
    type3 = [record["line"] for record in records if record["type3"]]
    if unembedded:
        raise ValueError(f"PDF contains unembedded fonts: {path}: {unembedded}")
    if type3:
        raise ValueError(f"PDF contains Type 3 fonts: {path}: {type3}")
    return {
        "font_count": len(records),
        "all_fonts_embedded": True,
        "type3_font_count": 0,
    }


def pdf_checks(
    path: Path,
    maximum_embedded_raster_images: int = 0,
    require_embedded_fonts: bool = False,
) -> dict:
    data = path.read_bytes()
    if len(data) < 1000 or not data.startswith(b"%PDF"):
        raise ValueError(f"invalid or empty PDF: {path}")
    image_count = data.count(b"/Subtype /Image")
    if image_count > maximum_embedded_raster_images:
        raise ValueError(
            f"PDF contains {image_count} embedded raster images, above the "
            f"allowed {maximum_embedded_raster_images}: {path}"
        )
    result = {
        "bytes": len(data),
        "embedded_raster_image_count": image_count,
    }
    if require_embedded_fonts:
        completed = subprocess.run(
            ["pdffonts", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        result["fonts"] = parse_pdffonts_output(completed.stdout, path)
    return result


def check_figure(root: Path, basename: str, specification: dict) -> dict:
    paths = {
        suffix: root / f"{basename}.{suffix}"
        for suffix in ("pdf", "svg", "png")
    }
    metadata_path = root / specification.get(
        "metadata_path", f"{basename}.json"
    )
    paths["json"] = metadata_path.resolve()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing final figure outputs: {missing}")
    metadata = json.loads(paths["json"].read_text(encoding="utf-8"))
    if metadata.get("status") != specification["status"]:
        raise ValueError(
            f"{basename} has status {metadata.get('status')}, expected "
            f"{specification['status']}"
        )
    for key, expected in specification.get("required_metadata_equal", {}).items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"{basename} has {key}={metadata.get(key)!r}, "
                f"expected {expected!r}"
            )
    for key in specification.get("required_metadata_nonempty", ()):
        if not metadata.get(key):
            raise ValueError(f"{basename} lacks final metadata field {key}")
    expected_size_mm = specification.get("figure_size_mm")
    metadata_size_mm = metadata.get("figure_size_mm")
    if metadata_size_mm is None and metadata.get("figure_size_inch") is not None:
        metadata_size_mm = [
            round(float(value) * 25.4, 6)
            for value in metadata["figure_size_inch"]
        ]
    if expected_size_mm is not None and metadata_size_mm != expected_size_mm:
        raise ValueError(
            f"{basename} has figure_size_mm={metadata_size_mm}, "
            f"expected {expected_size_mm}"
        )
    width, height = png_dimensions(paths["png"])
    aspect = width / height
    low, high = specification["aspect_range"]
    if not low <= aspect <= high:
        raise ValueError(
            f"{basename} PNG aspect ratio {aspect:.3f} is outside [{low}, {high}]"
        )
    if min(width, height) < 2200:
        raise ValueError(
            f"{basename} PNG is too small for the declared 600 dpi export: "
            f"{width} x {height}"
        )
    if not math.isfinite(aspect):
        raise ValueError(f"{basename} has invalid aspect ratio")
    return {
        "basename": basename,
        "layout": specification["layout"],
        "png_width_px": width,
        "png_height_px": height,
        "png_aspect_ratio": aspect,
        "svg": svg_checks(
            paths["svg"],
            specification["panel_count"],
            specification.get("maximum_svg_raster_layers", 0),
        ),
        "pdf": pdf_checks(
            paths["pdf"],
            specification.get("maximum_pdf_raster_layers", 0),
            specification.get("require_embedded_pdf_fonts", False),
        ),
        "metadata": str(paths["json"].resolve()),
        "figure_size_mm": metadata_size_mm,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.figure_dir.resolve()
    rows = [
        check_figure(root, basename, specification)
        for basename, specification in FIGURES.items()
    ]
    payload = {
        "status": "completed_p418_final_figure_output_check",
        "figures": rows,
        "requirements": {
            "curve_figures_are_fully_vector": True,
            "field_figure_uses_high_resolution_raster_fields_with_vector_text": True,
            "minimum_short_side_pixels": 2200,
            "layout_specific_aspect_ratio": True,
            "minimum_median_svg_font_size_px": (
                MINIMUM_MEDIAN_FONT_SIZE_PX
            ),
            "minimum_auxiliary_svg_font_size_px": (
                MINIMUM_AUXILIARY_FONT_SIZE_PX
            ),
            "minimum_mathematical_subscript_font_size_px": (
                MINIMUM_MATH_SUBSCRIPT_FONT_SIZE_PX
            ),
            "sans_serif_svg_text": True,
            "ordered_panel_labels": True,
            "equal_panel_plot_boxes": True,
            "embedded_pdf_fonts_without_type3": True,
        },
        "new_physical_parameters": [],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
