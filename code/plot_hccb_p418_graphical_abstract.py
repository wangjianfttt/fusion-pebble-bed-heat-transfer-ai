#!/usr/bin/env python3
"""Create the optional IJHMT graphical abstract from validated paper figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
INK = "#161616"
MODEL_LABELS = {
    "graph_transformer_data_only": "Data-only\ngraph Transformer",
    "graph_transformer_energy_flux": "Physics-constrained\ngraph Transformer",
    "graph_transformer_factorized_energy_flux": "Factorized\ngraph Transformer",
    "low_rank_residual_correction": "POD residual\ncorrection",
    "diffusion_residual_correction": "Diffusion residual\ncorrection",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crop_fraction(image: Image.Image, bounds: tuple[float, float, float, float]) -> Image.Image:
    left, top, right, bottom = bounds
    width, height = image.size
    return image.crop(
        (
            round(left * width),
            round(top * height),
            round(right * width),
            round(bottom * height),
        )
    )


def require_final_inputs(root: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    marker = root / "manuscript/generated_openfoam_model_field_comparison_validated.tex"
    domain = root / "figures/hccb_p418_physical_model_domain.png"
    field = root / "figures/hccb_p418_openfoam_model_field_comparison.png"
    selection = root / "figures/hccb_p418_openfoam_model_field_selection.json"
    missing = [
        str(path)
        for path in (marker, domain, field, selection)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "graphical abstract requires validated final inputs: " + ", ".join(missing)
        )
    if marker.stat().st_size == 0 or field.stat().st_size == 0:
        raise RuntimeError("validated field marker or final field figure is empty")
    selection_record = json.loads(selection.read_text(encoding="utf-8"))
    selected_model = str(selection_record.get("selected_model", ""))
    if (
        selection_record.get("status")
        != "selected_p418_field_figure_learned_model"
        or selection_record.get("selection_data_role") != "validation"
        or selection_record.get("display_data_role") != "test"
        or selected_model not in MODEL_LABELS
    ):
        raise RuntimeError("graphical abstract model selection is incomplete")
    return domain, field, selection, selection_record


def render(root: Path, output_stem: Path) -> dict[str, object]:
    domain_path, field_path, selection_path, selection_record = require_final_inputs(root)
    domain_image = Image.open(domain_path).convert("RGB")
    field_image = Image.open(field_path).convert("RGB")

    packed_bed = crop_fraction(domain_image, (0.50, 0.01, 0.99, 0.49))
    model = crop_fraction(domain_image, (0.50, 0.50, 0.99, 0.99))
    field = crop_fraction(field_image, (0.06, 0.01, 0.93, 0.67))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.linewidth": 0.8,
            "text.color": INK,
        }
    )
    fig = plt.figure(figsize=(13.0 / 2.54, 5.0 / 2.54), dpi=320, facecolor="white")
    axes = (
        fig.add_axes((0.005, 0.12, 0.285, 0.82)),
        fig.add_axes((0.355, 0.12, 0.285, 0.82)),
        fig.add_axes((0.700, 0.10, 0.295, 0.84)),
    )
    for axis, image in zip(axes, (packed_bed, model, field)):
        axis.imshow(image)
        axis.set_axis_off()

    headings = (
        (0.148, "Pore-resolved\npacked bed", ORANGE),
        (
            0.498,
            "Validation-selected\n"
            + MODEL_LABELS[str(selection_record["selected_model"])],
            BLUE,
        ),
        (0.848, "Independent\nfull-field test", GREEN),
    )
    for x, label, colour in headings:
        fig.text(x, 0.985, label, ha="center", va="top", fontsize=6.3, weight="bold", color=colour, linespacing=0.9)

    for start, end in ((0.294, 0.351), (0.643, 0.696)):
        fig.add_artist(
            FancyArrowPatch(
                (start, 0.51),
                (end, 0.51),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.5,
                color="#4D4D4D",
            )
        )

    fig.text(0.148, 0.012, "60 steady + 12 transient cases", ha="center", va="bottom", fontsize=5.0)
    fig.text(0.498, 0.012, "Selection uses validation trajectories", ha="center", va="bottom", fontsize=5.0)
    fig.text(0.848, 0.012, "Fluid + solid temperatures at 25 s", ha="center", va="bottom", fontsize=5.0)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pdf": output_stem.with_suffix(".pdf"),
        "png": output_stem.with_suffix(".png"),
        "svg": output_stem.with_suffix(".svg"),
    }
    fig.savefig(outputs["pdf"], bbox_inches=None, facecolor="white")
    fig.savefig(outputs["svg"], bbox_inches=None, facecolor="white")
    fig.savefig(outputs["png"], dpi=320, bbox_inches=None, facecolor="white")
    plt.close(fig)

    png_size = Image.open(outputs["png"]).size
    if png_size[0] < 1328 or png_size[1] < 531:
        raise RuntimeError(f"graphical abstract is below IJHMT minimum size: {png_size}")
    record = {
        "status": "p418_ijhmt_graphical_abstract_ready",
        "source_mode": "deterministic_crop_of_validated_project_figures",
        "generative_ai_used_for_image": False,
        "png_size_pixels": list(png_size),
        "inputs": {
            str(domain_path.relative_to(root)): sha256(domain_path),
            str(field_path.relative_to(root)): sha256(field_path),
            str(selection_path.relative_to(root)): sha256(selection_path),
        },
        "selected_model": selection_record["selected_model"],
        "selection_data_role": selection_record["selection_data_role"],
        "display_data_role": selection_record["display_data_role"],
        "outputs": {
            kind: {"path": str(path), "sha256": sha256(path)}
            for kind, path in outputs.items()
        },
        "new_physical_parameters": [],
    }
    record_path = output_stem.with_suffix(".json")
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.project_root.resolve(), args.output_stem.resolve()), indent=2))


if __name__ == "__main__":
    main()
