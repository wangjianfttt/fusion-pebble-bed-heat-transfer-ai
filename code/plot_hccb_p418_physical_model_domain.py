#!/usr/bin/env python3
"""Plot the actual P418 pebble-bed domain and graph--Transformer architecture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ijhmt_figure_style import apply_ijhmt_style


FIGURE_SIZE_INCH = (5.40, 4.82)
BLACK = "#171717"
GRAY = "#6B6B6B"
LIGHT_GRAY = "#E5E7EB"
BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
GOLD = "#D8B75A"
GREEN = "#009E73"
VERMILION = "#D55E00"
PURPLE = "#7A5195"
PALE_BLUE = "#EAF3FA"
PALE_ORANGE = "#FBF0DD"
PALE_GREEN = "#E6F4EE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_style() -> None:
    apply_ijhmt_style(
        font_size=7.8,
        label_size=8.1,
        tick_size=7.0,
        legend_size=6.9,
        axis_width=0.75,
    )


def add_panel_label(axis: plt.Axes, label: str) -> None:
    method = getattr(axis, "text2D", axis.text)
    method(
        -0.10,
        1.045,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        fontweight="bold",
        color=BLACK,
        clip_on=False,
    )


def equal_3d(axis: plt.Axes, limits_mm: np.ndarray) -> None:
    center = limits_mm.mean(axis=1)
    radius = 0.52 * np.max(limits_mm[:, 1] - limits_mm[:, 0])
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def draw_box(axis: plt.Axes, lower: np.ndarray, upper: np.ndarray, **kwargs: object) -> None:
    corners = np.asarray(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ]
    )
    pairs = [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]
    for left, right in pairs:
        axis.plot(
            corners[[left, right], 0],
            corners[[left, right], 1],
            corners[[left, right], 2],
            **kwargs,
        )


def clean_3d(axis: plt.Axes) -> None:
    axis.grid(False)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((1, 1, 1, 0))
        pane.set_edgecolor((1, 1, 1, 0))
    axis.tick_params(direction="in", width=0.75, length=3.0, pad=0.5)
    axis.view_init(elev=23, azim=-56)


def draw_spherical_pebbles(
    axis: plt.Axes,
    centers: np.ndarray,
    radius: float,
) -> None:
    """Render the local packing as actual spheres rather than marker discs."""
    azimuth = np.linspace(0.0, 2.0 * np.pi, 17)
    polar = np.linspace(0.0, np.pi, 10)
    unit_x = np.outer(np.cos(azimuth), np.sin(polar))
    unit_y = np.outer(np.sin(azimuth), np.sin(polar))
    unit_z = np.outer(np.ones_like(azimuth), np.cos(polar))

    # Draw from the back of the selected camera toward the viewer. Matplotlib
    # still performs polygon depth sorting, while this order reduces edge
    # artefacts for overlapping particles.
    camera_depth = -centers[:, 0] - centers[:, 1] + 0.45 * centers[:, 2]
    for center in centers[np.argsort(camera_depth)]:
        axis.plot_surface(
            center[0] + radius * unit_x,
            center[1] + radius * unit_y,
            center[2] + radius * unit_z,
            color=GOLD,
            edgecolor="none",
            linewidth=0,
            antialiased=True,
            shade=True,
            alpha=0.92,
            rasterized=True,
            zorder=1,
        )


def draw_source_packing(
    axis: plt.Axes,
    source: dict[str, np.ndarray],
    crop: dict[str, np.ndarray],
) -> None:
    centers = source["centres_m"] * 1.0e3
    particle_count = len(centers)
    box = source["box_lengths_m"] * 1.0e3
    depth = centers[:, 2]
    order = np.argsort(depth)
    axis.scatter(
        centers[order, 0],
        centers[order, 1],
        centers[order, 2],
        s=3.2,
        c=depth[order],
        cmap="Greys",
        vmin=float(depth.min() - 2.0),
        vmax=float(depth.max()),
        linewidths=0,
        alpha=0.72,
        rasterized=True,
    )
    draw_box(axis, np.zeros(3), box, color=BLACK, lw=0.8)
    crop_lo = crop["parent_crop_lower_m"] * 1.0e3
    crop_hi = crop["parent_crop_upper_m"] * 1.0e3
    draw_box(axis, crop_lo, crop_hi, color=VERMILION, lw=1.6)
    axis.text(
        crop_lo[0],
        crop_hi[1] + 0.45,
        crop_hi[2],
        "local CHT domain",
        color=VERMILION,
        fontsize=7.4,
        fontweight="bold",
    )
    axis.text2D(
        0.03,
        0.94,
        rf"${particle_count}$ pebbles; $12.5d_p\times12.5d_p\times10d_p$",
        transform=axis.transAxes,
        fontsize=7.0,
        color=GRAY,
        va="top",
    )
    axis.set_xlabel(r"$x/d_p$", labelpad=0)
    axis.set_ylabel(r"$y/d_p$", labelpad=0)
    axis.set_zlabel(r"$z/d_p$", labelpad=0)
    axis.set_xticks([0, 5, 10])
    axis.set_yticks([0, 5, 10])
    axis.set_zticks([0, 5, 10])
    equal_3d(axis, np.column_stack((np.zeros(3), box)))
    clean_3d(axis)
    add_panel_label(axis, "(a)")


def draw_local_domain(axis: plt.Axes, crop: dict[str, np.ndarray]) -> None:
    centers = crop["centres_m"] * 1.0e3
    box = crop["box_lengths_m"] * 1.0e3
    radius = float(crop["physical_radius_m"] * 1.0e3)
    axis.computed_zorder = False
    draw_spherical_pebbles(axis, centers, radius)
    draw_box(axis, np.zeros(3), box, color=BLACK, lw=0.9)
    # Flow direction is z; the x=0 wall is cooled.
    axis.annotate(
        "",
        xy=(0.90, 0.79),
        xytext=(0.90, 0.22),
        xycoords=axis.transAxes,
        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.5),
        annotation_clip=False,
        zorder=100,
    )
    axis.text2D(
        0.90,
        0.18,
        r"$U_{\rm in},\,T_{\rm in}$",
        transform=axis.transAxes,
        ha="center",
        color=BLUE,
        fontsize=7.4,
        fontweight="bold",
        zorder=100,
    )
    axis.text2D(
        0.96,
        0.73,
        "outlet",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color=BLUE,
        fontsize=7.2,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.8),
        zorder=100,
    )
    axis.text2D(
        0.035,
        0.075,
        r"cooled wall: $T_w=635$ K",
        transform=axis.transAxes,
        color=VERMILION,
        fontsize=7.0,
        fontweight="bold",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.0),
        zorder=100,
    )
    axis.text2D(
        0.03,
        0.94,
        rf"$125$ intersecting pebbles; $d_p={2*radius:.1f}$ mm"
        "\nHe flow; pebble heat source $q'''$",
        transform=axis.transAxes,
        fontsize=7.0,
        color=GRAY,
        linespacing=1.15,
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.2),
        zorder=100,
    )
    axis.set_xlabel(r"$x$ (mm)", labelpad=0)
    axis.set_ylabel(r"$y$ (mm)", labelpad=0)
    axis.set_zlabel(r"$z$ (mm)", labelpad=-7)
    axis.set_xticks([0, 2, 4])
    axis.set_yticks([0, 2, 4])
    axis.set_zticks([0, 1.5, 3])
    equal_3d(axis, np.column_stack((np.zeros(3), box)))
    clean_3d(axis)
    add_panel_label(axis, "(b)")


def axis_box(axis: plt.Axes) -> None:
    axis.tick_params(direction="in", top=True, right=True, width=0.75, length=3)
    for spine in axis.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color(BLACK)


def draw_regional_graph(
    axis: plt.Axes,
    geometry: dict[str, np.ndarray],
    boundary: dict[str, np.ndarray],
) -> None:
    coordinates = geometry["node_centroid_m"] * 1.0e3
    node_type = geometry["node_type"]
    edge_source = geometry["edge_source"]
    edge_target = geometry["edge_target"]
    center_y = 0.5 * (coordinates[:, 1].min() + coordinates[:, 1].max())
    selected = np.abs(coordinates[:, 1] - center_y) <= 0.08
    selected_ids = np.flatnonzero(selected)
    local_index = np.full(len(selected), -1, dtype=np.int64)
    local_index[selected_ids] = np.arange(len(selected_ids))
    edge_selected = selected[edge_source] & selected[edge_target]
    edges = np.flatnonzero(edge_selected)
    if len(edges) > 1300:
        generator = np.random.default_rng(20260729)
        edges = np.sort(generator.choice(edges, size=1300, replace=False))
    segments = np.stack(
        (
            coordinates[edge_source[edges]][:, [2, 0]],
            coordinates[edge_target[edges]][:, [2, 0]],
        ),
        axis=1,
    )
    axis.add_collection(
        LineCollection(segments, colors="#B9BDC2", linewidths=0.24, alpha=0.52)
    )
    for material, color, label in (
        (0, BLUE, "fluid region"),
        (1, ORANGE, "solid region"),
    ):
        current = selected & (node_type == material)
        axis.scatter(
            coordinates[current, 2],
            coordinates[current, 0],
            s=5.5,
            color=color,
            edgecolor="none",
            alpha=0.90,
            label=label,
            rasterized=True,
            zorder=3,
        )
    fractions = boundary["level_5_boundary_volume_fraction"]
    cooled = selected & (fractions[:, 2] > 0)
    axis.scatter(
        coordinates[cooled, 2],
        coordinates[cooled, 0],
        s=10,
        facecolors="none",
        edgecolors=VERMILION,
        linewidths=0.55,
        label="cooled-wall support",
        zorder=4,
    )
    axis.set_xlim(coordinates[:, 2].min(), coordinates[:, 2].max())
    axis.set_ylim(coordinates[:, 0].min(), coordinates[:, 0].max())
    axis.set_xlabel("Flow coordinate $z$ (mm)")
    axis.set_ylabel("Wall-normal coordinate $x$ (mm)")
    # Keep the physical x-z scale undistorted while retaining the same square
    # plotting box used by the two 3-D panels and the architecture panel.
    axis.set_box_aspect(1)
    axis.set_aspect("equal", adjustable="datalim")
    axis.legend(
        frameon=True,
        loc="upper right",
        fontsize=6.2,
        handletextpad=0.35,
        borderaxespad=0.35,
        borderpad=0.28,
        labelspacing=0.22,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
    )
    axis_box(axis)
    add_panel_label(axis, "(c)")


def orthogonal_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = BLACK,
    lw: float = 1.2,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            connectionstyle="angle3,angleA=0,angleB=90",
            color=color,
            linewidth=lw,
            shrinkA=2,
            shrinkB=2,
        )
    )


def architecture_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    edge: str,
    face: str,
    fontsize: float = 7.0,
) -> None:
    axis.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
        )
    )
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.15,
        color=BLACK,
    )


def draw_architecture(axis: plt.Axes) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_box_aspect(1)
    axis.axis("off")
    input_boxes = [
        (0.01, 0.83, 0.30, 0.12, "state\n$\\mathbf{u},p$; $T_f,T_s$", BLUE, PALE_BLUE),
        (0.35, 0.83, 0.30, 0.12, "operating inputs\n$U_{in},T_{in},q''',t$", BLUE, PALE_BLUE),
        (0.69, 0.83, 0.30, 0.12, "mesh\nnodes / faces / roles", GREEN, PALE_GREEN),
    ]
    for entry in input_boxes:
        architecture_box(axis, *entry, fontsize=6.1)

    # A common input bus keeps all connectors horizontal or vertical.
    for center_x in (0.16, 0.50, 0.84):
        axis.plot([center_x, center_x], [0.83, 0.78], color=GRAY, lw=0.9)
    axis.plot([0.16, 0.84], [0.78, 0.78], color=GRAY, lw=0.9)
    axis.plot([0.50, 0.24], [0.76, 0.76], color=GRAY, lw=0.9)
    axis.plot([0.50, 0.50], [0.78, 0.76], color=GRAY, lw=0.9)
    axis.add_patch(
        FancyArrowPatch(
            (0.24, 0.76),
            (0.24, 0.73),
            arrowstyle="-|>",
            mutation_scale=9,
            color=GRAY,
            linewidth=1.0,
        )
    )

    pipeline = [
        (0.06, 0.60, 0.36, 0.12, "graph encoder\n2 GNN blocks", ORANGE, PALE_ORANGE),
        (0.58, 0.60, 0.36, 0.12, "Physics-Attention\n2 blocks, 4 heads", PURPLE, "#F1EBF6"),
        (0.06, 0.42, 0.36, 0.12, "temporal Transformer\n3 layers", BLUE, PALE_BLUE),
        (0.58, 0.42, 0.36, 0.12, "graph decoder\n2 GNN blocks", ORANGE, PALE_ORANGE),
    ]
    for entry in pipeline:
        architecture_box(axis, *entry, fontsize=6.2)
    axis.add_patch(
        FancyArrowPatch(
            (0.42, 0.66), (0.58, 0.66), arrowstyle="-|>",
            mutation_scale=9, color=BLACK, linewidth=1.1,
        )
    )
    axis.plot([0.76, 0.76], [0.60, 0.57], color=BLACK, lw=1.1)
    axis.plot([0.76, 0.24], [0.57, 0.57], color=BLACK, lw=1.1)
    axis.add_patch(
        FancyArrowPatch(
            (0.24, 0.57), (0.24, 0.54), arrowstyle="-|>",
            mutation_scale=9, color=BLACK, linewidth=1.1,
        )
    )
    axis.add_patch(
        FancyArrowPatch(
            (0.42, 0.48), (0.58, 0.48), arrowstyle="-|>",
            mutation_scale=9, color=BLACK, linewidth=1.1,
        )
    )
    architecture_box(
        axis,
        0.06,
        0.24,
        0.34,
        0.10,
        "base $T$ field\n$T_f^{base},T_s^{base}$",
        VERMILION,
        "#FCEAE6",
        fontsize=6.2,
    )
    architecture_box(
        axis,
        0.60,
        0.24,
        0.34,
        0.10,
        "diffusion refiner\n3 $T$-residual steps",
        PURPLE,
        "#F1EBF6",
        fontsize=6.2,
    )
    axis.plot([0.76, 0.76], [0.42, 0.38], color=VERMILION, lw=1.1)
    axis.plot([0.23, 0.76], [0.38, 0.38], color=VERMILION, lw=1.1)
    axis.add_patch(
        FancyArrowPatch(
            (0.23, 0.38),
            (0.23, 0.34),
            arrowstyle="-|>",
            mutation_scale=9,
            color=VERMILION,
            linewidth=1.1,
        )
    )
    axis.add_patch(
        FancyArrowPatch(
            (0.40, 0.29),
            (0.60, 0.29),
            arrowstyle="-|>",
            mutation_scale=9,
            color=PURPLE,
            linewidth=1.1,
        )
    )
    architecture_box(
        axis,
        0.31,
        0.045,
        0.38,
        0.10,
        "final temperature\n$T_f^{pred},T_s^{pred}$",
        VERMILION,
        "#FCEAE6",
        fontsize=6.4,
    )
    axis.plot([0.77, 0.77], [0.24, 0.19], color=PURPLE, lw=1.1)
    axis.plot([0.50, 0.77], [0.19, 0.19], color=PURPLE, lw=1.1)
    axis.add_patch(
        FancyArrowPatch(
            (0.50, 0.19),
            (0.50, 0.145),
            arrowstyle="-|>",
            mutation_scale=9,
            color=PURPLE,
            linewidth=1.1,
        )
    )
    add_panel_label(axis, "(d)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-packing",
        type=Path,
        default=Path(
            "data/apd006_hccb_source_sequence_target_packings/"
            "seed101_s80_xlo_ycentre/packing.npz"
        ),
    )
    parser.add_argument(
        "--crop-packing",
        type=Path,
        default=Path("runs/hccb_dense_snappy_g2_nativezone_r2/geometry/packing_crop.npz"),
    )
    parser.add_argument(
        "--regional-geometry",
        type=Path,
        default=Path(
            "results/hccb_p418_actual_spatiotemporal_operator_37time_gpu_data_only/"
            "regional_sequence_geometry.npz"
        ),
    )
    parser.add_argument(
        "--model-geometry",
        type=Path,
        default=Path("results/hccb_p418_60_sourceflow_r3_model_geometry/model_geometry.npz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def main() -> int:
    args = parse_args()
    paths = {
        "source_packing": args.source_packing.resolve(),
        "crop_packing": args.crop_packing.resolve(),
        "regional_geometry": args.regional_geometry.resolve(),
        "model_geometry": args.model_geometry.resolve(),
    }
    source = load_npz(paths["source_packing"])
    crop = load_npz(paths["crop_packing"])
    regional = load_npz(paths["regional_geometry"])
    model_geometry = load_npz(paths["model_geometry"])
    if len(source["centres_m"]) != 2039 or len(crop["centres_m"]) != 125:
        raise ValueError("packing files do not match the registered source and crop")
    if len(regional["node_type"]) != 46089:
        raise ValueError("regional graph is not the registered 46,089-node representation")
    if model_geometry["level_5_boundary_volume_fraction"].shape != (46089, 5):
        raise ValueError("boundary-role matrix has unexpected dimensions")

    setup_style()
    # Match the 390 pt manuscript text width directly.  Generating a wider
    # canvas and shrinking it in LaTeX makes the already compact labels too
    # small in print.
    figure = plt.figure(figsize=FIGURE_SIZE_INCH)
    grid = figure.add_gridspec(
        2,
        2,
        left=0.055,
        right=0.985,
        bottom=0.07,
        # Keep the top-row panel labels outside the axes but inside the
        # exported canvas.  A top value near 1 clips (a) and (b) in PDF.
        top=0.94,
        wspace=0.19,
        hspace=0.23,
    )
    draw_source_packing(figure.add_subplot(grid[0, 0], projection="3d"), source, crop)
    draw_local_domain(figure.add_subplot(grid[0, 1], projection="3d"), crop)
    draw_regional_graph(figure.add_subplot(grid[1, 0]), regional, model_geometry)
    draw_architecture(figure.add_subplot(grid[1, 1]))

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / "hccb_p418_physical_model_domain.pdf"
    svg = output / "hccb_p418_physical_model_domain.svg"
    png = output / "hccb_p418_physical_model_domain.png"
    canvas_bounds = figure.bbox_inches
    figure.savefig(pdf, dpi=600, bbox_inches=canvas_bounds)
    figure.savefig(svg, dpi=600, bbox_inches=canvas_bounds)
    figure.savefig(png, dpi=600, bbox_inches=canvas_bounds)
    plt.close(figure)

    record = {
        "status": "complete_actual_geometry_and_model_figure",
        "source_files": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "source_pebble_count": int(len(source["centres_m"])),
        "local_intersecting_pebble_count": int(len(crop["centres_m"])),
        "figure_size_inch": list(FIGURE_SIZE_INCH),
        "regional_node_count": int(len(regional["node_type"])),
        "fluid_regional_node_count": int(np.sum(regional["node_type"] == 0)),
        "solid_regional_node_count": int(np.sum(regional["node_type"] == 1)),
        "architecture": {
            "hidden_width": 64,
            "local_pre_message_blocks": 2,
            "physics_attention_blocks": 2,
            "physics_attention_heads": 4,
            "physics_slices": 128,
            "temporal_transformer_layers": 3,
            "local_post_message_blocks": 2,
            "diffusion_temperature_refinement_steps": 3,
            "diffusion_corrected_channels": ["temperature"],
        },
        "outputs": {
            "pdf": str(pdf),
            "svg": str(svg),
            "png": str(png),
        },
        "new_physical_parameters": [],
    }
    summary = output / "hccb_p418_physical_model_domain.json"
    summary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
