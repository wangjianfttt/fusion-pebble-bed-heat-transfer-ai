#!/usr/bin/env python3
"""Draw the paper workflow for the P418 pebble-bed heat-transfer study."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
PURPLE = "#8B5FBF"
CHARCOAL = "#222222"
MIDGRAY = "#777777"
LIGHTGRAY = "#F3F4F5"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def box(ax, x: float, width: float, color: str, title: str) -> tuple[float, float]:
    y, height = 0.27, 0.59
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor="white",
            edgecolor=CHARCOAL,
            linewidth=1.0,
            zorder=1,
        )
    )
    ax.add_patch(
        Rectangle(
            (x, y + height - 0.075),
            width,
            0.075,
            facecolor=color,
            edgecolor=CHARCOAL,
            linewidth=1.0,
            zorder=2,
        )
    )
    ax.text(
        x + width / 2,
        y + height - 0.037,
        title,
        ha="center",
        va="center",
        color="white",
        fontsize=7.0,
        fontweight="bold",
        zorder=3,
    )
    return y, height


def horizontal_arrow(
    ax,
    x0: float,
    x1: float,
    y: float,
    label: str = "",
    color: str = CHARCOAL,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.5,
            color=color,
            connectionstyle="arc3,rad=0",
            shrinkA=0,
            shrinkB=0,
            zorder=8,
        )
    )
    if label:
        ax.text(
            (x0 + x1) / 2,
            y + 0.020,
            label,
            ha="center",
            va="bottom",
            color=color,
            fontsize=5.2,
            fontweight="bold",
            zorder=9,
        )


def draw_physical_reference(ax, x: float, width: float) -> None:
    box(ax, x, width, BLUE, "3D reference")
    # Pebble-bed icon with orthogonal through-flow arrows.
    cx0, cy0 = x + 0.030, 0.575
    for row in range(3):
        for col in range(4):
            cx = cx0 + col * 0.032 + (0.016 if row % 2 else 0.0)
            cy = cy0 + row * 0.052
            ax.add_patch(
                Circle(
                    (cx, cy),
                    0.014,
                    facecolor="#D9D9D9",
                    edgecolor=CHARCOAL,
                    linewidth=0.7,
                    zorder=4,
                )
            )
    for offset in (-0.020, 0.015, 0.050):
        ax.add_patch(
            FancyArrowPatch(
                (x + 0.012, 0.61 + offset),
                (x + width - 0.012, 0.61 + offset),
                arrowstyle="-|>",
                mutation_scale=7,
                linewidth=0.9,
                color=SKY,
                connectionstyle="arc3,rad=0",
                zorder=3,
            )
        )
    ax.plot([x + 0.012, x + 0.012], [0.535, 0.745], color=VERMILION, lw=2.0)
    ax.text(x + 0.020, 0.526, r"$T_{\rm wall}=635$ K", fontsize=6.2, color=VERMILION)
    ax.text(
        x + width / 2,
        0.455,
        "OpenFOAM CHT",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
    )
    ax.text(
        x + width / 2,
        0.385,
        r"$U_{\rm in},\;T_{\rm in},\;q'''$" + "\n" + r"$\mathbf{u},p,T_f,T_s,\dot Q$",
        ha="center",
        va="center",
        fontsize=7.0,
        linespacing=1.35,
    )
    ax.text(
        x + width / 2,
        0.305,
        "60-condition matrix",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=MIDGRAY,
    )


def draw_pinn(ax, x: float, width: float) -> None:
    box(ax, x, width, BLUE, "Steady PINN")
    layer_x = [x + 0.035, x + width / 2, x + width - 0.035]
    layer_y = [
        [0.59, 0.66],
        [0.555, 0.625, 0.695],
        [0.59, 0.66],
    ]
    for i in range(len(layer_x) - 1):
        for y0 in layer_y[i]:
            for y1 in layer_y[i + 1]:
                ax.plot(
                    [layer_x[i], layer_x[i + 1]],
                    [y0, y1],
                    color="#B7B7B7",
                    lw=0.45,
                    zorder=2,
                )
    for i, lx in enumerate(layer_x):
        for ly in layer_y[i]:
            ax.add_patch(
                Circle(
                    (lx, ly),
                    0.010,
                    facecolor="white",
                    edgecolor=BLUE,
                    linewidth=1.0,
                    zorder=4,
                )
            )
    ax.text(
        x + width / 2,
        0.510,
        "new operating\ncondition",
        ha="center",
        va="center",
        fontsize=6.2,
        linespacing=1.05,
    )
    ax.text(
        x + width / 2,
        0.430,
        r"$\widehat T_0(\mathbf{x})$" + "\ninitial thermal field",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        linespacing=1.3,
        color=BLUE,
    )
    ax.text(
        x + width / 2,
        0.320,
        "validation selects\ntraining length",
        ha="center",
        va="center",
        fontsize=6.4,
        color=MIDGRAY,
    )


def draw_graph_transformer(ax, x: float, width: float) -> None:
    box(ax, x, width, GREEN, "Graph--Transformer")
    # Local graph.
    nodes = [
        (x + 0.030, 0.590),
        (x + 0.058, 0.655),
        (x + 0.072, 0.570),
        (x + 0.098, 0.635),
    ]
    for i, j in [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]:
        ax.plot(
            [nodes[i][0], nodes[j][0]],
            [nodes[i][1], nodes[j][1]],
            color="#999999",
            lw=0.65,
            zorder=2,
        )
    for nx, ny in nodes:
        ax.add_patch(Circle((nx, ny), 0.010, facecolor="white", edgecolor=GREEN, lw=1.0))
    # Physics-attention slices and temporal stack.
    for i in range(3):
        ax.add_patch(
            Rectangle(
                (x + 0.115 + i * 0.008, 0.567 + i * 0.016),
                0.030,
                0.078,
                facecolor="#D9F0E8",
                edgecolor=GREEN,
                linewidth=0.7,
                zorder=3 + i,
            )
        )
    for i in range(4):
        ax.plot(
            [x + width - 0.040, x + width - 0.040],
            [0.555 + i * 0.025, 0.570 + i * 0.025],
            color=GREEN,
            lw=2.0,
        )
    ax.add_patch(
        FancyArrowPatch(
            (x + 0.101, 0.615),
            (x + 0.119, 0.615),
            arrowstyle="-|>",
            mutation_scale=7,
            lw=0.9,
            color=CHARCOAL,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (x + 0.156, 0.615),
            (x + width - 0.050, 0.615),
            arrowstyle="-|>",
            mutation_scale=7,
            lw=0.9,
            color=CHARCOAL,
        )
    )
    ax.text(
        x + width / 2,
        0.500,
        "local exchange\nPhysics-Attention + time",
        ha="center",
        va="center",
        fontsize=5.7,
        linespacing=1.05,
    )
    ax.text(
        x + width / 2,
        0.420,
        r"$\widehat{\mathbf{T}}_{1:N_t}$" + "\nfluid + solid trajectory",
        ha="center",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color=GREEN,
    )
    ax.text(
        x + width / 2,
        0.315,
        "complete endpoint pairs\nheld out",
        ha="center",
        va="center",
        fontsize=5.9,
        color=MIDGRAY,
    )


def draw_diffusion(ax, x: float, width: float) -> None:
    box(ax, x, width, PURPLE, "Diffusion correction")
    xs = [x + 0.025 + i * (width - 0.050) / 28 for i in range(29)]
    base = [0.655 - 0.060 * (i / 28) - 0.020 * __import__("math").sin(i / 4) for i in range(29)]
    residual = [0.010 * __import__("math").sin(i / 2.7) for i in range(29)]
    refined = [a + b for a, b in zip(base, residual)]
    ax.plot(xs, base, color="#8C8C8C", lw=1.0, ls="--")
    ax.plot(xs, refined, color=PURPLE, lw=1.5)
    ax.text(x + 0.025, 0.690, r"$\Delta T$ only", fontsize=6.5, color=PURPLE, fontweight="bold")
    ax.text(
        x + width / 2,
        0.505,
        "temperature improves",
        ha="center",
        fontsize=6.1,
    )
    ax.text(
        x + width / 2,
        0.435,
        "energy balance\ndoes not worsen",
        ha="center",
        va="center",
        fontsize=6.1,
        fontweight="bold",
    )
    ax.text(
        x + width / 2,
        0.335,
        r"fixed $\mathbf{u},p,T_0$" + "\nPOD control",
        ha="center",
        va="center",
        fontsize=6.5,
        color=MIDGRAY,
    )


def draw_tests(ax, x: float, width: float) -> None:
    box(ax, x, width, ORANGE, "Independent test")
    # Two distinct packing thumbnails.
    for shift, color in [(0.025, BLUE), (0.095, ORANGE)]:
        ax.add_patch(
            Rectangle(
                (x + shift, 0.585),
                0.055,
                0.110,
                facecolor="white",
                edgecolor=color,
                linewidth=0.8,
            )
        )
        centers = [
            (x + shift + 0.014, 0.606),
            (x + shift + 0.038, 0.610),
            (x + shift + 0.022, 0.638),
            (x + shift + 0.044, 0.656),
            (x + shift + 0.014, 0.676),
        ]
        for cx, cy in centers:
            ax.add_patch(Circle((cx, cy), 0.007, facecolor="#E6E6E6", edgecolor=color, lw=0.6))
    ax.text(
        x + width / 2,
        0.548,
        "condition holdout | new packing",
        ha="center",
        fontsize=5.1,
    )
    metrics = ["temperature + hotspot", "pressure + wall heat", "mass + energy"]
    for i, label in enumerate(metrics):
        yy = 0.468 - i * 0.060
        ax.plot([x + 0.020, x + 0.034], [yy, yy], color=ORANGE, lw=2.0)
        ax.text(x + 0.041, yy, label, va="center", fontsize=5.3)
    ax.text(
        x + width / 2,
        0.305,
        "accuracy + speed + conservation",
        ha="center",
        va="bottom",
        fontsize=5.1,
        color=MIDGRAY,
    )


def build_figure(output_stem: Path) -> list[Path]:
    setup_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.65))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    positions = [
        (0.010, 0.180),
        (0.210, 0.170),
        (0.400, 0.200),
        (0.620, 0.170),
        (0.810, 0.180),
    ]
    draw_physical_reference(ax, *positions[0])
    draw_pinn(ax, *positions[1])
    draw_graph_transformer(ax, *positions[2])
    draw_diffusion(ax, *positions[3])
    draw_tests(ax, *positions[4])

    y_arrow = 0.535
    horizontal_arrow(ax, 0.190, 0.205, y_arrow)
    horizontal_arrow(ax, 0.380, 0.395, y_arrow)
    horizontal_arrow(ax, 0.600, 0.615, y_arrow)
    horizontal_arrow(ax, 0.790, 0.805, y_arrow)

    # Shared physics strip across the complete workflow.
    ax.add_patch(
        Rectangle(
            (0.010, 0.055),
            0.980,
            0.115,
            facecolor=LIGHTGRAY,
            edgecolor=CHARCOAL,
            linewidth=0.9,
        )
    )
    ax.text(
        0.500,
        0.135,
        "Shared finite-volume physics",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
    )
    ax.text(
        0.500,
        0.090,
        r"$\nabla\!\cdot(\rho\mathbf{u})=0$"
        "     |     fluid and solid energy     |     "
        r"$T_{\Gamma,f}=T_{\Gamma,s}$"
        "     |     "
        r"$\dot Q_f+\dot Q_s=0$",
        ha="center",
        va="center",
        fontsize=7.2,
    )
    ax.text(
        0.010,
        0.925,
        "Source-backed inputs and complete physical holdouts",
        ha="left",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color=CHARCOAL,
    )
    ax.text(
        0.990,
        0.925,
        "No random cell or time-point split",
        ha="right",
        va="center",
        fontsize=7.0,
        color=MIDGRAY,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in [
        (".pdf", {}),
        (".svg", {}),
        (".png", {"dpi": 600}),
    ]:
        path = output_stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.03, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    for path in build_figure(args.output_stem.resolve()):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
