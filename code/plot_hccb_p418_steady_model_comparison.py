#!/usr/bin/env python3
"""Build the formal steady-model comparison figure for the P418 matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ijhmt_figure_style import apply_ijhmt_style


METHODS = {
    "response_surface": ("response surface", "#202020", "o"),
    "pinn_data_only": ("data-only PINN", "#8C8C8C", "s"),
    "pinn": ("physics PINN", "#0072B2", "o"),
    "graph": ("graph operator", "#009E73", "D"),
    "transolver": ("Physics-Attention", "#D55E00", "^"),
}
SPLITS = (
    "interleaved_all_ranges",
    "temperature_extrapolation",
    "velocity_extrapolation",
    "heat_source_interpolation",
    "heat_source_extrapolation",
)
SPLIT_LABELS = {
    "interleaved_all_ranges": "inside\nrange",
    "temperature_extrapolation": "highest\ninlet $T$",
    "velocity_extrapolation": "highest\ninlet $U$",
    "heat_source_interpolation": "middle\n$q'''$",
    "heat_source_extrapolation": "highest\n$q'''$",
}
PANELS = (
    ("test_fluid_temperature_normalized_rmse", r"$T_f$ RMSE (normalized)"),
    ("test_solid_temperature_normalized_rmse", r"$T_s$ RMSE (normalized)"),
    ("test_pressure_drop_p95_Pa", r"$\Delta p$ p95 error (Pa)"),
    ("test_solid_maximum_temperature_p95_K", r"$T_{s,\max}$ p95 error (K)"),
    (
        "test_cooling_wall_heat_over_generated_p95_percent",
        "Wall-heat p95 error (%)",
    ),
)
MASS_COLUMN = "test_local_mass_l1_over_two_inlet_mean"
ENERGY_COLUMN = "test_local_energy_l1_over_two_generated_power_mean"
FIGURE_SIZE_INCH = (5.40, 6.70)
FIGURE_ADJUST = {
    "left": 0.105,
    "right": 0.99,
    "bottom": 0.065,
    "top": 0.92,
    "wspace": 0.31,
    "hspace": 0.30,
}


def panel_width_to_height_ratio() -> float:
    panel_width = (
        FIGURE_SIZE_INCH[0]
        * (FIGURE_ADJUST["right"] - FIGURE_ADJUST["left"])
        / (2.0 + FIGURE_ADJUST["wspace"])
    )
    panel_height = (
        FIGURE_SIZE_INCH[1]
        * (FIGURE_ADJUST["top"] - FIGURE_ADJUST["bottom"])
        / (3.0 + 2.0 * FIGURE_ADJUST["hspace"])
    )
    return panel_width / panel_height


def read_formal_matrix(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "architecture",
        "split",
        MASS_COLUMN,
        ENERGY_COLUMN,
        *(name for name, _ in PANELS),
    }
    if not rows:
        raise ValueError(f"comparison table is empty: {path}")
    missing_columns = required - set(rows[0])
    if missing_columns:
        raise ValueError(f"comparison table lacks columns: {sorted(missing_columns)}")
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["architecture"], row["split"])
        if key in lookup:
            raise ValueError(f"duplicate steady comparison row: {key}")
        lookup[key] = row
    expected = {(method, split) for method in METHODS for split in SPLITS}
    if set(lookup) != expected:
        missing = sorted(expected - set(lookup))
        extra = sorted(set(lookup) - expected)
        raise ValueError(f"formal 5-model x 5-split matrix is incomplete: missing={missing}, extra={extra}")
    for key, row in lookup.items():
        for column in required - {"architecture", "split"}:
            value = float(row[column])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid {column} for {key}: {row[column]}")
    return lookup


def configure_axis(axis: plt.Axes) -> None:
    axis.tick_params(which="both", direction="in", top=True, right=True, width=0.75)
    axis.tick_params(which="major", length=4.5)
    axis.tick_params(which="minor", length=2.5)
    axis.minorticks_on()
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)


def maybe_log(axis: plt.Axes, values: list[float]) -> None:
    positive = [value for value in values if value > 0.0]
    if len(positive) == len(values) and max(positive) / min(positive) >= 100.0:
        axis.set_yscale("log")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.comparison_csv.resolve()
    lookup = read_formal_matrix(source)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    apply_ijhmt_style(
        font_size=7.9,
        label_size=8.2,
        tick_size=7.1,
        legend_size=6.8,
        axis_width=0.75,
    )
    figure, axes = plt.subplots(3, 2, figsize=FIGURE_SIZE_INCH)
    figure.subplots_adjust(**FIGURE_ADJUST)
    x = np.arange(len(SPLITS), dtype=float)
    offsets = np.linspace(-0.28, 0.28, len(METHODS))
    handles: dict[str, object] = {}

    for panel_index, (axis, (column, ylabel)) in enumerate(zip(axes.flat[:5], PANELS)):
        configure_axis(axis)
        values_for_scale: list[float] = []
        for offset, (method, (label, color, marker)) in zip(offsets, METHODS.items()):
            values = [float(lookup[(method, split)][column]) for split in SPLITS]
            values_for_scale.extend(values)
            handle = axis.scatter(
                x + offset,
                values,
                s=35,
                marker=marker,
                facecolor="white" if method == "response_surface" else color,
                edgecolor=color,
                linewidth=1.0,
                zorder=3,
            )
            handles.setdefault(method, handle)
        axis.set_xticks(x)
        axis.set_xticklabels([SPLIT_LABELS[split] for split in SPLITS])
        axis.set_ylabel(ylabel)
        axis.margins(x=0.055, y=0.14)
        maybe_log(axis, values_for_scale)
        axis.text(-0.025, 1.035, f"({chr(ord('a') + panel_index)})", transform=axis.transAxes,
                  fontweight="bold", fontsize=8.4, ha="left", va="bottom", clip_on=False)

    axis = axes.flat[5]
    configure_axis(axis)
    combined_values: list[float] = []
    for offset, (method, (_, color, marker)) in zip(offsets, METHODS.items()):
        mass = np.asarray([100.0 * float(lookup[(method, split)][MASS_COLUMN]) for split in SPLITS])
        energy = np.asarray([100.0 * float(lookup[(method, split)][ENERGY_COLUMN]) for split in SPLITS])
        combined_values.extend(mass.tolist())
        combined_values.extend(energy.tolist())
        for xpos, low, high in zip(x + offset, mass, energy):
            axis.plot([xpos, xpos], [low, high], color=color, linewidth=0.75, alpha=0.7, zorder=1)
        axis.scatter(x + offset, mass, s=34, marker=marker, facecolor="white", edgecolor=color,
                     linewidth=1.0, zorder=3)
        axis.scatter(x + offset, energy, s=34, marker=marker, facecolor=color, edgecolor=color,
                     linewidth=1.0, zorder=3)
    axis.set_xticks(x)
    axis.set_xticklabels([SPLIT_LABELS[split] for split in SPLITS])
    axis.set_ylabel("Regional balance difference (%)")
    axis.margins(x=0.055, y=0.14)
    maybe_log(axis, combined_values)
    axis.text(0.03, 0.96, "open: mass   filled: energy", transform=axis.transAxes,
              fontsize=8.0, ha="left", va="top")
    axis.text(-0.025, 1.035, "(f)", transform=axis.transAxes, fontweight="bold",
              fontsize=8.4, ha="left", va="bottom", clip_on=False)

    figure.legend(
        [handles[name] for name in METHODS],
        [METHODS[name][0] for name in METHODS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.2,
    )
    pdf = output / "hccb_p418_steady_model_comparison.pdf"
    svg = output / "hccb_p418_steady_model_comparison.svg"
    png = output / "hccb_p418_steady_model_comparison.png"
    canvas_bounds = figure.bbox_inches
    figure.savefig(pdf, bbox_inches=canvas_bounds)
    figure.savefig(svg, bbox_inches=canvas_bounds)
    figure.savefig(png, dpi=600, bbox_inches=canvas_bounds)
    plt.close(figure)
    summary = {
        "status": "complete_formal_p418_steady_model_comparison_figure",
        "comparison_csv": str(source),
        "architectures": list(METHODS),
        "splits": list(SPLITS),
        "matrix_shape": [len(METHODS), len(SPLITS)],
        "panels": [name for name, _ in PANELS] + [MASS_COLUMN, ENERGY_COLUMN],
        "figure_size_inch": list(FIGURE_SIZE_INCH),
        "figure_size_mm": [value * 25.4 for value in FIGURE_SIZE_INCH],
        "panel_width_to_height_ratio": panel_width_to_height_ratio(),
        "pdf": str(pdf),
        "svg": str(svg),
        "png": str(png),
        "new_physical_parameter_values_added": [],
    }
    metadata = output / "hccb_p418_steady_model_comparison.json"
    metadata.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
