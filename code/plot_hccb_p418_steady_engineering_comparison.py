#!/usr/bin/env python3
"""Plot like-for-like P418 engineering errors for the steady model comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = {
    "response_surface": ("Response surface", "#000000", "o"),
    "pinn_data_only": ("PINN, data only", "#999999", "^"),
    "pinn": ("PINN, mass + energy", "#0072B2", "o"),
    "graph": ("Graph operator", "#009E73", "s"),
    "transolver": ("Physics attention", "#D55E00", "D"),
}

PANELS = (
    ("test_outlet_temperature_p95_K", "Outlet-temperature p95 error (K)"),
    ("test_solid_maximum_temperature_p95_K", "Maximum-solid-temperature p95 error (K)"),
    (
        "test_cooling_wall_heat_over_generated_p95_percent",
        "Cooling-wall heat error / generated power (%)",
    ),
    (
        "test_interphase_net_heat_over_generated_p95_percent",
        "Solid-to-fluid heat error / generated power (%)",
    ),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"comparison table is empty: {path}")
    required = {"architecture", "split", *(name for name, _ in PANELS)}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"comparison table lacks columns: {sorted(missing)}")
    return rows


def configure_axis(axis: plt.Axes) -> None:
    axis.tick_params(which="both", direction="in", top=True, right=True, width=0.7)
    axis.tick_params(which="major", length=4)
    axis.tick_params(which="minor", length=2)
    axis.minorticks_on()
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.comparison_csv.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    splits = list(dict.fromkeys(row["split"] for row in rows))
    architectures = [name for name in METHODS if any(row["architecture"] == name for row in rows)]
    if not architectures:
        raise ValueError("comparison table contains no declared model architecture")
    lookup = {(row["architecture"], row["split"]): row for row in rows}

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "savefig.bbox": "tight",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(6.75, 5.25), constrained_layout=True)
    offsets = np.linspace(-0.28, 0.28, len(architectures))
    plotted: dict[str, object] = {}
    for panel_index, (axis, (column, ylabel)) in enumerate(zip(axes.flat, PANELS)):
        configure_axis(axis)
        all_values: list[float] = []
        for offset, architecture in zip(offsets, architectures):
            label, color, marker = METHODS[architecture]
            values = []
            positions = []
            for split_index, split in enumerate(splits):
                row = lookup.get((architecture, split))
                if row is None:
                    continue
                value = float(row[column])
                if not np.isfinite(value):
                    raise ValueError(f"non-finite {column} for {architecture} {split}")
                values.append(value)
                positions.append(split_index + offset)
                all_values.append(value)
            handle = axis.scatter(
                positions,
                values,
                s=25,
                marker=marker,
                facecolor="white" if architecture == "response_surface" else color,
                edgecolor=color,
                linewidth=0.9,
                zorder=3,
            )
            plotted.setdefault(architecture, handle)
        axis.set_xticks(np.arange(len(splits)))
        axis.set_xticklabels([name.replace("_", "\n") for name in splits])
        axis.set_ylabel(ylabel)
        axis.text(
            0.018,
            0.975,
            f"({chr(ord('a') + panel_index)})",
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=9.0,
            va="top",
            ha="left",
        )
        positive = [value for value in all_values if value > 0]
        if positive and max(positive) / min(positive) >= 100.0:
            axis.set_yscale("log")
        axis.margins(x=0.08, y=0.12)

    figure.legend(
        [plotted[name] for name in architectures],
        [METHODS[name][0] for name in architectures],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=min(len(architectures), 5),
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.1,
    )
    pdf = output / "steady_engineering_error_comparison.pdf"
    png = output / "steady_engineering_error_comparison.png"
    figure.savefig(pdf)
    figure.savefig(png, dpi=600)
    plt.close(figure)
    summary = {
        "status": "steady_engineering_comparison_plotted",
        "comparison_csv": str(args.comparison_csv.resolve()),
        "architectures": architectures,
        "splits": splits,
        "panels": [name for name, _ in PANELS],
        "pdf": str(pdf),
        "png": str(png),
        "new_physical_parameters": [],
    }
    (output / "steady_engineering_error_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
