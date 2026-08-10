#!/usr/bin/env python3
"""Plot P418 model accuracy against measured OpenFOAM training cost."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


STYLE = {
    "response_surface": ("black", "o", "Response surface"),
    "pinn_data_only": ("#777777", "s", "PINN, data only"),
    "pinn": ("#d20a0a", "o", "PINN + balances"),
    "graph": ("#075dcc", "^", "Graph operator"),
    "transolver": ("#168a45", "D", "Transolver"),
}

METRICS = (
    ("test_state_normalized_rmse", "Normalized state RMSE"),
    ("test_outlet_temperature_p95_K", "Outlet temperature p95 (K)"),
    ("test_solid_maximum_temperature_p95_K", "Maximum solid temperature p95 (K)"),
    ("test_energy_balance_normalized_rmse", "Normalized energy-equation RMSE"),
)


def plot(input_csv: Path, output_dir: Path) -> dict[str, object]:
    with input_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("learning-curve efficiency table is empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), constrained_layout=True)
    architectures = [name for name in STYLE if any(row["architecture"] == name for row in rows)]
    cost_by_count = {
        int(row["train_case_count"]): float(row["openfoam_training_core_hours_32ranks"])
        for row in rows
    }
    training_counts = sorted(cost_by_count)
    training_costs = [cost_by_count[count] for count in training_counts]
    for panel, (axis, (metric, ylabel)) in enumerate(zip(axes.flat, METRICS)):
        for architecture in architectures:
            color, marker, label = STYLE[architecture]
            values = sorted(
                (row for row in rows if row["architecture"] == architecture),
                key=lambda row: int(row["train_case_count"]),
            )
            x = [float(row["openfoam_training_core_hours_32ranks"]) for row in values]
            y = [float(row[metric]) for row in values]
            axis.plot(
                x,
                y,
                color=color,
                marker=marker,
                linewidth=1.8,
                markersize=5.0,
                label=label,
            )
        axis.set_xlabel("OpenFOAM training cost (core h)")
        axis.set_ylabel(ylabel)
        top_axis = axis.secondary_xaxis("top")
        top_axis.set_xticks(training_costs, [str(count) for count in training_counts])
        top_axis.set_xlabel("Training cases", labelpad=3)
        axis.grid(True, color="#dddddd", linewidth=0.6, alpha=0.75)
        axis.text(
            -0.13,
            1.04,
            f"({chr(ord('a') + panel)})",
            transform=axis.transAxes,
            fontsize=10.5,
            fontweight="bold",
        )
        for spine in axis.spines.values():
            spine.set_visible(True)
    axes.flat[0].legend(loc="best", fontsize=8.3)
    pdf = output_dir / "learning_curve_efficiency.pdf"
    png = output_dir / "learning_curve_efficiency.png"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.canvas.draw()
    panel_boxes = [axis.get_position().bounds for axis in axes.flat]
    panel_widths = [box[2] for box in panel_boxes]
    panel_heights = [box[3] for box in panel_boxes]
    width_to_height = [width / height for width, height in zip(panel_widths, panel_heights)]
    plt.close(figure)
    payload = {
        "status": "p418_learning_curve_figure_ready",
        "source": str(input_csv),
        "pdf": pdf.name,
        "png": png.name,
        "panels": [metric for metric, _ in METRICS],
        "panel_width_spread": max(panel_widths) - min(panel_widths),
        "panel_height_spread": max(panel_heights) - min(panel_heights),
        "panel_width_to_height_ratio": sum(width_to_height) / len(width_to_height),
        "new_physical_parameters": [],
    }
    (output_dir / "learning_curve_figure.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = plot(args.input_csv.resolve(), args.output_dir.resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
