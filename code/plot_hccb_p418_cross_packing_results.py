#!/usr/bin/env python3
"""Plot the formal seed202 architecture comparison and frozen-model seed303 transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARCHITECTURES = ("pinn_data_only", "pinn", "graph", "transolver")
LABELS = {
    "pinn_data_only": "data-only PINN",
    "pinn": "physics PINN",
    "graph": "graph operator",
    "transolver": "Transolver",
}
COLORS = {
    "pinn_data_only": "#999999",
    "pinn": "#0072B2",
    "graph": "#009E73",
    "transolver": "#D55E00",
}
MARKERS = {
    "pinn_data_only": "s",
    "pinn": "o",
    "graph": "D",
    "transolver": "^",
}
TRANSFER_METRICS = (
    ("fluid_temperature_volume_weighted_rmse_K", "$T_f$"),
    ("solid_temperature_volume_weighted_rmse_K", "$T_s$"),
    ("solid_hotspot_location_error_m", "hotspot"),
    ("engineering_absolute_errors.pressure_drop_Pa", r"$\Delta p$"),
    ("engineering_absolute_errors.cooling_wall_heat_into_fluid_W", "wall heat"),
    ("local_mass_l1_over_two_inlet", "mass"),
    ("local_energy_l1_over_two_generated_power", "energy"),
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing formal cross-packing result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def p95(run: dict, name: str) -> float:
    record = run.get("metrics", {}).get(name)
    if not isinstance(record, dict) or "p95" not in record:
        raise ValueError(f"{run.get('architecture')} lacks p95 {name}")
    value = float(record["p95"])
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"invalid p95 {name} for {run.get('architecture')}")
    return value


def development_runs(summary: dict) -> dict[str, dict]:
    if summary.get("status") != "cross_packing_model_summary_complete":
        raise ValueError("seed202 architecture comparison is not complete")
    runs = list(summary.get("runs", []))
    if {int(run["packing_seed"]) for run in runs} != {202}:
        raise ValueError("development comparison must contain seed202 only")
    output = {str(run["architecture"]): run for run in runs}
    if set(output) != set(ARCHITECTURES) or len(runs) != len(ARCHITECTURES):
        raise ValueError("seed202 comparison must contain the four registered architectures")
    return output


def final_runs(summary: dict, selected: str) -> dict[int, dict]:
    if summary.get("status") != "cross_packing_model_summary_complete":
        raise ValueError("seed303 frozen-model comparison is not complete")
    runs = list(summary.get("runs", []))
    if len(runs) != 2:
        raise ValueError("final cross-packing summary must contain seed202 and seed303")
    if {str(run["architecture"]) for run in runs} != {selected}:
        raise ValueError("seed303 result does not use the architecture fixed on seed202")
    output = {int(run["packing_seed"]): run for run in runs}
    if set(output) != {202, 303}:
        raise ValueError("final comparison does not contain both seed202 and seed303")
    return output


def axis_style(axis: plt.Axes) -> None:
    axis.tick_params(which="both", direction="in", top=True, right=True, width=0.85)
    axis.tick_params(which="major", length=4)
    axis.tick_params(which="minor", length=2)
    axis.minorticks_on()
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.85)


def panel_label(axis: plt.Axes, index: int) -> None:
    axis.text(
        -0.14,
        1.07,
        chr(ord("a") + index),
        transform=axis.transAxes,
        fontweight="bold",
        fontsize=9.2,
        ha="left",
        va="bottom",
        clip_on=False,
    )


def scatter_architectures(
    axis: plt.Axes,
    runs: dict[str, dict],
    selected: str,
    x_metric: str,
    y_metric: str,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
) -> None:
    x_values = [p95(runs[name], x_metric) * x_scale for name in ARCHITECTURES]
    y_values = [p95(runs[name], y_metric) * y_scale for name in ARCHITECTURES]
    x_mid = float(np.median(x_values))
    y_mid = float(np.median(y_values))
    for architecture, x_value, y_value in zip(ARCHITECTURES, x_values, y_values):
        axis.scatter(
            x_value,
            y_value,
            s=48 if architecture == selected else 32,
            color=COLORS[architecture],
            marker=MARKERS[architecture],
            edgecolor="black",
            linewidth=0.9 if architecture == selected else 0.4,
            zorder=3,
        )
        if x_value >= x_mid:
            x_offset, horizontal = -4, "right"
        else:
            x_offset, horizontal = 4, "left"
        if y_value >= y_mid:
            y_offset, vertical = -4, "top"
        else:
            y_offset, vertical = 3, "bottom"
        axis.annotate(
            LABELS[architecture],
            (x_value, y_value),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha=horizontal,
            va=vertical,
            fontsize=6.6,
            fontweight="semibold" if architecture == selected else "normal",
        )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-summary", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--final-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=root / "figures")
    args = parser.parse_args()

    development_summary = load_json(args.development_summary.resolve())
    selection = load_json(args.selection.resolve())
    final_summary = load_json(args.final_summary.resolve())
    if selection.get("status") != "seed202_architecture_fixed_before_seed303":
        raise ValueError("seed202 architecture was not formally fixed")
    if selection.get("seed303_fields_read") is not False:
        raise ValueError("architecture selection does not prove seed303 was unseen")
    if selection.get("composite_score_used") is not False:
        raise ValueError("cross-packing architecture selection used an undeclared combined score")
    selected = str(selection["selected_architecture"])
    if selected not in ARCHITECTURES:
        raise ValueError(f"unsupported selected architecture: {selected}")

    development = development_runs(development_summary)
    final = final_runs(final_summary, selected)
    development_selected = development[selected]
    final_seed202 = final[202]
    if development_selected.get("source_sha256") != final_seed202.get("source_sha256"):
        raise ValueError("final summary does not reuse the original seed202 selected-model result")

    ratios = []
    for metric, label in TRANSFER_METRICS:
        denominator = p95(final[202], metric)
        numerator = p95(final[303], metric)
        if denominator <= 0.0 or numerator <= 0.0:
            raise ValueError(f"seed303/seed202 p95 ratio is undefined for {metric}")
        ratios.append((label, numerator / denominator))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 8.2,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "axes.linewidth": 0.85,
            "savefig.bbox": "tight",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(6.75, 5.3))
    figure.subplots_adjust(left=0.11, right=0.985, bottom=0.105, top=0.96, wspace=0.33, hspace=0.34)

    first = axes[0, 0]
    axis_style(first)
    scatter_architectures(
        first,
        development,
        selected,
        "fluid_temperature_volume_weighted_rmse_K",
        "solid_temperature_volume_weighted_rmse_K",
    )
    first.set_xlabel(r"$T_f$ p95 RMSE (K)")
    first.set_ylabel(r"$T_s$ p95 RMSE (K)")
    panel_label(first, 0)

    second = axes[0, 1]
    axis_style(second)
    scatter_architectures(
        second,
        development,
        selected,
        "solid_hotspot_location_error_m",
        "engineering_absolute_errors.pressure_drop_Pa",
        x_scale=1000.0,
    )
    second.set_xlabel("Hotspot p95 error (mm)")
    second.set_ylabel(r"$\Delta p$ p95 error (Pa)")
    panel_label(second, 1)

    third = axes[1, 0]
    axis_style(third)
    scatter_architectures(
        third,
        development,
        selected,
        "local_energy_l1_over_two_generated_power",
        "engineering_absolute_errors.cooling_wall_heat_into_fluid_W",
        x_scale=100.0,
    )
    third.set_xlabel("Regional energy difference, p95 (%)")
    third.set_ylabel("Wall-heat p95 error (W)")
    panel_label(third, 2)

    fourth = axes[1, 1]
    axis_style(fourth)
    y = np.arange(len(ratios))
    ratio_values = np.asarray([value for _, value in ratios])
    fourth.axvline(1.0, color="0.25", linestyle="--", linewidth=0.9)
    for index, (label, ratio) in enumerate(ratios):
        color = "#D55E00" if ratio > 1.0 else "#0072B2"
        fourth.plot([min(1.0, ratio), max(1.0, ratio)], [index, index], color="0.65", linewidth=0.8)
        fourth.scatter(ratio, index, color=color, edgecolor="black", linewidth=0.35, s=28, zorder=3)
    fourth.set_yticks(y, [label for label, _ in ratios])
    fourth.invert_yaxis()
    ratio_min = float(np.min(ratio_values))
    ratio_max = float(np.max(ratio_values))
    comparison_min = min(1.0, ratio_min)
    comparison_max = max(1.0, ratio_max)
    if comparison_max / comparison_min >= 10.0:
        fourth.set_xscale("log")
        fourth.set_xlim(comparison_min / 1.15, comparison_max * 1.15)
    else:
        span = max(comparison_max - comparison_min, 0.1)
        fourth.set_xlim(max(0.0, comparison_min - 0.12 * span), comparison_max + 0.12 * span)
    fourth.set_xlabel(r"p95 error ratio, seed303 / seed202")
    panel_label(fourth, 3)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / "hccb_p418_cross_packing_results.pdf"
    svg = output / "hccb_p418_cross_packing_results.svg"
    png = output / "hccb_p418_cross_packing_results.png"
    figure.savefig(pdf)
    figure.savefig(svg)
    figure.savefig(png, dpi=600)
    plt.close(figure)

    record = {
        "status": "complete_formal_p418_cross_packing_figure",
        "development_packing_seed": 202,
        "final_unseen_packing_seed": 303,
        "selected_architecture": selected,
        "seed303_fields_read_during_selection": False,
        "architectures_compared_on_seed202": list(ARCHITECTURES),
        "conditions_per_packing": 9,
        "seed303_to_seed202_p95_ratios": {
            metric: ratio for (metric, _), (_, ratio) in zip(TRANSFER_METRICS, ratios)
        },
        "development_summary": str(args.development_summary.resolve()),
        "selection_file": str(args.selection.resolve()),
        "final_summary": str(args.final_summary.resolve()),
        "pdf": str(pdf),
        "svg": str(svg),
        "png": str(png),
        "composite_score_used": False,
        "new_physical_parameter_values_added": [],
    }
    summary_path = output / "hccb_p418_cross_packing_results.json"
    summary_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
