#!/usr/bin/env python3
"""Plot paired integral responses for valid seed101/seed202 cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ijhmt_figure_style import apply_ijhmt_style


TEMPERATURE_COLORS = {300: "#0072B2", 700: "#E69F00", 900: "#D55E00"}
VELOCITY_MARKERS = {0.05: "o", 0.15: "D", 0.25: "s"}
FIGURE_SIZE_INCH = (5.40, 4.04)


def parse_condition(condition_id: str) -> tuple[float, int, float]:
    parts = condition_id.split("_")
    if len(parts) != 3:
        raise ValueError(f"unexpected condition id: {condition_id}")
    velocity = float(parts[0][1:].replace("p", "."))
    temperature = int(parts[1][1:])
    source = float(parts[2][1:].replace("p", "."))
    return velocity, temperature, source


def read_rows(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("seed comparison contains no accepted cases")
    required = {
        "condition_id",
        "seed101_outlet_temperature_K",
        "seed202_outlet_temperature_K",
        "relative_change_outlet_temperature_K_percent",
        "seed101_maximum_solid_temperature_K",
        "seed202_maximum_solid_temperature_K",
        "relative_change_maximum_solid_temperature_K_percent",
        "seed101_pressure_drop_Pa",
        "seed202_pressure_drop_Pa",
        "relative_change_pressure_drop_Pa_percent",
    }
    if not rows or required - set(rows[0]):
        raise ValueError(f"comparison table lacks columns: {sorted(required - set(rows[0]))}")
    parsed: list[dict[str, float | str]] = []
    for row in rows:
        velocity, temperature, source = parse_condition(row["condition_id"])
        record: dict[str, float | str] = {
            "condition_id": row["condition_id"],
            "velocity": velocity,
            "temperature": float(temperature),
            "source": source,
        }
        for name in required - {"condition_id"}:
            value = float(row[name])
            if not np.isfinite(value):
                raise ValueError(f"non-finite {name} in {row['condition_id']}")
            record[name] = value
        parsed.append(record)
    return parsed


def axis_style(axis: plt.Axes) -> None:
    axis.tick_params(which="both", direction="in", top=True, right=True, width=0.75)
    axis.tick_params(which="major", length=4.0)
    axis.tick_params(which="minor", length=2.0)
    axis.minorticks_on()
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.17,
        1.02,
        label,
        transform=axis.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def parity_panel(
    axis: plt.Axes,
    rows: list[dict[str, float | str]],
    x_name: str,
    y_name: str,
    label: str,
) -> None:
    x_values = np.asarray([float(row[x_name]) for row in rows])
    y_values = np.asarray([float(row[y_name]) for row in rows])
    lower = min(float(x_values.min()), float(y_values.min()))
    upper = max(float(x_values.max()), float(y_values.max()))
    span = max(upper - lower, max(abs(lower), abs(upper), 1.0) * 0.06)
    pad = 0.08 * span
    lower -= pad
    upper += pad
    axis.plot([lower, upper], [lower, upper], color="#4D4D4D", linewidth=0.9, linestyle="--", zorder=1)
    for row in rows:
        temperature = int(float(row["temperature"]))
        velocity = float(row["velocity"])
        axis.scatter(
            float(row[x_name]),
            float(row[y_name]),
            s=36,
            marker=VELOCITY_MARKERS[velocity],
            facecolor=TEMPERATURE_COLORS[temperature],
            edgecolor="black",
            linewidth=0.55,
            zorder=3,
        )
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(rf"seed101 {label}")
    axis.set_ylabel(rf"seed202 {label}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem")
    args = parser.parse_args()

    rows = read_rows(args.comparison_csv.resolve())
    summary = json.loads(args.summary_json.resolve().read_text(encoding="utf-8"))
    if summary.get("accepted_common_case_count") != len(rows):
        raise ValueError("summary and comparison table contain different case counts")
    complete = summary.get("complete_nine_case_comparison") is True
    if complete and len(rows) != 9:
        raise ValueError("complete seed comparison requires nine cases")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    apply_ijhmt_style(
        font_size=7.9,
        label_size=8.2,
        tick_size=7.2,
        legend_size=7.0,
        axis_width=0.75,
    )

    figure, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_INCH)
    figure.subplots_adjust(
        left=0.09,
        right=0.99,
        bottom=0.085,
        top=0.955,
        wspace=0.24,
        hspace=0.24,
    )
    for axis, label in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)")):
        axis_style(axis)
        panel_label(axis, label)

    parity_panel(
        axes[0, 0],
        rows,
        "seed101_outlet_temperature_K",
        "seed202_outlet_temperature_K",
        r"$T_{\mathrm{out}}$ (K)",
    )
    parity_panel(
        axes[0, 1],
        rows,
        "seed101_maximum_solid_temperature_K",
        "seed202_maximum_solid_temperature_K",
        r"$T_{\mathrm{s,max}}$ (K)",
    )
    parity_panel(
        axes[1, 0],
        rows,
        "seed101_pressure_drop_Pa",
        "seed202_pressure_drop_Pa",
        r"$\Delta p$ (Pa)",
    )

    temperature_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=5.5,
            markerfacecolor=TEMPERATURE_COLORS[temperature],
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=f"{temperature} K",
        )
        for temperature in (300, 700, 900)
    ]
    velocity_handles = [
        Line2D(
            [0],
            [0],
            marker=VELOCITY_MARKERS[velocity],
            linestyle="none",
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.6,
            label=rf"$u={velocity:.2f}$ m s$^{{-1}}$",
        )
        for velocity in (0.05, 0.15, 0.25)
    ]
    legend_temperature = axes[0, 0].legend(
        handles=temperature_handles,
        frameon=False,
        loc="upper left",
        borderaxespad=0.35,
        handletextpad=0.4,
        labelspacing=0.3,
    )
    axes[0, 0].add_artist(legend_temperature)
    axes[0, 0].legend(
        handles=velocity_handles,
        frameon=False,
        loc="lower right",
        borderaxespad=0.35,
        handletextpad=0.4,
        labelspacing=0.3,
    )

    axis = axes[1, 1]
    # Match the square plotting boxes used by the three parity panels.
    axis.set_box_aspect(1)
    metric_rows = (
        ("Outlet $T$", "relative_change_outlet_temperature_K_percent", "#0072B2"),
        ("Maximum solid $T$", "relative_change_maximum_solid_temperature_K_percent", "#009E73"),
        ("Pressure drop", "relative_change_pressure_drop_Pa_percent", "#D55E00"),
    )
    offsets = np.linspace(-0.14, 0.14, len(rows))
    for index, (metric_label, column, color) in enumerate(metric_rows):
        values = np.asarray([float(row[column]) for row in rows])
        y_value = len(metric_rows) - index
        axis.scatter(
            values,
            y_value + offsets,
            s=24,
            facecolor=color,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        axis.plot(
            [float(values.min()), float(values.max())],
            [y_value, y_value],
            color=color,
            linewidth=1.25,
            zorder=2,
        )
        axis.scatter(
            float(np.mean(values)),
            y_value,
            marker="D",
            s=38,
            facecolor="white",
            edgecolor=color,
            linewidth=1.15,
            zorder=4,
        )
    axis.axvline(0.0, color="#4D4D4D", linestyle="--", linewidth=0.9, zorder=1)
    axis.set_xscale("symlog", linthresh=1.0, linscale=1.0, base=10)
    axis.set_xlim(-1.0, 22.0)
    axis.set_xticks([-1.0, -0.3, 0.0, 0.3, 1.0, 3.0, 10.0, 20.0])
    axis.set_xticklabels(["-1", "-0.3", "0", "0.3", "1", "3", "10", "20"])
    axis.set_ylim(0.45, 3.55)
    axis.set_yticks([3, 2, 1], [row[0] for row in metric_rows])
    axis.set_xlabel("seed202 relative to seed101 (%)")
    axis.tick_params(axis="y", which="minor", left=False, right=False)
    figure.canvas.draw()
    panel_bounds = [tuple(float(value) for value in axis.get_position().bounds) for axis in axes.flat]
    panel_widths = np.asarray([bounds[2] for bounds in panel_bounds])
    panel_heights = np.asarray([bounds[3] for bounds in panel_bounds])
    if np.ptp(panel_widths) > 1.0e-10 or np.ptp(panel_heights) > 1.0e-10:
        raise ValueError(f"four plotting boxes must have identical dimensions: {panel_bounds}")

    stem = args.output_stem or (
        "hccb_p418_seed202_integral_9"
        if complete
        else f"hccb_p418_seed202_integral_partial_{len(rows)}"
    )
    save_options = {"bbox_inches": figure.bbox_inches}
    figure.savefig(output / f"{stem}.pdf", **save_options)
    figure.savefig(output / f"{stem}.svg", **save_options)
    figure.savefig(output / f"{stem}.png", dpi=600, **save_options)
    plt.close(figure)
    metadata = {
        "status": (
            "complete_p418_seed202_integral_9_figure"
            if complete
            else "partial_p418_seed202_integral_figure"
        ),
        "condition_count": len(rows),
        "figure_size_inch": list(FIGURE_SIZE_INCH),
        "panel_d_xscale": {
            "type": "symlog",
            "linear_threshold_percent": 1.0,
            "purpose": (
                "show sub-percent thermal changes and 14.7--18.0 percent "
                "pressure-drop changes on one axis"
            ),
        },
        "complete_nine_case_comparison": complete,
        "layout": "two columns by two rows",
        "uniform_panel_dimensions": True,
        "panel_bounds_figure_fraction": panel_bounds,
        "subplot_spacing": {"wspace": 0.24, "hspace": 0.24},
        "source_summary": str(args.summary_json.resolve()),
        "source_comparison_csv": str(args.comparison_csv.resolve()),
        "outputs": {
            suffix: str((output / f"{stem}.{suffix}").resolve())
            for suffix in ("pdf", "svg", "png")
        },
        "new_physical_parameters": [],
    }
    (output / f"{stem}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
