#!/usr/bin/env python3
"""Plot the complete 60-condition P418 thermal response without fitted claims.

The figure deliberately summarizes repeated operating points with medians and
observed ranges.  It does not fit a response surface.  The wall-heat reversal
temperature is located only by piecewise-linear interpolation between adjacent
published inlet-temperature levels.
"""

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


VELOCITIES = (0.05, 0.10, 0.15, 0.20, 0.25)
TEMPERATURES = (300.0, 500.0, 700.0, 900.0)
SOURCES = (4.85, 6.85, 8.85)
SOURCE_COLORS = {4.85: "#0072B2", 6.85: "#009E73", 8.85: "#D55E00"}
TEMPERATURE_COLORS = {300.0: "#0072B2", 500.0: "#009E73", 700.0: "#E69F00", 900.0: "#D55E00"}
FIGURE_SIZE_INCH = (5.40, 6.70)
FIGURE_ADJUST = {
    "left": 0.105,
    "right": 0.985,
    "bottom": 0.065,
    "top": 0.916,
    "wspace": 0.33,
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


def read_complete_matrix(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "condition_id",
        "inlet_velocity_m_s",
        "inlet_temperature_K",
        "solid_heat_source_MW_m3",
        "pressure_drop_Pa",
        "outlet_temperature_K",
        "solid_maximum_temperature_K",
        "cooling_wall_heat_over_generated",
        "relative_mass_difference",
        "relative_energy_difference",
    }
    if len(rows) != 60:
        raise ValueError(f"physical-response figure requires all 60 conditions, found {len(rows)}")
    if not rows or required - set(rows[0]):
        raise ValueError(f"physical-response table lacks columns: {sorted(required - set(rows[0]))}")
    parsed = []
    for row in rows:
        record: dict[str, float | str] = {"condition_id": row["condition_id"]}
        for name in required - {"condition_id"}:
            value = float(row[name])
            if not np.isfinite(value):
                raise ValueError(f"non-finite {name} in {row['condition_id']}")
            record[name] = value
        parsed.append(record)
    combinations = {
        (
            float(row["inlet_velocity_m_s"]),
            float(row["inlet_temperature_K"]),
            float(row["solid_heat_source_MW_m3"]),
        )
        for row in parsed
    }
    expected = {(u, temperature, source) for u in VELOCITIES for temperature in TEMPERATURES for source in SOURCES}
    if combinations != expected:
        raise ValueError("physical-response table does not cover the exact P418 5 x 4 x 3 matrix")
    return parsed


def axis_style(axis: plt.Axes) -> None:
    axis.tick_params(which="both", direction="in", top=True, right=True, width=0.75)
    axis.tick_params(which="major", length=4.2)
    axis.tick_params(which="minor", length=2.2)
    axis.minorticks_on()
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)


def values_for(
    lookup: dict[tuple[float, float, float], dict[str, float | str]],
    metric: str,
    *,
    velocity: float | None = None,
    temperature: float | None = None,
    source: float | None = None,
) -> np.ndarray:
    values = []
    for (row_velocity, row_temperature, row_source), row in lookup.items():
        if velocity is not None and row_velocity != velocity:
            continue
        if temperature is not None and row_temperature != temperature:
            continue
        if source is not None and row_source != source:
            continue
        values.append(float(row[metric]))
    return np.asarray(values, dtype=float)


def grouped_range(
    lookup: dict[tuple[float, float, float], dict[str, float | str]],
    metric: str,
    x_values: tuple[float, ...],
    *,
    source: float | None = None,
    temperature: float | None = None,
    transform=lambda value, _temperature: value,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    medians = []
    lower = []
    upper = []
    for x_value in x_values:
        if source is None:
            values = values_for(lookup, metric, velocity=x_value, temperature=temperature)
            reference_temperature = float(temperature)
        else:
            values = values_for(lookup, metric, temperature=x_value, source=source)
            reference_temperature = x_value
        transformed = np.asarray(
            [transform(value, reference_temperature) for value in values],
            dtype=float,
        )
        if transformed.size == 0:
            raise ValueError(f"no values found for {metric} at {x_value}")
        medians.append(float(np.median(transformed)))
        lower.append(float(np.min(transformed)))
        upper.append(float(np.max(transformed)))
    return np.asarray(medians), np.asarray(lower), np.asarray(upper)


def reversal_temperature(
    lookup: dict[tuple[float, float, float], dict[str, float | str]],
    velocity: float,
    source: float,
) -> tuple[float, str]:
    ratios = np.asarray(
        [
            float(lookup[(velocity, temperature, source)]["cooling_wall_heat_over_generated"])
            for temperature in TEMPERATURES
        ],
        dtype=float,
    )
    temperatures = np.asarray(TEMPERATURES, dtype=float)
    exact = np.flatnonzero(np.isclose(ratios, 0.0, rtol=0.0, atol=1.0e-14))
    if exact.size:
        return float(temperatures[int(exact[0])]), "inside"
    for index in range(len(temperatures) - 1):
        first = ratios[index]
        second = ratios[index + 1]
        if first * second < 0.0:
            fraction = -first / (second - first)
            return float(temperatures[index] + fraction * (temperatures[index + 1] - temperatures[index])), "inside"
    if np.all(ratios > 0.0):
        return float(temperatures[-1]), "above"
    if np.all(ratios < 0.0):
        return float(temperatures[0]), "below"
    raise ValueError(f"ambiguous wall-heat reversal for u={velocity}, q={source}: {ratios.tolist()}")


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.16,
        1.06,
        label,
        transform=axis.transAxes,
        fontweight="bold",
        fontsize=8.5,
        ha="left",
        va="bottom",
        clip_on=False,
    )


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.abs(values))
    fractions = np.arange(1, ordered.size + 1, dtype=float) / ordered.size
    return ordered, fractions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_complete_matrix(args.physical_csv.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    lookup = {
        (
            float(row["inlet_velocity_m_s"]),
            float(row["inlet_temperature_K"]),
            float(row["solid_heat_source_MW_m3"]),
        ): row
        for row in rows
    }
    apply_ijhmt_style(
        font_size=7.9,
        label_size=8.2,
        tick_size=7.2,
        legend_size=6.9,
        axis_width=0.75,
    )
    # Generate at the 390 pt elsarticle preprint width so labels retain their
    # declared size instead of being reduced again by LaTeX.  The taller
    # canvas keeps each panel only slightly wider than square.
    figure, axes = plt.subplots(3, 2, figsize=FIGURE_SIZE_INCH)
    figure.subplots_adjust(**FIGURE_ADJUST)
    for axis, label in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")):
        axis_style(axis)
        panel_label(axis, label)

    # (a) Hydraulic response.  Each line is the median over the three heat
    # source levels; the band is their observed range.
    axis = axes[0, 0]
    for temperature in TEMPERATURES:
        median, lower, upper = grouped_range(
            lookup,
            "pressure_drop_Pa",
            VELOCITIES,
            temperature=temperature,
        )
        color = TEMPERATURE_COLORS[temperature]
        axis.fill_between(VELOCITIES, lower, upper, color=color, alpha=0.13, linewidth=0.0)
        axis.plot(
            VELOCITIES,
            median,
            color=color,
            marker="o",
            markersize=3.7,
            linewidth=1.25,
            label=f"{temperature:.0f} K",
        )
    axis.set_xlabel("Inlet velocity (m s$^{-1}$)")
    axis.set_ylabel("Pressure drop (Pa)")
    axis.legend(frameon=False, ncol=2, loc="upper left", columnspacing=0.9, handlelength=1.5)

    # (b)--(d) Thermal response.  Lines are medians over the five velocities;
    # bands show the complete velocity range at each published temperature.
    thermal_panels = (
        (
            axes[0, 1],
            "outlet_temperature_K",
            "Outlet temperature rise (K)",
            lambda value, temperature: value - temperature,
        ),
        (
            axes[1, 0],
            "solid_maximum_temperature_K",
            "Maximum solid temperature (K)",
            lambda value, _temperature: value,
        ),
        (
            axes[1, 1],
            "cooling_wall_heat_over_generated",
            "Wall heat / generated heat",
            lambda value, _temperature: value,
        ),
    )
    source_handles = []
    for axis, metric, ylabel, transform in thermal_panels:
        for source in SOURCES:
            median, lower, upper = grouped_range(
                lookup,
                metric,
                TEMPERATURES,
                source=source,
                transform=transform,
            )
            color = SOURCE_COLORS[source]
            axis.fill_between(TEMPERATURES, lower, upper, color=color, alpha=0.13, linewidth=0.0)
            line, = axis.plot(
                TEMPERATURES,
                median,
                color=color,
                marker="o",
                markersize=3.7,
                linewidth=1.3,
                label=f"{source:.2f}",
            )
            if axis is axes[0, 1]:
                source_handles.append(line)
        axis.set_xlabel("Inlet temperature (K)")
        axis.set_ylabel(ylabel)
    axes[1, 1].axhline(0.0, color="0.25", linestyle="--", linewidth=0.9, zorder=0)
    axes[0, 1].legend(
        source_handles,
        [f"{source:.2f}" for source in SOURCES],
        title="$q'''$ (MW m$^{-3}$)",
        frameon=False,
        loc="best",
        ncol=3,
        columnspacing=0.8,
        handlelength=1.4,
        title_fontsize=7.5,
    )

    # (e) Temperature at which the signed cooling-wall heat changes direction.
    axis = axes[2, 0]
    reversal_records = []
    for source in SOURCES:
        inside_x = []
        inside_y = []
        for velocity in VELOCITIES:
            temperature, status = reversal_temperature(lookup, velocity, source)
            reversal_records.append(
                {
                    "inlet_velocity_m_s": velocity,
                    "solid_heat_source_MW_m3": source,
                    "reversal_temperature_K": temperature,
                    "status": status,
                }
            )
            marker = "o" if status == "inside" else ("^" if status == "above" else "v")
            axis.plot(
                velocity,
                temperature,
                color=SOURCE_COLORS[source],
                marker=marker,
                markersize=4.2,
                linestyle="none",
            )
            if status == "inside":
                inside_x.append(velocity)
                inside_y.append(temperature)
        if len(inside_x) > 1:
            axis.plot(inside_x, inside_y, color=SOURCE_COLORS[source], linewidth=1.25)
    axis.set_xlabel("Inlet velocity (m s$^{-1}$)")
    axis.set_ylabel("Wall-heat reversal $T_\\mathrm{in}$ (K)")
    axis.set_ylim(TEMPERATURES[0] - 25.0, TEMPERATURES[-1] + 25.0)

    # (f) Numerical conservation over all 60 finite-volume fields.
    axis = axes[2, 1]
    mass_values = np.asarray([float(row["relative_mass_difference"]) for row in rows], dtype=float)
    energy_values = np.asarray([float(row["relative_energy_difference"]) for row in rows], dtype=float)
    positive = np.concatenate((np.abs(mass_values), np.abs(energy_values)))
    positive = positive[positive > 0.0]
    floor = max(float(np.min(positive)) * 0.5, np.finfo(float).tiny) if positive.size else 1.0e-16
    for values, color, label, linestyle in (
        (mass_values, "#0072B2", "Mass", "-"),
        (energy_values, "#D55E00", "Energy", "--"),
    ):
        x_values, fractions = empirical_cdf(values)
        x_values = np.maximum(x_values, floor)
        axis.step(x_values, fractions, where="post", color=color, linestyle=linestyle, linewidth=1.35, label=label)
    axis.set_xscale("log")
    axis.set_xlabel("Absolute relative difference")
    axis.set_ylabel("Cumulative fraction")
    axis.set_ylim(0.0, 1.02)
    axis.legend(frameon=False, loc="lower right")

    pdf = output / "hccb_p418_physical_response.pdf"
    svg = output / "hccb_p418_physical_response.svg"
    png = output / "hccb_p418_physical_response.png"
    canvas_bounds = figure.bbox_inches
    figure.savefig(pdf, bbox_inches=canvas_bounds)
    figure.savefig(svg, bbox_inches=canvas_bounds)
    figure.savefig(png, dpi=600, bbox_inches=canvas_bounds)
    plt.close(figure)
    maximum_mass = max(abs(float(row["relative_mass_difference"])) for row in rows)
    maximum_energy = max(abs(float(row["relative_energy_difference"])) for row in rows)
    summary = {
        "status": "complete_60_condition_physical_response_figure",
        "condition_count": 60,
        "figure_size_inch": list(FIGURE_SIZE_INCH),
        "figure_size_mm": [value * 25.4 for value in FIGURE_SIZE_INCH],
        "panel_width_to_height_ratio": panel_width_to_height_ratio(),
        "source_csv": str(args.physical_csv.resolve()),
        "maximum_relative_mass_difference": maximum_mass,
        "maximum_relative_energy_difference": maximum_energy,
        "pdf": str(pdf),
        "svg": str(svg),
        "png": str(png),
        "fitted_regression_used": False,
        "summary_definition": {
            "pressure_drop": "median and observed range over the three source levels at fixed velocity and inlet temperature",
            "thermal_panels": "median and observed range over the five inlet velocities at fixed source and inlet temperature",
            "wall_heat_reversal": "piecewise-linear zero crossing between adjacent sampled inlet-temperature levels",
            "conservation": "empirical cumulative distributions over all 60 fields",
        },
        "wall_heat_reversal": reversal_records,
        "new_physical_parameter_values_added": [],
    }
    (output / "hccb_p418_physical_response.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
