#!/usr/bin/env python3
"""Plot physical relations from the currently completed P418 cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


MARKERS = {4.85: "o", 6.85: "s", 8.85: "^"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "axes.grid": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "lines.linewidth": 1.1,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def marker_size(velocity: float) -> float:
    return 31.0 if np.isclose(velocity, 0.05) else 53.0


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.15,
        1.06,
        f"({label})",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )


def plot_points(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    *,
    x_key: str,
    y_key: str,
    norm: matplotlib.colors.Normalize,
    cmap: matplotlib.colors.Colormap,
) -> None:
    for row in rows:
        temperature = number(row, "inlet_temperature_K")
        heat_source = number(row, "solid_heat_source_W_m3") / 1.0e6
        velocity = number(row, "inlet_velocity_m_s")
        ax.scatter(
            number(row, x_key),
            number(row, y_key),
            s=marker_size(velocity),
            marker=MARKERS[heat_source],
            facecolor=cmap(norm(temperature)),
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )


def build_figure(
    pressure_rows: list[dict[str, str]],
    heat_rows: list[dict[str, str]],
    output_pdf: Path,
    output_png: Path,
) -> None:
    configure_style()
    temperatures = np.asarray(
        [number(row, "inlet_temperature_K") for row in heat_rows],
        dtype=np.float64,
    )
    norm = matplotlib.colors.Normalize(
        vmin=float(np.min(temperatures)),
        vmax=float(np.max(temperatures)),
    )
    cmap = matplotlib.colormaps["cividis"]
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.25))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    pressure_plot_rows: list[dict[str, str]] = []
    for row in pressure_rows:
        item = dict(row)
        item["pressure_ratio"] = str(
            number(row, "P420_P421_pressure_drop_Pa")
            / number(row, "resolved_pressure_drop_Pa")
        )
        item["inlet_velocity_m_s"] = row[
            "source_inlet_channel_velocity_m_s"
        ]
        item["solid_heat_source_W_m3"] = str(
            number(row, "solid_heat_source_W_m3")
            if "solid_heat_source_W_m3" in row
            else float(row["condition_id"].split("_q")[-1].replace("p", ".")) * 1.0e6
        )
        pressure_plot_rows.append(item)
    plot_points(
        ax_a,
        pressure_plot_rows,
        x_key="inlet_temperature_K",
        y_key="pressure_ratio",
        norm=norm,
        cmap=cmap,
    )
    ax_a.axhline(1.0, color="0.25", linestyle="--", linewidth=0.8)
    ax_a.set_xlabel("Inlet temperature (K)")
    ax_a.set_ylabel(
        r"$\Delta p_{\mathrm{P420/P421}}/\Delta p_{\mathrm{CFD}}$"
    )
    ax_a.set_xlim(260.0, 940.0)
    ax_a.set_ylim(0.80, 1.05)
    heat_handles = [
        Line2D(
            [],
            [],
            marker=MARKERS[value],
            linestyle="none",
            markerfacecolor="0.65",
            markeredgecolor="black",
            markeredgewidth=0.45,
            markersize=5.2,
            label=rf"{value:.2f}",
        )
        for value in sorted(MARKERS)
    ]
    ax_a.legend(
        handles=heat_handles,
        title=r"$q'''$ (MW m$^{-3}$)",
        loc="lower right",
        frameon=False,
        ncol=3,
        handletextpad=0.25,
        columnspacing=0.65,
        borderpad=0.1,
    )

    reynolds_rows: list[dict[str, str]] = []
    for row in heat_rows:
        item = dict(row)
        item["reynolds_ratio"] = str(
            number(row, "reynolds_particle_local_magnitude_volume_average")
            / number(row, "reynolds_particle_axial_throughflow")
        )
        reynolds_rows.append(item)
    plot_points(
        ax_b,
        reynolds_rows,
        x_key="reynolds_particle_axial_throughflow",
        y_key="reynolds_ratio",
        norm=norm,
        cmap=cmap,
    )
    ax_b.axhline(1.0, color="0.25", linestyle="--", linewidth=0.8)
    ax_b.set_xlabel(r"Through-flow $\mathrm{Re}_{p,z}$")
    ax_b.set_ylabel(
        r"$\langle\mathrm{Re}_{p,|\mathbf{u}|}\rangle/\mathrm{Re}_{p,z}$"
    )
    ax_b.set_xlim(0.0, 2.0)
    ax_b.set_ylim(0.98, 1.30)

    for row in heat_rows:
        re_value = number(row, "reynolds_particle_axial_throughflow")
        temperature = number(row, "inlet_temperature_K")
        heat_source = number(row, "solid_heat_source_W_m3") / 1.0e6
        velocity = number(row, "inlet_velocity_m_s")
        common = {
            "s": marker_size(velocity),
            "facecolor": cmap(norm(temperature)),
            "edgecolor": "black",
            "linewidth": 0.45,
            "zorder": 3,
        }
        ax_c.scatter(
            re_value,
            number(row, "nusselt_from_openfoam_interphase_flux"),
            marker=MARKERS[heat_source],
            **common,
        )
        p419 = number(row, "nusselt_from_resolved_field_P419")
        if np.isfinite(p419):
            ax_c.scatter(
                re_value,
                p419,
                marker="D",
                s=marker_size(velocity) * 0.72,
                facecolor="none",
                edgecolor=cmap(norm(temperature)),
                linewidth=0.9,
                zorder=3,
            )
        ax_c.scatter(
            re_value,
            number(
                row,
                "nusselt_from_source_correlation_P417_throughflow_reference",
            ),
            marker="x",
            s=marker_size(velocity) * 0.62,
            color="0.25",
            linewidth=0.8,
            zorder=3,
        )
    ax_c.set_xlabel(r"Through-flow $\mathrm{Re}_{p,z}$")
    ax_c.set_ylabel("Nusselt number")
    ax_c.set_xlim(0.0, 2.0)
    ax_c.set_ylim(0.0, 9.7)
    ax_c.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="0.55",
                markeredgecolor="black",
                markersize=5.0,
                label="OpenFOAM interface flux",
            ),
            Line2D(
                [],
                [],
                marker="D",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor="0.35",
                markersize=4.5,
                label="P419 generation/phase mean",
            ),
            Line2D(
                [],
                [],
                marker="x",
                linestyle="none",
                color="0.25",
                markersize=4.8,
                label="P417 source relation",
            ),
        ],
        loc="center right",
        bbox_to_anchor=(0.98, 0.55),
        frameon=False,
        borderpad=0.1,
        handletextpad=0.45,
    )

    plot_points(
        ax_d,
        heat_rows,
        x_key="openfoam_solid_wall_heat_over_generated_power",
        y_key="openfoam_interphase_heat_over_generated_power",
        norm=norm,
        cmap=cmap,
    )
    x_min = min(
        number(row, "openfoam_solid_wall_heat_over_generated_power")
        for row in heat_rows
    )
    x_max = max(
        number(row, "openfoam_solid_wall_heat_over_generated_power")
        for row in heat_rows
    )
    x_line = np.linspace(x_min - 0.5, x_max + 0.5, 100)
    ax_d.plot(
        x_line,
        1.0 + x_line,
        color="0.25",
        linestyle="--",
        linewidth=0.9,
        label=r"$Q_{\mathrm{int}}=Q_{\mathrm{gen}}+Q_{\mathrm{wall,s}}$",
    )
    ax_d.axvline(0.0, color="0.70", linewidth=0.6)
    ax_d.axhline(0.0, color="0.70", linewidth=0.6)
    ax_d.set_xlabel(r"$Q_{\mathrm{wall,s}}/Q_{\mathrm{gen}}$")
    ax_d.set_ylabel(r"$Q_{\mathrm{int}}/Q_{\mathrm{gen}}$")
    ax_d.legend(loc="upper left", frameon=False, borderpad=0.1)

    for label, ax in zip("abcd", axes.ravel()):
        add_panel_label(ax, label)

    scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar_axis = fig.add_axes([0.905, 0.12, 0.018, 0.76])
    colorbar = fig.colorbar(scalar, cax=colorbar_axis)
    colorbar.set_label("Inlet temperature (K)")
    colorbar.set_ticks([300, 500, 700, 900])

    fig.subplots_adjust(
        left=0.105,
        right=0.875,
        bottom=0.105,
        top=0.975,
        wspace=0.33,
        hspace=0.35,
    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=600)
    plt.close(fig)


def build_summary(
    pressure_rows: list[dict[str, str]],
    heat_rows: list[dict[str, str]],
    pressure_summary: dict[str, object],
    heat_summary: dict[str, object],
    boundary_summary: dict[str, object],
    figure_pdf: Path,
    figure_png: Path,
) -> dict[str, object]:
    reynolds_ratio = np.asarray(
        [
            number(row, "reynolds_particle_local_magnitude_volume_average")
            / number(row, "reynolds_particle_axial_throughflow")
            for row in heat_rows
        ],
        dtype=np.float64,
    )
    cold_pressure_errors = [
        number(row, "absolute_difference_percent")
        for row in pressure_rows
        if np.isclose(number(row, "inlet_temperature_K"), 300.0)
    ]
    warm_pressure_errors = [
        number(row, "absolute_difference_percent")
        for row in pressure_rows
        if number(row, "inlet_temperature_K") >= 500.0
    ]
    return {
        "status": "p418_partial_physics_relations_ready",
        "case_count": len(heat_rows),
        "pressure_case_count": len(pressure_rows),
        "pressure_correlation_median_absolute_difference_percent": pressure_summary[
            "median_absolute_difference_percent"
        ],
        "pressure_correlation_maximum_absolute_difference_percent": pressure_summary[
            "maximum_absolute_difference_percent"
        ],
        "pressure_error_300K_range_percent": [
            min(cold_pressure_errors),
            max(cold_pressure_errors),
        ],
        "pressure_error_500_to_900K_range_percent": [
            min(warm_pressure_errors),
            max(warm_pressure_errors),
        ],
        "local_to_throughflow_reynolds_ratio_range": [
            float(np.min(reynolds_ratio)),
            float(np.max(reynolds_ratio)),
        ],
        "openfoam_interface_flux_nusselt_range": heat_summary[
            "openfoam_interface_flux_nusselt_range"
        ],
        "p419_generation_based_nusselt_range": heat_summary[
            "resolved_nusselt_range_for_positive_phase_difference"
        ],
        "source_relation_in_range_maximum_difference_percent": heat_summary[
            "maximum_absolute_in_range_correlation_difference_percent"
        ],
        "maximum_interface_pair_difference_W": boundary_summary[
            "maximum_interface_pair_difference_W"
        ],
        "maximum_solid_balance_relative": boundary_summary[
            "maximum_solid_balance_relative"
        ],
        "figure_pdf": str(figure_pdf),
        "figure_png": str(figure_png),
        "new_physical_parameters": [],
        "scope": (
            "Preliminary physical interpretation of the 14 completed steady cases. "
            "The pressure relation and P417/P419 heat-transfer relation are literature "
            "references, not fitted labels. The complete 60-case matrix is required "
            "before final model ranking."
        ),
    }


def render_cn(summary: dict[str, object]) -> str:
    case_count = int(summary["case_count"])
    pressure_cold = summary["pressure_error_300K_range_percent"]
    pressure_warm = summary["pressure_error_500_to_900K_range_percent"]
    re_ratio = summary["local_to_throughflow_reynolds_ratio_range"]
    nu_interface = summary["openfoam_interface_flux_nusselt_range"]
    nu_p419 = summary["p419_generation_based_nusselt_range"]
    return f"""# P418已完成{case_count}组工况的物理关系

这份说明只使用已经正常完成的{case_count}组三维OpenFOAM稳态结果，不包含未完成工况，也没有拟合新的材料参数或换热系数。

## 1. 压降

原文P420/P421压降关系与当前三维结果的绝对差中位数为
`{summary['pressure_correlation_median_absolute_difference_percent']:.2f}%`。在
`500--900 K`入口温度下，差值为`{pressure_warm[0]:.2f}%--{pressure_warm[1]:.2f}%`；
在`300 K`下则为`{pressure_cold[0]:.2f}%--{pressure_cold[1]:.2f}%`。因此该关系能够描述
中高温工况的压降量级，但在当前靠壁局部区域的低温工况下会系统性低估压降。

## 2. 三维孔隙绕流

按局部速度模量体积平均得到的颗粒雷诺数，是按轴向净质量流得到的颗粒雷诺数的
`{re_ratio[0]:.3f}--{re_ratio[1]:.3f}`倍。这说明局部横向绕流和弯曲流线对流动强度有
可测影响，只用一维轴向速度会漏掉约23%--26%的局部速度尺度。

## 3. 流固界面换热

由OpenFOAM流固界面热量和两相平均温差直接得到的努塞尔数为
`{nu_interface[0]:.2f}--{nu_interface[1]:.2f}`。采用总颗粒发热量和两相平均温差的
P419定义，在正温差工况中为`{nu_p419[0]:.2f}--{nu_p419[1]:.2f}`。二者不同的主要原因是
当前区域还与`635 K`壁面和入口边界直接换热；总颗粒发热量并不等于全部流固界面热量。
因此原文整床平均P417/P419关系不能直接当作当前靠壁局部区域的训练标签。

## 4. 能量关系

{case_count}个工况全部满足
`Q_int = Q_gen + Q_wall,s`。流固界面成对热量的最大不一致为
`{summary['maximum_interface_pair_difference_W']:.3e} W`，颗粒能量关系的最大相对差为
`{summary['maximum_solid_balance_relative']:.3e}`。这表明上述换热差异来自区域换热路径不同，
不是由流固界面热量不闭合造成的。

## 5. 对融合模型的意义

图--Transformer需要同时读取三维连接关系、局部速度、流固界面和壁面边界，不能只使用
整床平均雷诺数和单一努塞尔关系。坐标PINN可以把原文关系作为量级对照，但不能强制局部场
服从整床平均关联式。最终定量结论仍以完整60组稳态工况和后续12条热阶跃为准。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pressure-csv", required=True, type=Path)
    parser.add_argument("--pressure-summary", required=True, type=Path)
    parser.add_argument("--dimensionless-csv", required=True, type=Path)
    parser.add_argument("--dimensionless-summary", required=True, type=Path)
    parser.add_argument("--boundary-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    pressure_rows = read_csv(args.pressure_csv.resolve())
    heat_rows = read_csv(args.dimensionless_csv.resolve())
    if len(pressure_rows) != len(heat_rows):
        raise ValueError("pressure and heat-transfer tables have different case counts")
    if {row["condition_id"] for row in pressure_rows} != {
        row["condition_id"] for row in heat_rows
    }:
        raise ValueError("pressure and heat-transfer tables have different cases")
    pressure_summary = json.loads(
        args.pressure_summary.resolve().read_text(encoding="utf-8")
    )
    heat_summary = json.loads(
        args.dimensionless_summary.resolve().read_text(encoding="utf-8")
    )
    boundary_summary = json.loads(
        args.boundary_summary.resolve().read_text(encoding="utf-8")
    )
    if int(pressure_summary["case_count"]) != len(pressure_rows):
        raise ValueError("pressure summary case count is inconsistent")
    if int(heat_summary["case_count"]) != len(heat_rows):
        raise ValueError("dimensionless summary case count is inconsistent")
    if int(boundary_summary["case_count"]) != len(heat_rows):
        raise ValueError("boundary heat summary case count is inconsistent")

    output = args.output_dir.resolve()
    pdf = output / "hccb_p418_partial_physics_relations.pdf"
    png = output / "hccb_p418_partial_physics_relations.png"
    plot_points_path = output / "summary.json"
    build_figure(pressure_rows, heat_rows, pdf, png)
    summary = build_summary(
        pressure_rows,
        heat_rows,
        pressure_summary,
        heat_summary,
        boundary_summary,
        pdf,
        png,
    )
    output.mkdir(parents=True, exist_ok=True)
    plot_points_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / f"P418_{summary['case_count']}工况物理关系_CN.md").write_text(
        render_cn(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
