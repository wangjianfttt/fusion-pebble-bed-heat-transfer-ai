#!/usr/bin/env python3
"""Build the external thermal and hydraulic comparison used by the manuscript."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ijhmt_figure_style import apply_ijhmt_style


ROOT = Path(__file__).resolve().parents[1]
PREMUX = ROOT / "results/apd006_premux_chapter6_five_levels/sensor_residuals.csv"
TESOMEX = ROOT / "results/apd006_tesomex_1d_transient_baseline/profile_comparison.csv"
HCPB_NU = ROOT / "data/apd006_hcpb_pin_experiment_2023/classical_baseline.csv"
HCPB_SUMMARY = ROOT / "data/apd006_hcpb_pin_experiment_2023/classical_baseline_summary.json"
PRESSURE = ROOT / "results/independent_pressure_drop_comparison/comparison.csv"
OUTPUT = ROOT / "results/hccb_heat_ai_external_evidence"
FIGURE = ROOT / "figures/hccb_heat_ai_external_evidence.pdf"
FIGURE_SVG = ROOT / "figures/hccb_heat_ai_external_evidence.svg"
FIGURE_PNG = ROOT / "figures/hccb_heat_ai_external_evidence.png"
FIGURE_SIZE_INCH = (5.40, 4.38)

PREMUX_BRANCH = "medium_edge_insulation/flow_origin_66mm"
POWER_ORDER = ["VLO", "LO", "MED", "HI", "VHI"]
POWER_COLORS = {
    "VLO": "#56B4E9",
    "LO": "#009E73",
    "MED": "#E69F00",
    "HI": "#D55E00",
    "VHI": "#CC79A7",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    error = predicted - observed
    return {
        "n": int(error.size),
        "bias": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs": float(np.max(np.abs(error))),
    }


def foust_christian_nu(reynolds: np.ndarray, prandtl: float, diameter_ratio: float) -> np.ndarray:
    return 0.04 * diameter_ratio * reynolds**0.8 * prandtl**0.4 / (diameter_ratio + 1.0) ** 0.2


def configure_axes(ax: plt.Axes) -> None:
    ax.tick_params(direction="in", top=True, right=True, which="both", width=0.75)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)


def build() -> dict[str, object]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)

    premux_all_rows = read_csv(PREMUX)
    premux_branch_rows = [
        row
        for row in premux_all_rows
        if row["model_branch"] == PREMUX_BRANCH
    ]
    premux_rows = [row for row in premux_branch_rows if row["anomaly"].lower() != "true"]
    premux_excluded = [row for row in premux_branch_rows if row["anomaly"].lower() == "true"]
    premux_power_scope = {
        power: {
            "total": sum(row["power_level"] == power for row in premux_branch_rows),
            "used": sum(row["power_level"] == power for row in premux_rows),
            "excluded_anomalous": sum(row["power_level"] == power for row in premux_excluded),
        }
        for power in POWER_ORDER
    }
    if len(premux_branch_rows) != 70 or len(premux_rows) != 65 or len(premux_excluded) != 5:
        raise ValueError("PREMUX five-power comparison must contain 70 selected-branch points, including 5 anomalies")
    if any(scope != {"total": 14, "used": 13, "excluded_anomalous": 1} for scope in premux_power_scope.values()):
        raise ValueError("PREMUX five-power comparison must retain 13 of 14 thermocouples at every power level")
    premux_observed = np.asarray([float(row["experimental_C"]) for row in premux_rows])
    premux_predicted = np.asarray([float(row["model_C"]) for row in premux_rows])
    premux_metrics = metrics(premux_observed, premux_predicted)
    premux_branch_metrics: dict[str, dict[str, float | int]] = {}
    for branch in sorted({row["model_branch"] for row in premux_all_rows}):
        rows = [
            row
            for row in premux_all_rows
            if row["model_branch"] == branch and row["anomaly"].lower() != "true"
        ]
        premux_branch_metrics[branch] = metrics(
            np.asarray([float(row["experimental_C"]) for row in rows]),
            np.asarray([float(row["model_C"]) for row in rows]),
        )
    selected_by_rmse = min(premux_branch_metrics, key=lambda branch: float(premux_branch_metrics[branch]["rmse"]))
    if selected_by_rmse != PREMUX_BRANCH:
        raise ValueError("displayed PREMUX branch is not the lowest-RMSE member of the four-branch comparison")

    tesomex_rows = read_csv(TESOMEX)
    tesomex_metrics: dict[str, dict[str, float | int]] = {}
    for snapshot in ("A", "B"):
        rows = [row for row in tesomex_rows if row["snapshot"] == snapshot]
        tesomex_metrics[snapshot] = metrics(
            np.asarray([float(row["observed_C"]) for row in rows]),
            np.asarray([float(row["predicted_C"]) for row in rows]),
        )

    nu_rows = read_csv(HCPB_NU)
    nu_observed = np.asarray([float(row["nu_experiment"]) for row in nu_rows])
    nu_predicted = np.asarray([float(row["nu_foust_christian"]) for row in nu_rows])
    nu_relative = np.abs(nu_predicted - nu_observed) / nu_observed * 100.0
    nu_metrics = {
        **metrics(nu_observed, nu_predicted),
        "mean_absolute_relative_error_percent": float(np.mean(nu_relative)),
        "points_inside_published_uncertainty": int(
            sum(row["inside_published_nu_uncertainty"] == "True" for row in nu_rows)
        ),
    }

    pressure_rows = [row for row in read_csv(PRESSURE) if row["comparison_role"] == "target_diameter_hydrodynamics"]
    pressure_observed = np.asarray([float(row["measured_pressure_gradient_kPa_m"]) for row in pressure_rows])
    pressure_predicted = np.asarray([float(row["predicted_pressure_gradient_kPa_m"]) for row in pressure_rows])
    pressure_absolute_relative = np.abs(pressure_predicted - pressure_observed) / pressure_observed * 100.0
    pressure_metrics = {
        **metrics(pressure_observed, pressure_predicted),
        "median_absolute_relative_error_percent": float(np.median(pressure_absolute_relative)),
        "max_absolute_relative_error_percent": float(np.max(pressure_absolute_relative)),
    }

    apply_ijhmt_style(
        font_size=8.0,
        label_size=8.4,
        tick_size=7.3,
        legend_size=6.9,
        axis_width=0.75,
    )
    # Match the 390 pt elsarticle preprint width.  This preserves the intended
    # journal-scale typography when the PDF is inserted at text width.
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_INCH)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.10, top=0.95, wspace=0.28, hspace=0.30)

    ax = axes[0, 0]
    for power in POWER_ORDER:
        rows = [row for row in premux_rows if row["power_level"] == power]
        observed = np.asarray([float(row["experimental_C"]) for row in rows])
        predicted = np.asarray([float(row["model_C"]) for row in rows])
        uncertainty = np.asarray([float(row["experimental_errorbar_half_width_C"]) for row in rows])
        ax.errorbar(
            observed,
            predicted,
            xerr=uncertainty,
            fmt="o",
            ms=3.5,
            capsize=1.5,
            elinewidth=0.7,
            color=POWER_COLORS[power],
            label=power,
        )
    lower = min(float(premux_observed.min()), float(premux_predicted.min())) - 25.0
    upper = max(float(premux_observed.max()), float(premux_predicted.max())) + 25.0
    ax.plot([lower, upper], [lower, upper], "--", color="0.2", lw=1.0)
    ax.set(xlim=(lower, upper), ylim=(lower, upper), xlabel="PREMUX thermocouple ($^\circ$C)", ylabel="2D prediction ($^\circ$C)")
    branch_rmse = [float(item["rmse"]) for item in premux_branch_metrics.values()]
    ax.text(
        0.04,
        0.94,
        f"selected branch: RMSE = {premux_metrics['rmse']:.1f} $^\circ$C\n"
        f"four-branch range: {min(branch_rmse):.1f}--{max(branch_rmse):.1f} $^\circ$C",
        transform=ax.transAxes,
        va="top",
    )
    ax.legend(frameon=False, ncol=2, loc="lower right", columnspacing=0.8, handletextpad=0.3)

    ax = axes[0, 1]
    time_styles = {"A": ("#D55E00", "65.0 min", "o"), "B": ("#0072B2", "78.5 min", "s")}
    for snapshot in ("A", "B"):
        color, label, marker = time_styles[snapshot]
        rows = sorted(
            [row for row in tesomex_rows if row["snapshot"] == snapshot],
            key=lambda row: float(row["position_mm"]),
        )
        position = np.asarray([float(row["position_mm"]) for row in rows])
        observed = np.asarray([float(row["observed_C"]) for row in rows])
        predicted = np.asarray([float(row["predicted_C"]) for row in rows])
        ax.scatter(
            position,
            observed,
            s=24,
            marker=marker,
            facecolors="white",
            edgecolors=color,
            linewidths=1.1,
            label=f"TESOMEX experiment, {label}",
        )
        ax.plot(position, predicted, color=color, ls="--", label=f"1D control, {label}")
    ax.set(xlabel="radial position (mm)", ylabel="temperature ($^\circ$C)")
    ax.legend(frameon=False, loc="best")

    ax = axes[1, 0]
    reynolds = np.asarray([float(row["re_mean_digitized"]) for row in nu_rows])
    uncertainty_minus = np.asarray([float(row["nu_uncertainty_minus"]) for row in nu_rows])
    uncertainty_plus = np.asarray([float(row["nu_uncertainty_plus"]) for row in nu_rows])
    with HCPB_SUMMARY.open(encoding="utf-8") as handle:
        hcpb_summary = json.load(handle)
    prandtl = float(hcpb_summary["nist_helium_properties"]["prandtl"])
    diameter_ratio = float(hcpb_summary["geometry"]["annular_diameter_ratio"])
    reynolds_curve = np.linspace(reynolds.min() * 0.92, reynolds.max() * 1.05, 240)
    ax.errorbar(
        reynolds,
        nu_observed,
        yerr=np.vstack([uncertainty_minus, uncertainty_plus]),
        fmt="o",
        ms=4.2,
        capsize=2.0,
        color="#0072B2",
        label="HELOKA experiment (Abou-Sena et al.)",
    )
    ax.plot(
        reynolds_curve,
        foust_christian_nu(reynolds_curve, prandtl, diameter_ratio),
        color="#D55E00",
        label="Foust-Christian correlation",
    )
    ax.set(xlabel="Reynolds number", ylabel="Nusselt number")
    ax.text(0.04, 0.94, f"4/4 within uncertainty; MARE = {nu_metrics['mean_absolute_relative_error_percent']:.1f}%", transform=ax.transAxes, va="top")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1, 1]
    identity_low = min(float(pressure_observed.min()), float(pressure_predicted.min())) * 0.82
    identity_high = max(float(pressure_observed.max()), float(pressure_predicted.max())) * 1.2
    ax.plot(
        [identity_low, identity_high],
        [identity_low, identity_high],
        "--",
        color="0.2",
        lw=1.0,
        label="1:1",
    )
    ax.scatter(
        pressure_observed,
        pressure_predicted,
        s=28,
        color="#009E73",
        edgecolor="black",
        linewidth=0.45,
        label="Cheng data / Liu correlation",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(
        xlim=(identity_low, identity_high),
        ylim=(identity_low, identity_high),
        xlabel="measured $\Delta p/L$ (kPa m$^{-1}$)",
        ylabel="correlation $\Delta p/L$ (kPa m$^{-1}$)",
    )
    ax.text(
        0.04,
        0.94,
        f"6 points; median error = {pressure_metrics['median_absolute_relative_error_percent']:.1f}%",
        transform=ax.transAxes,
        va="top",
    )
    ax.legend(frameon=False, loc="lower right")

    for label, ax in zip("abcd", axes.flat):
        configure_axes(ax)
        ax.text(-0.12, 1.02, f"({label})", transform=ax.transAxes, fontsize=9.0, fontweight="bold")

    raw_pdf = OUTPUT / "external_evidence_vector_raw.pdf"
    canvas_bounds = fig.bbox_inches
    fig.savefig(raw_pdf, bbox_inches=canvas_bounds)
    fig.savefig(FIGURE_SVG, bbox_inches=canvas_bounds)
    fig.savefig(FIGURE_PNG, dpi=600, bbox_inches=canvas_bounds)
    shutil.copy2(FIGURE_SVG, OUTPUT / "external_evidence.svg")
    shutil.copy2(FIGURE_PNG, OUTPUT / "external_evidence.png")
    plt.close(fig)
    ghostscript = shutil.which("gs")
    if ghostscript:
        subprocess.run(
            [
                ghostscript,
                "-q",
                "-dSAFER",
                "-dBATCH",
                "-dNOPAUSE",
                "-sDEVICE=pdfwrite",
                "-dCompatibilityLevel=1.4",
                "-dPDFSETTINGS=/prepress",
                f"-sOutputFile={FIGURE}",
                str(raw_pdf),
            ],
            check=True,
        )
    else:
        shutil.copy2(raw_pdf, FIGURE)

    summary: dict[str, object] = {
        "status": "external_thermal_hydraulic_comparison_complete",
        "premux": {
            "source": "Hernandez 2017, DOI 10.5445/IR/1000070791",
            "source_data_path": str(PREMUX.relative_to(ROOT)),
            "scope_name": "five_power_internal_thermocouple_comparison",
            "model_branch": PREMUX_BRANCH,
            "selected_branch_total_points": len(premux_branch_rows),
            "excluded_anomalous_points": len(premux_excluded),
            "points_by_power": premux_power_scope,
            "coolant_endpoint_role": "boundary input; excluded from validation metrics",
            "parameter_fitting_to_thermocouples": False,
            "branch_selection_used_thermocouple_rmse": True,
            "branch_selection_rule": "lowest aggregate RMSE among four precomputed coordinate/insulation branches",
            "branch_metrics": premux_branch_metrics,
            "relation_to_vhi_only_result": (
                "This 65-point five-power comparison is the manuscript result. "
                "The separate results/hccb_heat_ai_external_experiments PREMUX entry is a VHI-only diagnostic comparison."
            ),
            **premux_metrics,
        },
        "tesomex": {
            "source": "Mohammed 2018, https://escholarship.org/uc/item/8bw768p3",
            "source_data_path": str(TESOMEX.relative_to(ROOT)),
            "scope_name": "two_radial_temperature_snapshots",
            "parameter_fitting": False,
            "snapshot_A": tesomex_metrics["A"],
            "snapshot_B": tesomex_metrics["B"],
        },
        "hcpb_annulus": {
            "source": "Abou-Sena et al. 2023, DOI 10.3390/jne4010002",
            "source_data_path": str(HCPB_NU.relative_to(ROOT)),
            "scope_name": "four_annular_channel_nusselt_points",
            **nu_metrics,
        },
        "fixed_bed_pressure": {
            "experiment_source": "Cheng et al. 2024, DOI 10.3390/en17061309",
            "correlation_source": "DOI 10.1016/j.fusengdes.2024.114434",
            "source_data_path": str(PRESSURE.relative_to(ROOT)),
            "scope_name": "six_one_millimetre_fixed_bed_pressure_gradients",
            **pressure_metrics,
        },
        "comparison_scope_file": "results/hccb_heat_ai_external_evidence/comparison_scope.csv",
        "figure_size_inch": list(FIGURE_SIZE_INCH),
        "premux_branch_metrics_file": "results/hccb_heat_ai_external_evidence/premux_branch_metrics.csv",
        "figures": {
            "pdf": str(FIGURE.relative_to(ROOT)),
            "svg": str(FIGURE_SVG.relative_to(ROOT)),
            "png_600dpi": str(FIGURE_PNG.relative_to(ROOT)),
        },
        "use_in_p418_training": False,
        "interpretation": (
            "The four comparisons check external temperature, heat-transfer and hydraulic scales. "
            "They do not provide a pore-resolved P418 field and are not mixed into P418 training."
        ),
        "new_physical_parameters": [],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rows = [
        {
            "experiment": "PREMUX",
            "quantity": "65 internal steady temperatures",
            "comparison": "literature-constrained 2D model",
            "metric": "RMSE_degC",
            "value": f"{premux_metrics['rmse']:.6f}",
        },
        {
            "experiment": "TESOMEX A",
            "quantity": "7 radial temperatures at 65.0 min",
            "comparison": "1D nonlinear conduction",
            "metric": "RMSE_degC",
            "value": f"{tesomex_metrics['A']['rmse']:.6f}",
        },
        {
            "experiment": "TESOMEX B",
            "quantity": "7 radial temperatures at 78.5 min",
            "comparison": "1D nonlinear conduction",
            "metric": "RMSE_degC",
            "value": f"{tesomex_metrics['B']['rmse']:.6f}",
        },
        {
            "experiment": "HELOKA HCPB pin",
            "quantity": "4 annular-channel Nusselt numbers",
            "comparison": "Foust-Christian correlation",
            "metric": "mean_absolute_relative_error_percent",
            "value": f"{nu_metrics['mean_absolute_relative_error_percent']:.6f}",
        },
        {
            "experiment": "1 mm fixed bed",
            "quantity": "6 helium pressure gradients",
            "comparison": "published unitary-bed correlation",
            "metric": "median_absolute_relative_error_percent",
            "value": f"{pressure_metrics['median_absolute_relative_error_percent']:.6f}",
        },
    ]
    with (OUTPUT / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    premux_branch_metric_rows = [
        {
            "model_branch": branch,
            "n": values["n"],
            "bias_degC": f"{float(values['bias']):.12f}",
            "rmse_degC": f"{float(values['rmse']):.12f}",
            "mae_degC": f"{float(values['mae']):.12f}",
            "max_abs_degC": f"{float(values['max_abs']):.12f}",
            "selected_for_figure": branch == PREMUX_BRANCH,
        }
        for branch, values in premux_branch_metrics.items()
    ]
    with (OUTPUT / "premux_branch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(premux_branch_metric_rows[0]))
        writer.writeheader()
        writer.writerows(premux_branch_metric_rows)

    scope_rows = [
        {
            "comparison_id": "PREMUX_five_power",
            "apparatus": "PREMUX",
            "material": "Li4SiO4 pebble bed",
            "measured_quantity": "internal steady thermocouple temperature",
            "model_or_correlation": "literature-constrained 2D reconstruction",
            "total_points_in_selected_scope": len(premux_branch_rows),
            "points_used": len(premux_rows),
            "points_excluded": len(premux_excluded),
            "use_in_p418_training": False,
            "role_in_paper": "branch-selected internal-temperature comparison on the PREMUX apparatus",
            "source_reference": "Hernandez 2017, DOI 10.5445/IR/1000070791",
            "source_data_path": str(PREMUX.relative_to(ROOT)),
            "scope_note": (
                "five power levels; displayed branch has the lowest 65-point RMSE among four precomputed branches; "
                "distinct from the separate VHI-only 13-point diagnostic result"
            ),
        },
        {
            "comparison_id": "TESOMEX_two_snapshots",
            "apparatus": "TESOMEX",
            "material": "Li2TiO3 pebble bed",
            "measured_quantity": "radial transient thermocouple temperature",
            "model_or_correlation": "no-fit 1D nonlinear conduction control",
            "total_points_in_selected_scope": len(tesomex_rows),
            "points_used": len(tesomex_rows),
            "points_excluded": 0,
            "use_in_p418_training": False,
            "role_in_paper": "external transient radial-temperature comparison on the TESOMEX apparatus",
            "source_reference": "Mohammed 2018, https://escholarship.org/uc/item/8bw768p3",
            "source_data_path": str(TESOMEX.relative_to(ROOT)),
            "scope_note": "seven points at each of 65.0 and 78.5 min",
        },
        {
            "comparison_id": "HELOKA_HCPB_pin",
            "apparatus": "HELOKA HCPB fuel-breeder pin",
            "material": "helium annular channel around breeder pin",
            "measured_quantity": "Nusselt number",
            "model_or_correlation": "Foust-Christian annulus correlation",
            "total_points_in_selected_scope": len(nu_rows),
            "points_used": len(nu_rows),
            "points_excluded": 0,
            "use_in_p418_training": False,
            "role_in_paper": "external convective heat-transfer scale comparison",
            "source_reference": "Abou-Sena et al. 2023, DOI 10.3390/jne4010002",
            "source_data_path": str(HCPB_NU.relative_to(ROOT)),
            "scope_note": "four published Nu-Re points with reported uncertainty",
        },
        {
            "comparison_id": "fixed_bed_pressure_1mm",
            "apparatus": "1 mm helium fixed bed",
            "material": "1 mm steel spheres",
            "measured_quantity": "pressure gradient",
            "model_or_correlation": "published unitary-bed correlation",
            "total_points_in_selected_scope": len(pressure_rows),
            "points_used": len(pressure_rows),
            "points_excluded": 0,
            "use_in_p418_training": False,
            "role_in_paper": "external hydraulic scale comparison",
            "source_reference": "Cheng et al. 2024 and Liu et al. 2024",
            "source_data_path": str(PRESSURE.relative_to(ROOT)),
            "scope_note": "only target_diameter_hydrodynamics rows; 5 mm particle-size contrast is excluded",
        },
    ]
    with (OUTPUT / "comparison_scope.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scope_rows[0]))
        writer.writeheader()
        writer.writerows(scope_rows)
    return summary


def main() -> int:
    print(json.dumps(build(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
