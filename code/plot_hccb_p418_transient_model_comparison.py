#!/usr/bin/env python3
"""Plot formal P418 held-out thermal trajectories and model/physics trade-offs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from hccb_p418_selected_fixed_flow_chain import (
    selected_chain_record_path,
    selected_model_directories,
    sha256,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ijhmt_figure_style import apply_ijhmt_style


STRICT_SPLIT = "pair_disjoint_stress_test"
ENERGY_METRIC = "projection_aware_volume_weighted_energy_equation_normalized_RMSE"
MODEL_ORDER = (
    "initial_temperature_persistence",
    "dmdc",
    "graph_transformer_data_only",
    "graph_transformer_energy_flux",
    "graph_transformer_factorized_energy_flux",
    "low_rank_residual_correction",
    "diffusion_residual_correction",
)
MODEL_LABELS = {
    "initial_temperature_persistence": "initial-field persistence",
    "dmdc": "DMDc",
    "graph_transformer_data_only": "data-only GT",
    "graph_transformer_energy_flux": "physics GT",
    "graph_transformer_factorized_energy_flux": "factorized GT",
    "low_rank_residual_correction": "POD residual",
    "diffusion_residual_correction": "diffusion",
}
MODEL_COLORS = {
    "initial_temperature_persistence": "#8C564B",
    "dmdc": "#000000",
    "graph_transformer_data_only": "#999999",
    "graph_transformer_energy_flux": "#0072B2",
    "graph_transformer_factorized_energy_flux": "#56B4E9",
    "low_rank_residual_correction": "#009E73",
    "diffusion_residual_correction": "#D55E00",
}
MODEL_MARKERS = {
    "initial_temperature_persistence": "X",
    "dmdc": "o",
    "graph_transformer_data_only": "s",
    "graph_transformer_energy_flux": "^",
    "graph_transformer_factorized_energy_flux": "v",
    "low_rank_residual_correction": "D",
    "diffusion_residual_correction": "P",
}
EVENT_LABELS = {
    "velocity_up_T900_q6p85": "velocity increase",
    "velocity_down_T900_q6p85": "velocity decrease",
    "source_up_u0p15_T700": "heat-source increase",
    "source_down_u0p15_T700": "heat-source decrease",
}
FIGURE_SIZE_INCH = (5.40, 6.70)
FIGURE_ADJUST = {
    "left": 0.105,
    "right": 0.985,
    "bottom": 0.065,
    "top": 0.92,
    "wspace": 0.32,
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


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing formal result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require_formal_split(summary: dict, expected_ids: list[str], label: str) -> None:
    if summary.get("split_name") != STRICT_SPLIT:
        raise ValueError(f"{label} is not the strict endpoint-pair split")
    recorded = summary.get("split_case_ids", {}).get("test", [])
    if list(map(str, recorded)) != expected_ids:
        raise ValueError(f"{label} test trajectories differ from the registered strict split")


def load_npz(path: Path, required: set[str]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing formal prediction file: {path}")
    with np.load(path, allow_pickle=False) as source:
        missing = required.difference(source.files)
        if missing:
            raise ValueError(f"{path} lacks {sorted(missing)}")
        return {name: source[name].copy() for name in required}


def physical_temperature(normalized: np.ndarray, data: dict[str, np.ndarray]) -> np.ndarray:
    node_type = data["node_type"].astype(int)
    mean = data["temperature_mean_K_by_node_type"][node_type]
    scale = data["temperature_std_K_by_node_type"][node_type]
    return normalized[..., 0] * scale[None, None, :] + mean[None, None, :]


def solid_volume_average(values: np.ndarray, data: dict[str, np.ndarray]) -> np.ndarray:
    selected = data["node_type"].astype(int) == 1
    if not np.any(selected):
        raise ValueError("prediction file contains no solid regional nodes")
    volume = data["node_volume_m3"][selected].astype(float)
    if np.any(volume <= 0.0) or not np.all(np.isfinite(volume)):
        raise ValueError("solid regional volumes are invalid")
    return np.sum(values[..., selected] * volume[None, None, :], axis=-1) / volume.sum()


def casewise_curve_rmse(
    prediction: np.ndarray,
    target: np.ndarray,
    sequence_ids: list[str],
) -> dict[str, float]:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("case-wise curve RMSE requires matching case-by-time arrays")
    if prediction.shape[0] != len(sequence_ids):
        raise ValueError("case-wise curve RMSE sequence count is inconsistent")
    values = np.sqrt(np.mean(np.square(prediction - target), axis=1))
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("case-wise curve RMSE contains invalid values")
    return {
        sequence_id: float(value)
        for sequence_id, value in zip(sequence_ids, values)
    }


def split_field_rmse(summary: dict, label: str) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for role in ("train", "validation", "test"):
        metrics = summary.get("metrics", {}).get(role, {})
        role_values = {
            "fluid_temperature_RMSE_K": float(
                metrics["fluid_temperature_RMSE_K"]
            ),
            "solid_temperature_RMSE_K": float(
                metrics["solid_temperature_RMSE_K"]
            ),
        }
        if not all(
            np.isfinite(value) and value >= 0.0 for value in role_values.values()
        ):
            raise ValueError(f"{label} contains invalid {role} field RMSE")
        output[role] = role_values
    return output


def require_matching_prediction_coordinates(
    reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray], label: str
) -> None:
    for name in ("sequence_id", "time_s", "node_type", "node_volume_m3"):
        if not np.array_equal(reference[name], candidate[name]):
            raise ValueError(f"{label} differs from the physics model in {name}")


def validate_formal_time_axes(time_s: np.ndarray) -> None:
    if time_s.ndim != 2 or time_s.shape[0] != 4 or time_s.shape[1] < 2:
        raise ValueError("formal transient comparison requires four complete time axes")
    if not np.all(np.isfinite(time_s)) or np.any(np.diff(time_s, axis=1) <= 0.0):
        raise ValueError("formal trajectory time axes must be finite and strictly increasing")
    if not np.allclose(time_s[:, 0], 0.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("formal held-out trajectories must begin at 0 s")
    if not np.allclose(time_s[:, -1], 300.0, rtol=0.0, atol=1.0e-8):
        raise ValueError("formal held-out trajectories must reach 300 s")


def read_metric_table(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("formal model metric table is empty")
    output: dict[str, dict[str, float]] = {model: {} for model in MODEL_ORDER}
    for row in rows:
        if row["split_name"] != STRICT_SPLIT or row["data_role"] != "test":
            continue
        model = row["model"]
        if model not in output:
            continue
        metric = row["metric"]
        if metric == ENERGY_METRIC:
            output[model]["energy"] = float(row["value"])
        if metric == "solid_temperature_RMSE_K":
            output[model]["temperature"] = float(row["value"])
        if metric == "diffusion_refined_solid_temperature_RMSE_K":
            output[model]["temperature"] = float(row["value"])
    for model, values in output.items():
        if set(values) != {"temperature", "energy"}:
            raise ValueError(f"formal metric table lacks temperature/energy values for {model}")
        if not all(np.isfinite(value) and value >= 0.0 for value in values.values()):
            raise ValueError(f"formal metric values are invalid for {model}")
    return output


def read_speed_table(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    output = {}
    openfoam_times = []
    for row in rows:
        model = row["model"]
        if row["split_name"] == STRICT_SPLIT and model in MODEL_ORDER:
            if model in output:
                raise ValueError(f"formal speed table contains duplicate {model}")
            openfoam_seconds = float(row["openfoam_median_clock_time_s"])
            inference_seconds = float(row["model_inference_seconds_per_curve"])
            speedup = float(row["wall_clock_speedup"])
            if (
                not np.isfinite(openfoam_seconds)
                or not np.isfinite(inference_seconds)
                or not np.isfinite(speedup)
                or min(openfoam_seconds, inference_seconds, speedup) <= 0.0
            ):
                raise ValueError("formal speed table contains invalid timing values")
            if not np.isclose(
                speedup,
                openfoam_seconds / inference_seconds,
                rtol=5.0e-6,
                atol=0.0,
            ):
                raise ValueError(
                    f"formal speedup for {model} is inconsistent with its recorded "
                    "OpenFOAM and inference wall times"
                )
            if not row.get("compute_device", "").strip():
                raise ValueError(f"formal speed table does not record a device for {model}")
            output[model] = speedup
            openfoam_times.append(openfoam_seconds)
    missing = set(MODEL_ORDER).difference(output)
    if missing:
        raise ValueError(f"formal speed table lacks {sorted(missing)}")
    if not all(np.isfinite(value) and value > 0.0 for value in output.values()):
        raise ValueError("formal speed table contains non-positive or non-finite speedup")
    if not np.allclose(openfoam_times, openfoam_times[0], rtol=1.0e-9, atol=0.0):
        raise ValueError("models were compared against different OpenFOAM wall times")
    return output


def axis_style(axis: plt.Axes) -> None:
    axis.tick_params(which="both", direction="in", top=True, right=True, width=0.75)
    axis.tick_params(which="major", length=4)
    axis.tick_params(which="minor", length=2)
    axis.minorticks_on()
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)


def add_panel_label(axis: plt.Axes, index: int) -> None:
    axis.text(
        -0.16,
        1.08,
        f"({chr(ord('a') + index)})",
        transform=axis.transAxes,
        fontsize=8.4,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def annotate_point(
    axis: plt.Axes,
    x: float,
    y: float,
    label: str,
    x_midpoint: float,
    y_midpoint: float,
) -> None:
    """Place a model name toward the plot interior instead of beyond the right spine."""
    if x >= x_midpoint:
        offset = (-4, 3)
        horizontal = "right"
    else:
        offset = (4, 3)
        horizontal = "left"
    if y >= y_midpoint:
        y_offset = -4
        vertical = "top"
    else:
        y_offset = 3
        vertical = "bottom"
    axis.annotate(
        label,
        (x, y),
        xytext=(offset[0], y_offset),
        textcoords="offset points",
        fontsize=6.1,
        ha=horizontal,
        va=vertical,
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir", type=Path, default=root / "results/hccb_p418_physical_steps_12"
    )
    parser.add_argument(
        "--splits", type=Path, default=root / "parameters/hccb_p418_step_response_splits.json"
    )
    parser.add_argument("--output-dir", type=Path, default=root / "figures")
    parser.add_argument("--validation-marker", type=Path)
    parser.add_argument(
        "--allow-preselection-diagnostic",
        action="store_true",
        help=(
            "Render a clearly named diagnostic layout before validation-only model "
            "selection. This mode never writes the formal figure or validation marker."
        ),
    )
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    comparison_dir = result_dir / "model_comparison"
    comparison = load_json(comparison_dir / "summary.json")
    if comparison.get("status") != "completed_p418_physical_step_model_comparison":
        raise ValueError("formal physical-step model comparison is not complete")
    if STRICT_SPLIT not in comparison.get("splits", []):
        raise ValueError("formal comparison does not contain the strict endpoint-pair split")

    split_source = load_json(args.splits.resolve())["splits"][STRICT_SPLIT]
    expected_ids = list(map(str, split_source["test"]))
    if len(expected_ids) != 4 or set(expected_ids) != set(EVENT_LABELS):
        raise ValueError("registered strict split is not the expected four-trajectory test")

    integration_path = selected_chain_record_path(result_dir)
    selected_loss_ready = integration_path.is_file()
    if not selected_loss_ready and not args.allow_preselection_diagnostic:
        raise ValueError(
            "formal transient figure requires validation-selected loss balancing; "
            "use --allow-preselection-diagnostic only for a non-manuscript diagnostic"
        )
    if args.validation_marker is not None and not selected_loss_ready:
        raise ValueError(
            "a formal validation marker requires validation-selected loss balancing"
        )
    selected_directories = selected_model_directories(
        result_dir,
        STRICT_SPLIT,
        allow_registered_preselection=not selected_loss_ready,
    )
    physics_dir = selected_directories["graph_transformer_energy_flux"]
    persistence_dir = result_dir / f"regional_persistence_{STRICT_SPLIT}"
    data_only_dir = (
        result_dir / f"regional_graph_transformer_bounded_data_only_{STRICT_SPLIT}"
    )
    diffusion_dir = selected_directories["diffusion_residual_correction"]
    physics_summary = load_json(physics_dir / "summary.json")
    persistence_summary = load_json(persistence_dir / "summary.json")
    data_only_summary = load_json(data_only_dir / "summary.json")
    diffusion_summary = load_json(diffusion_dir / "summary.json")
    require_formal_split(physics_summary, expected_ids, "physics graph--Transformer")
    require_formal_split(
        persistence_summary, expected_ids, "initial-temperature persistence"
    )
    require_formal_split(data_only_summary, expected_ids, "data-only graph--Transformer")
    require_formal_split(diffusion_summary, expected_ids, "diffusion correction")

    common_required = {
        "sequence_id",
        "time_s",
        "baseline_temperature_normalized",
        "target_temperature_normalized",
        "node_type",
        "node_volume_m3",
        "temperature_mean_K_by_node_type",
        "temperature_std_K_by_node_type",
    }
    physics = load_npz(physics_dir / "test_temporal_temperature_predictions.npz", common_required)
    persistence_required = {
        "sequence_id",
        "time_s",
        "temperature_prediction_K",
        "temperature_target_K",
        "node_type",
        "node_volume_m3",
    }
    persistence = load_npz(
        persistence_dir / "test_temperature_predictions.npz",
        persistence_required,
    )
    data_only = load_npz(data_only_dir / "test_temporal_temperature_predictions.npz", common_required)
    diffusion_required = {
        "sequence_id",
        "time_s",
        "refined_temperature_normalized",
        "refined_temperature_q05_normalized",
        "refined_temperature_q95_normalized",
        "target_temperature_normalized",
        "node_type",
        "node_volume_m3",
        "temperature_mean_K_by_node_type",
        "temperature_std_K_by_node_type",
    }
    diffusion = load_npz(diffusion_dir / "test_refined_temperature.npz", diffusion_required)
    require_matching_prediction_coordinates(physics, data_only, "data-only prediction")
    require_matching_prediction_coordinates(
        physics, persistence, "initial-temperature persistence"
    )
    require_matching_prediction_coordinates(physics, diffusion, "diffusion prediction")
    if list(map(str, physics["sequence_id"])) != expected_ids:
        raise ValueError("prediction file trajectory order differs from the registered strict split")
    validate_formal_time_axes(np.asarray(physics["time_s"], dtype=float))

    truth = solid_volume_average(
        physical_temperature(physics["target_temperature_normalized"], physics), physics
    )
    if not np.allclose(
        persistence["temperature_target_K"],
        physical_temperature(physics["target_temperature_normalized"], physics),
        rtol=0.0,
        atol=1.0e-5,
    ):
        raise ValueError(
            "initial-temperature persistence and physics models use different targets"
        )
    persistence_curve = solid_volume_average(
        persistence["temperature_prediction_K"], persistence
    )
    physics_curve = solid_volume_average(
        physical_temperature(physics["baseline_temperature_normalized"], physics), physics
    )
    data_only_curve = solid_volume_average(
        physical_temperature(data_only["baseline_temperature_normalized"], data_only), data_only
    )
    diffusion_curve = solid_volume_average(
        physical_temperature(diffusion["refined_temperature_normalized"], diffusion), diffusion
    )
    diffusion_q05 = solid_volume_average(
        physical_temperature(diffusion["refined_temperature_q05_normalized"], diffusion), diffusion
    )
    diffusion_q95 = solid_volume_average(
        physical_temperature(diffusion["refined_temperature_q95_normalized"], diffusion), diffusion
    )
    if not all(
        np.all(np.isfinite(array))
        for array in (
            truth,
            persistence_curve,
            physics_curve,
            data_only_curve,
            diffusion_curve,
            diffusion_q05,
            diffusion_q95,
        )
    ):
        raise ValueError("formal trajectory file contains non-finite temperatures")
    if np.any(diffusion_q05 > diffusion_q95):
        raise ValueError("diffusion prediction contains reversed 5--95% intervals")

    metric_values = read_metric_table(comparison_dir / "physical_step_model_metrics.csv")
    speed_values = read_speed_table(comparison_dir / "physical_step_model_speedup.csv")
    split_rmse = {
        "data_only_graph_transformer": split_field_rmse(
            data_only_summary, "data-only graph--Transformer"
        ),
        "physics_constrained_graph_transformer": split_field_rmse(
            physics_summary, "physics-constrained graph--Transformer"
        ),
    }
    casewise_rmse = {
        "initial_temperature_persistence": casewise_curve_rmse(
            persistence_curve, truth, expected_ids
        ),
        "data_only_graph_transformer": casewise_curve_rmse(
            data_only_curve, truth, expected_ids
        ),
        "physics_constrained_graph_transformer": casewise_curve_rmse(
            physics_curve, truth, expected_ids
        ),
        "diffusion_residual_correction": casewise_curve_rmse(
            diffusion_curve, truth, expected_ids
        ),
    }

    apply_ijhmt_style(
        font_size=7.8,
        label_size=8.1,
        tick_size=7.0,
        legend_size=6.7,
        axis_width=0.75,
    )
    figure, axes = plt.subplots(3, 2, figsize=FIGURE_SIZE_INCH)
    figure.subplots_adjust(**FIGURE_ADJUST)
    line_handles = None
    for index, (axis, sequence_id) in enumerate(zip(axes.flat[:4], expected_ids)):
        axis_style(axis)
        time_s = physics["time_s"][index]
        truth_handle, = axis.plot(time_s, truth[index], color="black", linewidth=1.25)
        persistence_handle, = axis.plot(
            time_s,
            persistence_curve[index],
            color=MODEL_COLORS["initial_temperature_persistence"],
            linestyle=":",
            linewidth=1.0,
        )
        data_handle, = axis.plot(
            time_s, data_only_curve[index], color="#999999", linestyle="--", linewidth=1.0
        )
        physics_handle, = axis.plot(
            time_s, physics_curve[index], color="#0072B2", linewidth=1.15
        )
        diffusion_handle, = axis.plot(
            time_s, diffusion_curve[index], color="#D55E00", linewidth=1.05
        )
        axis.fill_between(
            time_s,
            diffusion_q05[index],
            diffusion_q95[index],
            color="#D55E00",
            alpha=0.16,
            linewidth=0.0,
        )
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(r"Solid $\langle T\rangle_V$ (K)")
        axis.text(
            0.97,
            0.06,
            EVENT_LABELS[sequence_id],
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.0,
            fontweight="semibold",
        )
        add_panel_label(axis, index)
        line_handles = (
            truth_handle,
            persistence_handle,
            data_handle,
            physics_handle,
            diffusion_handle,
        )

    if line_handles is None:
        raise RuntimeError("no strict held-out trajectories were plotted")
    figure.legend(
        line_handles,
        ("OpenFOAM", "persistence", "data-only GT", "physics GT", "diffusion mean"),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.987),
        frameon=False,
        ncol=5,
        handlelength=2.4,
        columnspacing=1.25,
    )

    error_axis = axes[2, 0]
    axis_style(error_axis)
    temperature_midpoint = float(
        np.median([metric_values[model]["temperature"] for model in MODEL_ORDER])
    )
    energy_midpoint = float(
        np.median([metric_values[model]["energy"] for model in MODEL_ORDER])
    )
    for model in MODEL_ORDER:
        point = metric_values[model]
        error_axis.scatter(
            point["temperature"],
            point["energy"],
            s=28,
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            edgecolor="black",
            linewidth=0.35,
            zorder=3,
        )
        annotate_point(
            error_axis,
            point["temperature"],
            point["energy"],
            MODEL_LABELS[model],
            temperature_midpoint,
            energy_midpoint,
        )
    error_axis.set_xlabel("Solid-temperature RMSE (K)")
    error_axis.set_ylabel("Projection-aware energy RMSE")
    positive_energy = [metric_values[model]["energy"] for model in MODEL_ORDER]
    if min(positive_energy) > 0.0 and max(positive_energy) / min(positive_energy) >= 20.0:
        error_axis.set_yscale("log")
    add_panel_label(error_axis, 4)

    speed_axis = axes[2, 1]
    axis_style(speed_axis)
    speed_midpoint = float(
        np.exp(np.median(np.log([speed_values[model] for model in MODEL_ORDER])))
    )
    speed_temperature_midpoint = float(
        np.median([metric_values[model]["temperature"] for model in MODEL_ORDER])
    )
    for model in MODEL_ORDER:
        speed_axis.scatter(
            speed_values[model],
            metric_values[model]["temperature"],
            s=28,
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            edgecolor="black",
            linewidth=0.35,
            zorder=3,
        )
        annotate_point(
            speed_axis,
            speed_values[model],
            metric_values[model]["temperature"],
            MODEL_LABELS[model],
            speed_midpoint,
            speed_temperature_midpoint,
        )
    speed_axis.set_xscale("log")
    speed_axis.set_xlabel("Inference speed-up vs 32-rank OpenFOAM")
    speed_axis.set_ylabel("Solid-temperature RMSE (K)")
    add_panel_label(speed_axis, 5)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    output_stem = (
        "hccb_p418_transient_model_comparison"
        if selected_loss_ready
        else "hccb_p418_transient_model_comparison_preselection_diagnostic"
    )
    pdf = output / f"{output_stem}.pdf"
    svg = output / f"{output_stem}.svg"
    png = output / f"{output_stem}.png"
    canvas_bounds = figure.bbox_inches
    figure.savefig(pdf, bbox_inches=canvas_bounds)
    figure.savefig(svg, bbox_inches=canvas_bounds)
    figure.savefig(png, dpi=600, bbox_inches=canvas_bounds)
    plt.close(figure)

    provenance = {
        "status": (
            "complete_formal_p418_transient_model_comparison_figure"
            if selected_loss_ready
            else "diagnostic_p418_transient_model_comparison_preselection"
        ),
        "split_name": STRICT_SPLIT,
        "held_out_trajectory_ids": expected_ids,
        "held_out_trajectory_count": len(expected_ids),
        "full_thermal_step_dataset_count": 12,
        "figure_size_inch": list(FIGURE_SIZE_INCH),
        "figure_size_mm": [137.16, 170.18],
        "panel_width_to_height_ratio": panel_width_to_height_ratio(),
        "physics_prediction": str((physics_dir / "test_temporal_temperature_predictions.npz").resolve()),
        "persistence_prediction": str(
            (persistence_dir / "test_temperature_predictions.npz").resolve()
        ),
        "data_only_prediction": str((data_only_dir / "test_temporal_temperature_predictions.npz").resolve()),
        "diffusion_prediction": str((diffusion_dir / "test_refined_temperature.npz").resolve()),
        "strict_split_loss_balancing_stage": (
            "validation_selected" if selected_loss_ready else "registered_preselection"
        ),
        "selected_loss_balancing_integration_record": (
            str(integration_path) if selected_loss_ready else None
        ),
        "selected_loss_balancing_integration_record_sha256": (
            sha256(integration_path) if selected_loss_ready else None
        ),
        "model_metric_table": str((comparison_dir / "physical_step_model_metrics.csv").resolve()),
        "energy_axis_metric": ENERGY_METRIC,
        "energy_axis_metric_unit": "dimensionless",
        "energy_axis_metric_source": (
            "common finite-volume energy post-processing applied to every model"
        ),
        "speed_table": str((comparison_dir / "physical_step_model_speedup.csv").resolve()),
        "speedup_definition": (
            "median 32-rank OpenFOAM wall time per held-out trajectory divided by "
            "complete-chain inference wall time per trajectory; training and reference-data "
            "costs are reported separately through break-even curve counts"
        ),
        "split_regional_field_RMSE_K": split_rmse,
        "held_out_volume_averaged_solid_curve_RMSE_K_by_case": casewise_rmse,
        "pdf": str(pdf),
        "svg": str(svg),
        "png": str(png),
        "new_physical_parameter_values_added": [],
        "interpretation_scope": (
            "Computed fixed-flow thermal-step predictions on the registered P418 endpoint-pair "
            "split. The figure is not generated from incomplete trajectories or smoke-test output."
        ),
    }
    summary_path = output / f"{output_stem}.json"
    summary_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validation_marker = None
    if selected_loss_ready:
        validation_marker = (
            args.validation_marker.resolve()
            if args.validation_marker is not None
            else output.parent
            / "manuscript"
            / "generated_transient_model_comparison_validated.tex"
        )
        validation_marker.parent.mkdir(parents=True, exist_ok=True)
        validation_marker.write_text(
            "% Generated only after validation-selected loss balancing.\n",
            encoding="utf-8",
        )
    provenance["validation_marker"] = (
        str(validation_marker) if validation_marker is not None else None
    )
    summary_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
