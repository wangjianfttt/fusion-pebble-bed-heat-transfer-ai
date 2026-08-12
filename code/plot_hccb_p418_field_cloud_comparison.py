#!/usr/bin/env python3
"""Plot same-scale OpenFOAM, model and signed-error temperature fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ijhmt_figure_style import apply_ijhmt_style


BLACK = "#171717"
BOUNDARY_GRAY = "#8A8A8A"
FIGURE_WIDTH_INCH = 5.40
FIGURE_HEIGHT_INCH = 6.70
FIGURE_WIDTH_MM = FIGURE_WIDTH_INCH * 25.4
FIGURE_HEIGHT_MM = FIGURE_HEIGHT_INCH * 25.4
FIGURE_SIZE_INCH = (FIGURE_WIDTH_INCH, FIGURE_HEIGHT_INCH)
PANEL_ROWS = 3
PANEL_COLUMNS = 2
FORMAL_OUTPUT_STEM = "hccb_p418_openfoam_model_field_comparison"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_relative(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def verify_selected_file(
    record: dict,
    path_key: str,
    sha_key: str,
    project_root: Path | None = None,
) -> Path:
    """Resolve one selection-record file and prove it has not changed."""
    raw_path = record.get(path_key)
    expected_sha = record.get(sha_key)
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"field-model selection is missing {path_key}")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError(f"field-model selection is missing {sha_key}")
    raw = Path(raw_path)
    base = project_root.resolve() if project_root is not None else Path.cwd().resolve()
    path = (base / raw).resolve() if not raw.is_absolute() else raw.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256(path) != expected_sha:
        raise ValueError(f"selected field-model file changed after selection: {path_key}")
    return path


def latex_text(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    return "".join(replacements.get(character, character) for character in value)


def setup_style() -> None:
    apply_ijhmt_style(
        font_size=7.2,
        label_size=7.5,
        tick_size=7.0,
        legend_size=6.8,
        axis_width=0.65,
    )


def panel_style(axis: plt.Axes) -> None:
    axis.tick_params(direction="in", top=True, right=True, width=0.65, length=2.6)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_color(BLACK)
    axis.set_aspect("equal", adjustable="box")


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.065,
        1.018,
        label,
        transform=axis.transAxes,
        fontsize=7.5,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def physical_temperature(
    normalized: np.ndarray, node_type: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    if normalized.ndim != 4 or normalized.shape[-1] != 1:
        raise ValueError("temperature prediction must have shape [sequence,time,node,1]")
    if len(node_type) != normalized.shape[2] or mean.shape != (2,) or std.shape != (2,):
        raise ValueError("temperature normalization metadata is inconsistent")
    scale = std[node_type][None, None, :, None]
    offset = mean[node_type][None, None, :, None]
    temperature = normalized * scale + offset
    if not np.all(np.isfinite(temperature)):
        raise ValueError("temperature file contains non-finite values")
    return temperature[..., 0]


def read_temperature_fields(
    data: np.lib.npyio.NpzFile,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, str]:
    common = {
        "sequence_id",
        "time_s",
        "node_type",
        "node_volume_m3",
        "node_centroid_m",
    }
    missing_common = sorted(common - set(data.files))
    if missing_common:
        raise ValueError(f"prediction file is missing {missing_common}")
    arrays = {key: data[key] for key in common}
    normalized_metadata = {
        "target_temperature_normalized",
        "temperature_mean_K_by_node_type",
        "temperature_std_K_by_node_type",
    }
    physical_keys = {"temperature_prediction_K", "temperature_target_K"}
    node_type = arrays["node_type"].astype(np.int8)
    normalized_predictions = (
        ("refined_temperature_normalized", "diffusion-refined normalized temperature restored to K"),
        ("corrected_temperature_normalized", "low-rank-corrected normalized temperature restored to K"),
        ("baseline_temperature_normalized", "node-type-normalized temperature restored to K"),
    )
    selected_normalized = next(
        (
            (prediction_key, representation)
            for prediction_key, representation in normalized_predictions
            if normalized_metadata | {prediction_key} <= set(data.files)
        ),
        None,
    )
    if selected_normalized is not None:
        prediction_key, representation = selected_normalized
        selected_keys = normalized_metadata | {prediction_key}
        arrays.update({key: data[key] for key in selected_keys})
        truth = physical_temperature(
            arrays["target_temperature_normalized"],
            node_type,
            arrays["temperature_mean_K_by_node_type"],
            arrays["temperature_std_K_by_node_type"],
        )
        prediction = physical_temperature(
            arrays[prediction_key],
            node_type,
            arrays["temperature_mean_K_by_node_type"],
            arrays["temperature_std_K_by_node_type"],
        )
    elif physical_keys <= set(data.files):
        arrays.update({key: data[key] for key in physical_keys})
        truth = np.asarray(arrays["temperature_target_K"], dtype=float)
        prediction = np.asarray(arrays["temperature_prediction_K"], dtype=float)
        if truth.ndim != 3 or prediction.shape != truth.shape:
            raise ValueError(
                "physical temperature arrays must share shape [sequence,time,node]"
            )
        if truth.shape[2] != len(node_type):
            raise ValueError("physical temperature nodes do not match node_type")
        if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
            raise ValueError("physical temperature arrays contain non-finite values")
        representation = "physical temperature in K"
    else:
        raise ValueError(
            "prediction file contains neither the formal normalized-temperature "
            "fields nor the physical-temperature fields"
        )
    return arrays, truth, prediction, representation


def validate_prediction_geometry(
    arrays: dict[str, np.ndarray],
    geometry_node_type: np.ndarray,
    geometry_volume_m3: np.ndarray,
    geometry_centroid_m: np.ndarray,
) -> None:
    """Require identical node order while allowing float32 storage rounding."""
    prediction_node_type = arrays["node_type"].astype(np.int8)
    prediction_volume_m3 = arrays["node_volume_m3"].astype(float)
    prediction_centroid_m = arrays["node_centroid_m"].astype(float)
    if (
        len(geometry_node_type) != len(prediction_node_type)
        or not np.array_equal(
            geometry_node_type.astype(np.int8), prediction_node_type
        )
        or not np.allclose(
            geometry_volume_m3.astype(float),
            prediction_volume_m3,
            rtol=1.0e-6,
            atol=0.0,
        )
        or not np.allclose(
            geometry_centroid_m.astype(float),
            prediction_centroid_m,
            rtol=1.0e-6,
            atol=1.0e-11,
        )
    ):
        raise ValueError(
            "prediction nodes do not match the specified regional geometry; "
            "field plotting is stopped to prevent coordinate misalignment"
        )


def select_sequence_and_time(
    sequence_ids: np.ndarray,
    time_s: np.ndarray,
    sequence_id: str,
    requested_time_s: float,
) -> tuple[int, int, float]:
    matches = np.flatnonzero(sequence_ids.astype(str) == sequence_id)
    if len(matches) != 1:
        raise ValueError(
            f"sequence {sequence_id!r} occurs {len(matches)} times; "
            f"available={list(map(str, sequence_ids))}"
        )
    sequence_index = int(matches[0])
    current_time = np.asarray(time_s[sequence_index], dtype=float)
    time_index = int(np.argmin(np.abs(current_time - requested_time_s)))
    selected_time = float(current_time[time_index])
    if abs(selected_time - requested_time_s) > max(1.0e-8, 0.01 * max(requested_time_s, 1.0)):
        raise ValueError(
            f"requested time {requested_time_s:g} s is not represented; "
            f"nearest={selected_time:g} s"
        )
    return sequence_index, time_index, selected_time


def particle_section_mask(
    horizontal_grid_mm: np.ndarray,
    vertical_grid_mm: np.ndarray,
    centers_m: np.ndarray,
    radius_m: float,
    section_y_m: float,
) -> tuple[np.ndarray, int]:
    intersects = np.abs(centers_m[:, 1] - section_y_m) < radius_m
    section_centers = centers_m[intersects] * 1.0e3
    section_radii_mm = (
        np.sqrt(radius_m**2 - (centers_m[intersects, 1] - section_y_m) ** 2)
        * 1.0e3
    )
    mask = np.zeros(horizontal_grid_mm.shape, dtype=bool)
    for center, section_radius in zip(section_centers, section_radii_mm):
        mask |= (
            (horizontal_grid_mm - center[2]) ** 2
            + (vertical_grid_mm - center[0]) ** 2
            <= section_radius**2
        )
    return mask, int(np.sum(intersects))


def interpolate_on_phase_grid(
    horizontal_mm: np.ndarray,
    vertical_mm: np.ndarray,
    field: np.ndarray,
    horizontal_grid_mm: np.ndarray,
    vertical_grid_mm: np.ndarray,
    phase_mask: np.ndarray,
) -> np.ma.MaskedArray:
    triangulation = mtri.Triangulation(horizontal_mm, vertical_mm)
    interpolator = mtri.LinearTriInterpolator(triangulation, field)
    interpolated = np.ma.asarray(
        interpolator(horizontal_grid_mm, vertical_grid_mm), dtype=float
    )
    return np.ma.masked_where(
        ~phase_mask | np.ma.getmaskarray(interpolated), interpolated
    )


def volume_weighted_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    volume: np.ndarray,
    selected: np.ndarray,
) -> tuple[float, float]:
    weights = volume[selected].astype(float)
    weights /= weights.sum()
    error = prediction[selected] - truth[selected]
    return float(np.sqrt(np.sum(weights * error**2))), float(np.max(np.abs(error)))


def error_localization_metrics(
    error: np.ndarray,
    volume: np.ndarray,
    coordinates_m: np.ndarray,
    boundary_fraction: np.ndarray,
    boundary_role_names: np.ndarray,
    selected: np.ndarray,
) -> dict[str, object]:
    """Record whether a large field error is concentrated at a known boundary."""
    indices = np.flatnonzero(selected)
    if len(indices) == 0:
        raise ValueError("error localization requires at least one selected node")
    selected_error = np.abs(error[indices])
    maximum_local = int(np.argmax(selected_error))
    maximum_index = int(indices[maximum_local])
    selected_volume = volume[indices].astype(float)
    maximum_volume_fraction = float(
        volume[maximum_index] / selected_volume.sum()
    )
    return {
        "absolute_error_percentile_K": {
            "p99": float(np.percentile(selected_error, 99.0)),
            "p99p9": float(np.percentile(selected_error, 99.9)),
        },
        "maximum_error_node": {
            "index": maximum_index,
            "signed_error_K": float(error[maximum_index]),
            "absolute_error_K": float(abs(error[maximum_index])),
            "centroid_mm": (coordinates_m[maximum_index] * 1.0e3).tolist(),
            "regional_volume_fraction": maximum_volume_fraction,
            "boundary_fraction": {
                str(name): float(value)
                for name, value in zip(
                    boundary_role_names,
                    boundary_fraction[maximum_index],
                )
                if value > 0.0
            },
        },
    }


def robust_symmetric_error_limit(error: np.ndarray, percentile: float) -> float:
    if not 90.0 <= percentile <= 100.0:
        raise ValueError("error display percentile must be between 90 and 100")
    absolute = np.abs(np.asarray(error, dtype=float))
    if absolute.size == 0 or not np.all(np.isfinite(absolute)):
        raise ValueError("error field is empty or non-finite")
    limit = float(np.percentile(absolute, percentile))
    if limit <= 0.0:
        raise ValueError("selected field has no plottable error range")
    return limit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prediction", type=Path)
    source.add_argument(
        "--selection",
        type=Path,
        help="Completed model-selection record for the final manuscript field figure.",
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        default=Path(
            "results/hccb_p418_physical_steps_12/regional_sequences/"
            "regional_sequence_geometry.npz"
        ),
        help="Regional geometry supplying node coordinates.",
    )
    parser.add_argument(
        "--packing",
        type=Path,
        default=Path(
            "runs/hccb_dense_snappy_g2_nativezone_r2/geometry/packing_crop.npz"
        ),
        help="Local spherical-particle subset used by the OpenFOAM mesh.",
    )
    parser.add_argument(
        "--expected-particle-count",
        type=int,
        default=125,
        help="Specified number of spherical particles in the local P418 domain.",
    )
    parser.add_argument("--sequence-id", default="source_up_u0p15_T700")
    parser.add_argument("--time-s", type=float, default=25.0)
    parser.add_argument("--model-label", default="graph--Transformer")
    parser.add_argument(
        "--slice-half-width-mm",
        type=float,
        default=0.10,
        help="Half-width of the y-centred plotting slab.",
    )
    parser.add_argument(
        "--error-display-percentile",
        type=float,
        default=99.0,
        help=(
            "Symmetric error colour limit as a percentile of absolute slice error. "
            "Full-field RMSE and maximum error are always reported without clipping."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--validation-marker", type=Path)
    parser.add_argument(
        "--allow-unselected-diagnostic",
        action="store_true",
        help=(
            "Allow a direct prediction only for a clearly named diagnostic figure. "
            "Formal manuscript output requires --selection."
        ),
    )
    parser.add_argument(
        "--output-stem",
        default=FORMAL_OUTPUT_STEM,
    )
    return parser.parse_args()


def resolve_output_mode(args: argparse.Namespace) -> tuple[str, bool]:
    """Separate validation-selected manuscript output from unselected diagnostics."""
    selected = args.selection is not None
    if selected:
        if args.allow_unselected_diagnostic:
            raise ValueError("--allow-unselected-diagnostic cannot be used with --selection")
        return str(args.output_stem), True
    if not args.allow_unselected_diagnostic:
        raise ValueError(
            "formal OpenFOAM--model field figure requires validation-only --selection; "
            "use --allow-unselected-diagnostic only for a non-manuscript diagnostic"
        )
    if args.validation_marker is not None:
        raise ValueError("an unselected diagnostic cannot write a validation marker")
    stem = str(args.output_stem)
    if stem == FORMAL_OUTPUT_STEM:
        stem = f"{FORMAL_OUTPUT_STEM}_unselected_diagnostic"
    if "diagnostic" not in stem.lower():
        raise ValueError("an unselected field output stem must contain 'diagnostic'")
    return stem, False


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    args.output_stem, formal_output = resolve_output_mode(args)
    selection_path = None
    selection_record = None
    if args.selection is not None:
        selection_path = args.selection.resolve()
        if not selection_path.is_file():
            raise FileNotFoundError(selection_path)
        selection_record = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection_record.get("status") != "selected_p418_field_figure_learned_model":
            raise ValueError("field-figure model selection is incomplete")
        if selection_record.get("new_physical_parameters") != []:
            raise ValueError("field-figure model selection introduced a physical parameter")
        if selection_record.get("selection_data_role") != "validation":
            raise ValueError("field-figure model must be selected on validation data")
        if selection_record.get("display_data_role") != "test":
            raise ValueError("field figure must display a held-out test trajectory")
        if selection_record.get("split_name") != "pair_disjoint_stress_test":
            raise ValueError("field figure must use the registered strict split")
        if (
            selection_record.get("strict_split_loss_balancing_stage")
            != "validation_selected"
        ):
            raise ValueError(
                "field figure requires validation-selected loss balancing"
            )
        prediction_path = verify_selected_file(
            selection_record,
            "prediction_file",
            "prediction_file_sha256",
            project_root,
        )
        for path_key, sha_key in (
            ("model_summary", "model_summary_sha256"),
            ("comparison_summary", "comparison_summary_sha256"),
            ("metrics_csv", "metrics_csv_sha256"),
            (
                "selected_loss_balancing_integration_record",
                "selected_loss_balancing_integration_record_sha256",
            ),
        ):
            verify_selected_file(selection_record, path_key, sha_key, project_root)
        args.model_label = str(selection_record["selected_model_label"])
    else:
        prediction_path = args.prediction.resolve()
    if not prediction_path.is_file():
        raise FileNotFoundError(
            f"formal prediction file is not available: {prediction_path}. "
            "No placeholder cloud field will be generated."
        )
    with np.load(prediction_path, allow_pickle=False) as data:
        arrays, truth_all, prediction_all, temperature_representation = (
            read_temperature_fields(data)
        )
    geometry_path = args.geometry.resolve()
    if not geometry_path.is_file():
        raise FileNotFoundError(f"regional geometry is unavailable: {geometry_path}")
    with np.load(geometry_path, allow_pickle=False) as geometry:
        geometry_required = {
            "node_centroid_m",
            "node_volume_m3",
            "node_type",
            "node_boundary_fraction",
            "boundary_role_names",
        }
        missing_geometry = sorted(geometry_required - set(geometry.files))
        if missing_geometry:
            raise ValueError(f"geometry file is missing {missing_geometry}")
        coordinates_m = geometry["node_centroid_m"].astype(float)
        geometry_volume = geometry["node_volume_m3"].astype(float)
        geometry_node_type = geometry["node_type"].astype(np.int8)
        geometry_boundary_fraction = geometry["node_boundary_fraction"].astype(float)
        geometry_boundary_role_names = geometry["boundary_role_names"].astype(str)
    if geometry_boundary_fraction.shape != (
        len(geometry_node_type),
        len(geometry_boundary_role_names),
    ):
        raise ValueError("geometry boundary-role arrays are inconsistent")
    packing_path = args.packing.resolve()
    if not packing_path.is_file():
        raise FileNotFoundError(f"spherical packing is unavailable: {packing_path}")
    with np.load(packing_path, allow_pickle=False) as packing:
        packing_required = {"centres_m", "meshing_radius_m"}
        missing_packing = sorted(packing_required - set(packing.files))
        if missing_packing:
            raise ValueError(f"packing file is missing {missing_packing}")
        particle_centers_m = packing["centres_m"].astype(float)
        meshing_radius_m = float(packing["meshing_radius_m"])
    if len(particle_centers_m) != args.expected_particle_count:
        raise ValueError(
            "packing geometry does not match the specified local P418 domain: "
            f"expected {args.expected_particle_count} particles, found "
            f"{len(particle_centers_m)}"
        )
    validate_prediction_geometry(
        arrays,
        geometry_node_type,
        geometry_volume,
        coordinates_m,
    )

    sequence_index, time_index, selected_time = select_sequence_and_time(
        arrays["sequence_id"],
        arrays["time_s"],
        args.sequence_id,
        args.time_s,
    )
    node_type = arrays["node_type"].astype(np.int8)
    truth = truth_all[sequence_index, time_index]
    prediction = prediction_all[sequence_index, time_index]
    error = prediction - truth
    coordinates = coordinates_m * 1.0e3
    volume = arrays["node_volume_m3"].astype(float)
    center_y = 0.5 * (coordinates[:, 1].min() + coordinates[:, 1].max())
    slab = np.abs(coordinates[:, 1] - center_y) <= args.slice_half_width_mm
    material_masks = {
        0: slab & (node_type == 0),
        1: slab & (node_type == 1),
    }
    if min(np.sum(mask) for mask in material_masks.values()) < 300:
        raise ValueError("the requested plotting slab contains too few fluid or solid nodes")

    temperature_values = np.concatenate(
        [truth[slab], prediction[slab]]
    )
    temperature_min = float(np.min(temperature_values))
    temperature_max = float(np.max(temperature_values))
    full_error_limit = float(np.max(np.abs(error[slab])))
    error_limit = robust_symmetric_error_limit(
        error[slab], args.error_display_percentile
    )
    saturated_error_fraction = float(np.mean(np.abs(error[slab]) > error_limit))
    if temperature_max <= temperature_min or full_error_limit <= 0:
        raise ValueError("selected field has no plottable temperature or error range")

    setup_style()
    horizontal_grid_mm, vertical_grid_mm = np.meshgrid(
        np.linspace(coordinates[:, 2].min(), coordinates[:, 2].max(), 440),
        np.linspace(coordinates[:, 0].min(), coordinates[:, 0].max(), 500),
    )
    particle_mask, intersecting_particle_count = particle_section_mask(
        horizontal_grid_mm,
        vertical_grid_mm,
        particle_centers_m,
        meshing_radius_m,
        center_y * 1.0e-3,
    )
    if intersecting_particle_count < 1:
        raise ValueError("the specified packing does not intersect the requested section")
    figure, axes = plt.subplots(
        PANEL_ROWS,
        PANEL_COLUMNS,
        figsize=FIGURE_SIZE_INCH,
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(
        left=0.105,
        right=0.885,
        bottom=0.070,
        top=0.932,
        wspace=0.085,
        hspace=0.095,
    )
    phase_names = ("fluid", "solid")
    temperature_mappable = None
    error_mappable = None
    metrics: dict[str, dict[str, float]] = {}
    error_localization: dict[str, dict[str, object]] = {}
    fields_by_material: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for material in (0, 1):
        mask = material_masks[material]
        rmse, maximum = volume_weighted_metrics(
            truth, prediction, volume, node_type == material
        )
        metrics[phase_names[material]] = {
            "full_field_volume_weighted_RMSE_K": rmse,
            "full_field_max_abs_error_K": maximum,
            "slice_node_count": int(np.sum(mask)),
        }
        error_localization[phase_names[material]] = error_localization_metrics(
            error,
            volume,
            coordinates_m,
            geometry_boundary_fraction,
            geometry_boundary_role_names,
            node_type == material,
        )
        fields_by_material[material] = (
            truth[mask],
            prediction[mask],
            error[mask],
        )

    for row in range(PANEL_ROWS):
        for column, material in enumerate((0, 1)):
            axis = axes[row, column]
            mask = material_masks[material]
            horizontal = coordinates[mask, 2]
            vertical = coordinates[mask, 0]
            field = fields_by_material[material][row]
            panel_style(axis)
            phase_geometry = ~particle_mask if material == 0 else particle_mask
            plotted_field = interpolate_on_phase_grid(
                horizontal,
                vertical,
                field,
                horizontal_grid_mm,
                vertical_grid_mm,
                phase_geometry,
            )
            if row < 2:
                mappable = axis.pcolormesh(
                    horizontal_grid_mm,
                    vertical_grid_mm,
                    plotted_field,
                    shading="auto",
                    cmap="inferno",
                    vmin=temperature_min,
                    vmax=temperature_max,
                    rasterized=True,
                )
                temperature_mappable = mappable
            else:
                mappable = axis.pcolormesh(
                    horizontal_grid_mm,
                    vertical_grid_mm,
                    plotted_field,
                    shading="auto",
                    cmap="RdBu_r",
                    vmin=-error_limit,
                    vmax=error_limit,
                    rasterized=True,
                )
                error_mappable = mappable
            axis.contour(
                horizontal_grid_mm,
                vertical_grid_mm,
                particle_mask.astype(float),
                levels=[0.5],
                colors=[BOUNDARY_GRAY],
                linewidths=0.22,
                alpha=0.8,
            )
            axis.set_xlim(coordinates[:, 2].min(), coordinates[:, 2].max())
            axis.set_ylim(coordinates[:, 0].min(), coordinates[:, 0].max())
            if row == 2:
                metric = metrics[phase_names[material]]
                metric_text = axis.text(
                    0.025,
                    0.030,
                    rf"RMSE {metric['full_field_volume_weighted_RMSE_K']:.2f} K"
                    rf"  |  max {metric['full_field_max_abs_error_K']:.2f} K",
                    transform=axis.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=6.4,
                    color=BLACK,
                )
                metric_text.set_path_effects(
                    [path_effects.withStroke(linewidth=1.2, foreground="white")]
                )
            add_panel_label(
                axis, f"({chr(ord('a') + row * PANEL_COLUMNS + column)})"
            )
        axes[row, 0].set_ylabel("$x$ (mm)")
    for axis in axes[-1]:
        axis.set_xlabel("$z$ (mm)")

    if temperature_mappable is None or error_mappable is None:
        raise RuntimeError("field panels were not generated")
    temperature_colorbar_axis = figure.add_axes([0.915, 0.395, 0.014, 0.520])
    temperature_colorbar = figure.colorbar(
        temperature_mappable, cax=temperature_colorbar_axis
    )
    temperature_colorbar.set_label("$T$ (K)", fontsize=7.3)
    temperature_colorbar.ax.tick_params(labelsize=6.8, direction="in", width=0.6)
    error_colorbar_axis = figure.add_axes([0.915, 0.085, 0.014, 0.235])
    error_colorbar = figure.colorbar(
        error_mappable, cax=error_colorbar_axis, extend="both"
    )
    error_colorbar.set_label("Error (K)", fontsize=7.3)
    error_colorbar.ax.tick_params(labelsize=6.8, direction="in", width=0.6)
    figure.canvas.draw()
    panel_axis_bounds = [list(map(float, axis.get_position().bounds)) for axis in axes.flat]
    panel_widths = [bounds[2] for bounds in panel_axis_bounds]
    panel_heights = [bounds[3] for bounds in panel_axis_bounds]
    if max(panel_widths) - min(panel_widths) > 1.0e-6:
        raise RuntimeError("field-comparison panels do not have equal widths")
    if max(panel_heights) - min(panel_heights) > 1.0e-6:
        raise RuntimeError("field-comparison panels do not have equal heights")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / f"{args.output_stem}.pdf"
    svg = output / f"{args.output_stem}.svg"
    png = output / f"{args.output_stem}.png"
    # Preserve the declared manuscript-width canvas. Tight cropping would make
    # this field map narrower than the other full-width multi-panel figures.
    canvas_bounds = figure.bbox_inches
    figure.savefig(pdf, dpi=600, bbox_inches=canvas_bounds)
    figure.savefig(svg, dpi=600, bbox_inches=canvas_bounds)
    figure.savefig(png, dpi=600, bbox_inches=canvas_bounds)
    plt.close(figure)

    record = {
        "status": (
            "complete_same_scale_openfoam_model_field_comparison"
            if formal_output
            else "diagnostic_same_scale_openfoam_model_field_comparison"
        ),
        "prediction_file": project_relative(prediction_path, project_root),
        "prediction_file_sha256": sha256(prediction_path),
        "geometry_file": project_relative(geometry_path, project_root),
        "geometry_file_sha256": sha256(geometry_path),
        "packing_file": project_relative(packing_path, project_root),
        "packing_file_sha256": sha256(packing_path),
        "registered_local_particle_count": int(len(particle_centers_m)),
        "sequence_id": args.sequence_id,
        "model_label": args.model_label,
        "selected_model": (
            selection_record.get("selected_model")
            if selection_record is not None
            else None
        ),
        "selection_data_role": (
            selection_record.get("selection_data_role")
            if selection_record is not None
            else None
        ),
        "display_data_role": (
            selection_record.get("display_data_role")
            if selection_record is not None
            else None
        ),
        "strict_split_loss_balancing_stage": (
            selection_record.get("strict_split_loss_balancing_stage")
            if selection_record is not None
            else None
        ),
        "model_selection_file": (
            project_relative(selection_path, project_root) if selection_path else None
        ),
        "model_selection_file_sha256": sha256(selection_path) if selection_path else None,
        "temperature_representation": temperature_representation,
        "requested_time_s": args.time_s,
        "selected_time_s": selected_time,
        "slice": {
            "normal_coordinate": "y",
            "center_mm": center_y,
            "half_width_mm": args.slice_half_width_mm,
            "intersecting_spherical_particles": intersecting_particle_count,
        },
        "temperature_scale_K": [temperature_min, temperature_max],
        "signed_error_display_scale_K": [-error_limit, error_limit],
        "signed_error_full_slice_range_K": [-full_error_limit, full_error_limit],
        "error_display_percentile": args.error_display_percentile,
        "slice_error_saturated_fraction": saturated_error_fraction,
        "figure_size_mm": [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM],
        "panel_layout": [PANEL_ROWS, PANEL_COLUMNS],
        "panel_axis_bounds": panel_axis_bounds,
        "panel_width_to_height_ratio": panel_widths[0] / panel_heights[0],
        "panel_order": {
            "rows": ["OpenFOAM", args.model_label, "model minus OpenFOAM"],
            "columns": ["fluid phase", "solid phase"],
        },
        "metrics": metrics,
        "error_localization": error_localization,
        "outputs": {
            "pdf": project_relative(pdf, project_root),
            "svg": project_relative(svg, project_root),
            "png": project_relative(png, project_root),
        },
        "plotting_rules": [
            "OpenFOAM and model panels use one common physical temperature scale.",
            "Error panels show signed model-minus-OpenFOAM temperature on a zero-centred "
            f"{args.error_display_percentile:g}th-percentile display scale; colourbar "
            "extensions mark saturated extremes.",
            "Full-field RMSE and maximum absolute error are computed before display clipping.",
            "Fluid and solid full-field metrics are regional-volume weighted.",
            "No interpolated or image-generated field is used as numerical evidence.",
            "Display interpolation is clipped by exact spherical-particle section geometry; "
            "reported metrics are computed on the original three-dimensional regional nodes.",
            "Panels contain no method-name titles or boxes; their order and the exact model "
            "identity are stated in the caption and record.",
            "The figure uses two panels per row: fluid on the left and solid on the right; "
            "rows show OpenFOAM, model and signed error.",
        ],
        "new_physical_parameters": [],
    }
    summary = output / f"{args.output_stem}.json"
    validation_marker = None
    if args.selection is not None or args.validation_marker is not None:
        if args.selection is None:
            raise ValueError(
                "a formal validation marker requires --selection; a direct prediction "
                "may be plotted only as a named baseline"
            )
        validation_marker = (
            args.validation_marker.resolve()
            if args.validation_marker is not None
            else output.parent
            / "manuscript"
            / "generated_openfoam_model_field_comparison_validated.tex"
        )
        validation_marker.parent.mkdir(parents=True, exist_ok=True)
        validation_marker.write_text(
            "% Generated only from the complete formal OpenFOAM--model field comparison.\n"
            f"\\renewcommand{{\\PFieldModelLabel}}{{{latex_text(args.model_label)}}}\n"
            f"\\renewcommand{{\\PFieldFluidRMSE}}{{{metrics['fluid']['full_field_volume_weighted_RMSE_K']:.2f}}}\n"
            f"\\renewcommand{{\\PFieldSolidRMSE}}{{{metrics['solid']['full_field_volume_weighted_RMSE_K']:.2f}}}\n"
            f"\\renewcommand{{\\PFieldFluidMaxError}}{{{metrics['fluid']['full_field_max_abs_error_K']:.2f}}}\n"
            f"\\renewcommand{{\\PFieldSolidMaxError}}{{{metrics['solid']['full_field_max_abs_error_K']:.2f}}}\n",
            encoding="utf-8",
        )
    record["validation_marker"] = (
        project_relative(validation_marker, project_root)
        if validation_marker is not None
        else None
    )
    summary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
