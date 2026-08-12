#!/usr/bin/env python3
"""Collect physical-step model results into one comparison table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np

from report_hccb_p418_runtime_progress import accumulated_solver_clock_time
from hccb_p418_selected_fixed_flow_chain import (
    STRICT_SPLIT,
    selected_chain_record_path,
    selected_model_directories,
)


SPLITS = (
    "direction_down_test",
    "direction_up_test",
    "pair_disjoint_stress_test",
)
SPLIT_ROLES = ("train", "validation", "test")
PRIMARY_ENERGY_METRIC = (
    "projection_aware_volume_weighted_energy_equation_normalized_RMSE"
)
REFERENCE_ENERGY_METRIC = (
    "openfoam_reference_volume_weighted_energy_equation_normalized_RMSE"
)
TEMPERATURE_METRIC_DEFINITION = (
    "regional-volume-weighted RMSE, reported separately for fluid and solid"
)
COMMON_HOTSPOT_METRICS = {
    "solid_maximum_temperature_history_RMSE_K",
    "solid_maximum_temperature_history_maximum_absolute_error_K",
    "solid_regional_hotspot_location_mean_error_m",
    "solid_regional_hotspot_location_p95_error_m",
    "solid_regional_hotspot_location_maximum_error_m",
    "solid_regional_hotspot_exact_match_fraction",
    "solid_hotspot_target_temperature_deficit_mean_K",
    "solid_hotspot_target_temperature_deficit_p95_K",
    "solid_hotspot_target_temperature_deficit_maximum_K",
    "solid_hotspot_prediction_temperature_deficit_mean_K",
    "solid_hotspot_prediction_temperature_deficit_p95_K",
    "solid_hotspot_prediction_temperature_deficit_maximum_K",
    "solid_hotspot_dynamic_sample_count",
}


def require_strict_loss_selection(
    split_names: list[str] | tuple[str, ...], integration_path: Path
) -> None:
    """Do not read strict-split test results before validation-only loss selection."""
    if STRICT_SPLIT in split_names and not integration_path.is_file():
        raise ValueError(
            "strict model comparison requires the completed validation-selected "
            "loss-balancing chain before independent test aggregation"
        )


def load_summary(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing completed model result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require_temperature_metric_definition(summary: dict, model_name: str) -> None:
    recorded = summary.get("temperature_metric_definition")
    if recorded != TEMPERATURE_METRIC_DEFINITION:
        raise ValueError(
            f"{model_name} temperature metric definition differs from the common "
            f"comparison basis: recorded={recorded!r}, "
            f"expected={TEMPERATURE_METRIC_DEFINITION!r}"
        )


def require_exact_split(
    summary: dict,
    expected: dict[str, list[str]],
    *,
    split_name: str,
    model_name: str,
) -> None:
    recorded_name = summary.get("split_name")
    if recorded_name is not None and recorded_name != split_name:
        raise ValueError(f"{model_name} reports split {recorded_name}, expected {split_name}")
    recorded = summary.get("split_case_ids")
    if recorded is None:
        raise ValueError(f"{model_name} does not record the actual train/validation/test curves")
    for role, identifiers in expected.items():
        actual = [str(value) for value in recorded.get(role, [])]
        if len(actual) != len(set(actual)) or set(actual) != set(identifiers):
            raise ValueError(
                f"{model_name} {role} curves differ from {split_name}: "
                f"actual={actual}, expected={identifiers}"
            )


def require_validation_selection(summary: dict, model_name: str) -> dict[str, object]:
    if summary.get("selection_split") != "validation":
        raise ValueError(f"{model_name} was not selected using validation curves only")
    metric = str(summary.get("selection_metric", "")).strip()
    if not metric:
        raise ValueError(f"{model_name} does not record its validation selection metric")
    if "selected_epoch" in summary:
        selected_name = "epoch"
        selected_value = int(summary["selected_epoch"])
    elif "selected_rank" in summary:
        selected_name = "rank"
        selected_value = int(summary["selected_rank"])
    else:
        raise ValueError(f"{model_name} does not record the selected epoch or rank")
    return {
        "model": model_name,
        "selection_split": "validation",
        "selection_metric": metric,
        "selected_quantity": selected_name,
        "selected_value": selected_value,
    }


def require_identical_training_statistics(first: Path, second: Path) -> None:
    with np.load(first, allow_pickle=False) as left, np.load(second, allow_pickle=False) as right:
        if set(left.files) != set(right.files):
            raise ValueError("data-only and physics-constrained normalization fields differ")
        changed = [name for name in left.files if not np.array_equal(left[name], right[name])]
    if changed:
        raise ValueError(f"data-only and physics-constrained training statistics differ: {changed}")


def openfoam_clock_times(step_root: Path) -> dict[str, float]:
    result = {}
    for case in step_root.iterdir():
        log = case / "log.foamMultiRun.step"
        if not case.is_dir() or not log.is_file():
            continue
        clock_time = accumulated_solver_clock_time(log)
        if clock_time is not None:
            result[case.name] = float(clock_time)
    return result


def model_size(summary: dict, model_name: str) -> tuple[int, str]:
    if model_name == "initial_temperature_persistence":
        return 0, "no fitted or stored model scalars"
    if model_name == "dmdc":
        return int(summary["model_storage_scalar_count"]), "stored DMDc scalars"
    if model_name == "low_rank_residual_correction":
        return int(summary["model_storage_scalar_count"]), summary["model_size_definition"]
    return int(summary["model_parameter_count"]), "trainable neural parameters"


def add_speed_row(
    rows: list[dict],
    *,
    split: str,
    model: str,
    component_size: int,
    component_size_definition: str,
    component_device: str,
    component_training_seconds: float,
    component_inference_seconds_per_curve: float,
    openfoam_median_seconds: float,
    reference_data_seconds: float,
    upstream_model: str = "",
    upstream_summary: dict | None = None,
    allow_zero_model_size: bool = False,
) -> None:
    upstream_size = 0
    upstream_training = 0.0
    upstream_inference = 0.0
    compute_device = component_device
    size_definition = component_size_definition
    if upstream_summary is not None:
        upstream_size = int(upstream_summary["model_parameter_count"])
        upstream_training = float(upstream_summary["training_seconds"])
        upstream_inference = float(
            upstream_summary["metrics"]["test"]["inference_seconds_per_curve"]
        )
        compute_device = (
            f"upstream={upstream_summary['compute_device']}; "
            f"component={component_device}"
        )
        size_definition = (
            "complete prediction chain: upstream trainable parameters + "
            f"{component_size_definition}"
        )

    total_size = upstream_size + int(component_size)
    total_training = upstream_training + float(component_training_seconds)
    total_inference = upstream_inference + float(component_inference_seconds_per_curve)
    values = (
        total_training,
        total_inference,
        openfoam_median_seconds,
        reference_data_seconds,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError(f"invalid timing value for {model} {split}: {values}")
    if (
        total_size < 0
        or (total_size == 0 and not allow_zero_model_size)
        or total_inference <= 0.0
        or openfoam_median_seconds <= 0.0
    ):
        raise ValueError(f"invalid model size or prediction time for {model} {split}")
    saved_seconds = openfoam_median_seconds - total_inference
    if saved_seconds > 0.0:
        training_break_even = int(math.ceil(total_training / saved_seconds))
        full_break_even = int(
            math.ceil((total_training + reference_data_seconds) / saved_seconds)
        )
    else:
        training_break_even = None
        full_break_even = None
    rows.append(
        {
            "split_name": split,
            "model": model,
            "upstream_model": upstream_model,
            "model_size_scalar_count": total_size,
            "component_model_size_scalar_count": int(component_size),
            "model_size_definition": size_definition,
            "compute_device": compute_device,
            "training_wall_time_s": total_training,
            "component_training_wall_time_s": float(component_training_seconds),
            "reference_training_validation_clock_time_s": reference_data_seconds,
            "openfoam_median_clock_time_s": openfoam_median_seconds,
            "model_inference_seconds_per_curve": total_inference,
            "component_inference_seconds_per_curve": float(
                component_inference_seconds_per_curve
            ),
            "wall_clock_speedup": openfoam_median_seconds / total_inference,
            "training_only_break_even_curve_count": training_break_even,
            "full_workflow_break_even_curve_count": full_break_even,
        }
    )


def add_metric(
    rows: list[dict],
    *,
    split: str,
    model: str,
    scope: str,
    role: str,
    metric: str,
    value: float,
    unit: str,
    training_seconds: float,
    source: Path,
) -> None:
    rows.append(
        {
            "split_name": split,
            "model": model,
            "result_scope": scope,
            "data_role": role,
            "metric": metric,
            "value": float(value),
            "unit": unit,
            "training_seconds": float(training_seconds),
            "source_summary": str(source),
        }
    )


def project_relative(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"model-comparison source is outside the project root: {path}") from error


def temperature_metric_unit(metric: str) -> str:
    if metric.endswith("sample_count") or metric == "observation_count":
        return "count"
    if metric.startswith("inference_seconds"):
        return "s"
    if "location_" in metric and metric.endswith("_m"):
        return "m"
    if "fraction" in metric:
        return "fraction"
    if "normalized_RMSE" in metric:
        return "dimensionless"
    if "_K" in metric or metric.startswith("ensemble_mean_RMSE"):
        return "K"
    return "dimensionless"


def require_hotspot_metrics(metrics: dict, model_name: str, prefix: str = "") -> None:
    required = {f"{prefix}{name}" for name in COMMON_HOTSPOT_METRICS}
    missing = required.difference(metrics)
    if missing:
        raise ValueError(f"{model_name} lacks common transient hotspot metrics: {sorted(missing)}")


def diffusion_metric_unit(metric: str) -> str:
    if "coverage_fraction" in metric:
        return "fraction"
    if "energy_residual_ratio" in metric:
        return "ratio"
    if "energy_equation_normalized_RMSE" in metric:
        return "dimensionless"
    return temperature_metric_unit(metric)


def load_common_energy_summary(
    directory: Path,
    *,
    split_name: str,
    model_name: str,
    allow_registered_range_rejections: bool = False,
) -> tuple[Path, dict]:
    path = directory / "energy_balance_summary.json"
    summary = load_summary(path)
    if summary.get("split_name") != split_name:
        raise ValueError(
            f"{model_name} energy balance reports split {summary.get('split_name')}, "
            f"expected {split_name}"
        )
    roles = summary.get("role_metrics")
    if not isinstance(roles, dict):
        raise ValueError(f"{model_name} has no common energy metrics")
    expected = set(SPLIT_ROLES)
    if allow_registered_range_rejections:
        failures = summary.get("role_failures", {})
        requested = set(summary.get("requested_roles", SPLIT_ROLES))
        evaluated = set(summary.get("evaluated_roles", roles))
        rejected = set(summary.get("rejected_roles", failures))
        if (
            requested != expected
            or evaluated != set(roles)
            or rejected != set(failures)
            or evaluated & rejected
            or evaluated | rejected != expected
        ):
            raise ValueError(
                f"{model_name} has an inconsistent evaluated/rejected energy-role record"
            )
        for role, failure in failures.items():
            if failure.get("status") != (
                "prediction_outside_registered_thermophysical_range"
            ):
                raise ValueError(
                    f"{model_name} {role} energy result failed for an unsupported reason"
                )
    elif set(roles) != expected:
        raise ValueError(f"{model_name} lacks common train/validation/test energy metrics")
    return path, summary


def pareto_temperature_energy(points: list[dict[str, float | str]]) -> list[dict]:
    output = []
    for candidate in points:
        dominated = any(
            other["solid_temperature_RMSE_K"]
            <= candidate["solid_temperature_RMSE_K"]
            and other["projection_aware_volume_weighted_energy_normalized_RMSE"]
            <= candidate["projection_aware_volume_weighted_energy_normalized_RMSE"]
            and (
                other["solid_temperature_RMSE_K"]
                < candidate["solid_temperature_RMSE_K"]
                or other["projection_aware_volume_weighted_energy_normalized_RMSE"]
                < candidate["projection_aware_volume_weighted_energy_normalized_RMSE"]
            )
            for other in points
        )
        if not dominated:
            output.append(candidate)
    return sorted(
        output,
        key=lambda row: (
            row["solid_temperature_RMSE_K"],
            row["projection_aware_volume_weighted_energy_normalized_RMSE"],
        ),
    )


def diffusion_temperature_energy_decision(
    deterministic_temperature_rmse: float,
    refined_temperature_rmse: float,
    deterministic_energy_rmse: float,
    refined_energy_rmse: float,
) -> dict[str, bool | str]:
    """Describe the held-out temperature--energy outcome without selecting a model."""
    values = (
        deterministic_temperature_rmse,
        refined_temperature_rmse,
        deterministic_energy_rmse,
        refined_energy_rmse,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("diffusion temperature/energy comparison values are invalid")
    temperature_improved = refined_temperature_rmse < deterministic_temperature_rmse
    energy_not_worse = refined_energy_rmse <= deterministic_energy_rmse
    joint_improvement = temperature_improved and energy_not_worse
    if joint_improvement:
        reason = "temperature_error_decreased_and_energy_residual_did_not_increase"
    elif not temperature_improved and not energy_not_worse:
        reason = "temperature_error_did_not_decrease_and_energy_residual_increased"
    elif not temperature_improved:
        reason = "temperature_error_did_not_decrease"
    else:
        reason = "energy_residual_increased_despite_lower_temperature_error"
    return {
        "temperature_improved": temperature_improved,
        "projection_aware_energy_not_worse": energy_not_worse,
        "diffusion_is_joint_temperature_energy_improvement": joint_improvement,
        "held_out_outcome": "joint_improvement" if joint_improvement else "not_joint_improvement",
        "outcome_reason": reason,
        "model_selection_uses_this_outcome": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--step-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split-names", nargs="+", default=list(SPLITS))
    parser.add_argument("--seed-robustness-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.result_dir = args.result_dir.resolve()
    args.step_root = args.step_root.resolve()
    args.splits = args.splits.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.seed_robustness_summary is not None:
        args.seed_robustness_summary = args.seed_robustness_summary.resolve()
    project_root = args.result_dir.parents[1]

    rows: list[dict] = []
    speed_rows: list[dict] = []
    selection_rows: list[dict] = []
    best_by_split: dict[str, dict] = {}
    diffusion_energy_effect_by_split: dict[str, dict] = {}
    temperature_energy_pareto_by_split: dict[str, list[dict]] = {}
    energy_range_rejections_by_split: dict[str, dict[str, dict]] = {}
    selected_model_sources_by_split: dict[str, dict[str, str]] = {}
    strict_integration_path = selected_chain_record_path(args.result_dir)
    require_strict_loss_selection(args.split_names, strict_integration_path)
    strict_loss_selection_ready = strict_integration_path.is_file()
    clock_times = openfoam_clock_times(args.step_root)
    split_source = json.loads(args.splits.read_text(encoding="utf-8"))["splits"]
    for split in args.split_names:
        test_ids = [str(value) for value in split_source[split]["test"]]
        expected_ids = {
            role: [str(value) for value in split_source[split][role]]
            for role in ("train", "validation", "test")
        }
        missing_clock = [value for value in test_ids if value not in clock_times]
        if missing_clock:
            raise ValueError(f"missing OpenFOAM wall-clock time for {split}: {missing_clock}")
        openfoam_median = statistics.median(clock_times[value] for value in test_ids)
        reference_ids = expected_ids["train"] + expected_ids["validation"]
        missing_reference_clock = [
            value for value in reference_ids if value not in clock_times
        ]
        if missing_reference_clock:
            raise ValueError(
                f"missing training/validation OpenFOAM wall-clock time for {split}: "
                f"{missing_reference_clock}"
            )
        reference_data_seconds = sum(clock_times[value] for value in reference_ids)
        observable_path = args.result_dir / f"transformer_{split}" / "summary.json"
        observable = load_summary(observable_path)
        require_exact_split(
            observable, expected_ids, split_name=split, model_name="observable_transformer"
        )
        selection_rows.append(
            {
                "split_name": split,
                **require_validation_selection(observable, "observable_transformer"),
            }
        )
        for metric, value in observable["test_mean_rmse_by_target"].items():
            unit = "K" if "temperature" in metric else "SI unit of target"
            add_metric(
                rows,
                split=split,
                model="observable_transformer",
                scope="integral_observables",
                role="test",
                metric=f"{metric}_RMSE",
                value=value,
                unit=unit,
                training_seconds=observable["training_seconds"],
                source=observable_path,
            )
        observable_seconds = (
            observable["inference_seconds_by_split"]["test"]
            / observable["split_case_counts"]["test"]
        )
        add_speed_row(
            speed_rows,
            split=split,
            model="observable_transformer",
            component_size=int(observable["model_parameter_count"]),
            component_size_definition="trainable neural parameters",
            component_device=observable["compute_device"],
            component_training_seconds=float(observable["training_seconds"]),
            component_inference_seconds_per_curve=observable_seconds,
            openfoam_median_seconds=openfoam_median,
            reference_data_seconds=reference_data_seconds,
        )

        selected_directories = selected_model_directories(
            args.result_dir,
            split,
        )
        selected_model_sources_by_split[split] = {
            name: project_relative(path, project_root)
            for name, path in selected_directories.items()
        }
        regional_models = (
            (
                "initial_temperature_persistence",
                args.result_dir / f"regional_persistence_{split}",
            ),
            ("dmdc", args.result_dir / f"regional_dmdc_{split}"),
            (
                "graph_transformer_data_only",
                args.result_dir / f"regional_graph_transformer_bounded_data_only_{split}",
            ),
            (
                "graph_transformer_energy_flux",
                selected_directories["graph_transformer_energy_flux"],
            ),
            (
                "graph_transformer_factorized_energy_flux",
                selected_directories["graph_transformer_factorized_energy_flux"],
            ),
        )
        candidates: list[tuple[str, float]] = []
        energy_by_model: dict[str, float] = {}
        regional_summaries: dict[str, dict] = {}
        for model_name, directory in regional_models:
            summary_path = directory / "summary.json"
            summary = load_summary(summary_path)
            require_temperature_metric_definition(summary, model_name)
            require_exact_split(summary, expected_ids, split_name=split, model_name=model_name)
            if model_name == "initial_temperature_persistence":
                selection_rows.append(
                    {
                        "split_name": split,
                        "model": model_name,
                        "selection_split": "not_applicable",
                        "selection_metric": (
                            "none; initial temperature is repeated without fitting"
                        ),
                        "selected_quantity": "none",
                        "selected_value": "",
                    }
                )
            else:
                selection_rows.append(
                    {
                        "split_name": split,
                        **require_validation_selection(summary, model_name),
                    }
                )
            regional_summaries[model_name] = summary
            energy_path, energy_summary = load_common_energy_summary(
                args.result_dir / directory,
                split_name=split,
                model_name=model_name,
            )
            for role, energy_metrics in energy_summary["role_metrics"].items():
                for metric, value in energy_metrics.items():
                    add_metric(
                        rows,
                        split=split,
                        model=model_name,
                        scope="transient_energy_balance",
                        role=role,
                        metric=metric,
                        value=value,
                        unit="count" if metric == "curve_count" else (
                            "ratio" if metric.endswith("_ratio") else "dimensionless"
                        ),
                        training_seconds=summary["training_seconds"],
                        source=energy_path,
                    )
            energy_by_model[model_name] = float(
                energy_summary["role_metrics"]["test"][PRIMARY_ENERGY_METRIC]
            )
            require_hotspot_metrics(summary["metrics"]["test"], model_name)
            for role, metrics in summary["metrics"].items():
                for metric, value in metrics.items():
                    add_metric(
                        rows,
                        split=split,
                        model=model_name,
                        scope="regional_temperature_field",
                        role=role,
                        metric=metric,
                        value=value,
                        unit=temperature_metric_unit(metric),
                        training_seconds=summary["training_seconds"],
                        source=summary_path,
                    )
            candidates.append(
                (model_name, summary["metrics"]["test"]["solid_temperature_RMSE_K"])
            )
            inference_seconds = summary["metrics"]["test"]["inference_seconds_per_curve"]
            size_count, size_definition = model_size(summary, model_name)
            add_speed_row(
                speed_rows,
                split=split,
                model=model_name,
                component_size=size_count,
                component_size_definition=size_definition,
                component_device=summary["compute_device"],
                component_training_seconds=float(summary["training_seconds"]),
                component_inference_seconds_per_curve=inference_seconds,
                openfoam_median_seconds=openfoam_median,
                reference_data_seconds=reference_data_seconds,
                allow_zero_model_size=(
                    model_name == "initial_temperature_persistence"
                ),
            )

        data_only_dir = (
            args.result_dir / f"regional_graph_transformer_bounded_data_only_{split}"
        )
        physics_dir = (
            selected_directories["graph_transformer_energy_flux"]
        )
        factorized_dir = (
            selected_directories["graph_transformer_factorized_energy_flux"]
        )
        if (
            regional_summaries["graph_transformer_data_only"].get("architecture")
            != regional_summaries["graph_transformer_energy_flux"].get("architecture")
        ):
            raise ValueError("data-only and physics-constrained graph--Transformers use different architectures")
        require_identical_training_statistics(
            data_only_dir / "training_statistics.npz",
            physics_dir / "training_statistics.npz",
        )
        factorized_architecture = regional_summaries[
            "graph_transformer_factorized_energy_flux"
        ].get("architecture", {})
        repeated_architecture = regional_summaries[
            "graph_transformer_energy_flux"
        ].get("architecture", {})
        if factorized_architecture.get("spatial_temporal_mode") != "factorized_static_spatial":
            raise ValueError("factorized graph--Transformer does not record the factorized mode")
        if repeated_architecture.get("spatial_temporal_mode") != "repeated_query_spatial":
            raise ValueError("reference graph--Transformer does not record the repeated-query mode")
        if {
            key: value
            for key, value in factorized_architecture.items()
            if key != "spatial_temporal_mode"
        } != {
            key: value
            for key, value in repeated_architecture.items()
            if key != "spatial_temporal_mode"
        }:
            raise ValueError(
                "factorized and repeated-query graph--Transformers differ beyond spatial timing"
            )
        if (
            regional_summaries["graph_transformer_factorized_energy_flux"].get(
                "loss_weights"
            )
            != regional_summaries["graph_transformer_energy_flux"].get(
                "loss_weights"
            )
        ):
            raise ValueError(
                "factorized and repeated-query graph--Transformers use different loss weights"
            )
        require_identical_training_statistics(
            physics_dir / "training_statistics.npz",
            factorized_dir / "training_statistics.npz",
        )

        low_rank_path = (
            selected_directories["low_rank_residual_correction"] / "summary.json"
        )
        low_rank = load_summary(low_rank_path)
        require_temperature_metric_definition(
            low_rank, "low_rank_residual_correction"
        )
        require_exact_split(
            low_rank, expected_ids, split_name=split, model_name="low_rank_residual_correction"
        )
        selection_rows.append(
            {
                "split_name": split,
                **require_validation_selection(
                    low_rank, "low_rank_residual_correction"
                ),
            }
        )
        if Path(low_rank.get("deterministic_prediction_dir", "")).resolve() != physics_dir.resolve():
            raise ValueError("low-rank correction does not use the physics-constrained predictions")
        low_rank_energy_path, low_rank_energy = load_common_energy_summary(
            low_rank_path.parent,
            split_name=split,
            model_name="low_rank_residual_correction",
            allow_registered_range_rejections=True,
        )
        for role, energy_metrics in low_rank_energy["role_metrics"].items():
            for metric, value in energy_metrics.items():
                add_metric(
                    rows,
                    split=split,
                    model="low_rank_residual_correction",
                    scope="transient_energy_balance",
                    role=role,
                    metric=metric,
                    value=value,
                    unit="count" if metric == "curve_count" else (
                        "ratio" if metric.endswith("_ratio") else "dimensionless"
                    ),
                    training_seconds=low_rank["training_seconds"],
                    source=low_rank_energy_path,
                )
        low_rank_failures = low_rank_energy.get("role_failures", {})
        if low_rank_failures:
            energy_range_rejections_by_split.setdefault(split, {})[
                "low_rank_residual_correction"
            ] = low_rank_failures
            for role, failure in low_rank_failures.items():
                for metric in (
                    "prediction_fluid_temperature_min_K",
                    "prediction_fluid_temperature_max_K",
                    "prediction_solid_temperature_min_K",
                    "prediction_solid_temperature_max_K",
                    "prediction_nonfinite_value_count",
                    "prediction_fluid_nonpositive_value_count",
                    "prediction_solid_out_of_range_value_count",
                    "prediction_solid_out_of_range_fraction",
                ):
                    value = failure.get(metric)
                    if value is None:
                        continue
                    unit = (
                        "K"
                        if metric.endswith("_K")
                        else "fraction"
                        if metric.endswith("_fraction")
                        else "count"
                    )
                    add_metric(
                        rows,
                        split=split,
                        model="low_rank_residual_correction",
                        scope="registered_thermophysical_range_check",
                        role=role,
                        metric=metric,
                        value=value,
                        unit=unit,
                        training_seconds=low_rank["training_seconds"],
                        source=low_rank_energy_path,
                    )
        if "test" in low_rank_energy["role_metrics"]:
            energy_by_model["low_rank_residual_correction"] = float(
                low_rank_energy["role_metrics"]["test"][PRIMARY_ENERGY_METRIC]
            )
        require_hotspot_metrics(
            low_rank["metrics"]["test"], "low_rank_residual_correction"
        )
        for role, role_metrics in low_rank["metrics"].items():
            for metric, value in role_metrics.items():
                add_metric(
                    rows,
                    split=split,
                    model="low_rank_residual_correction",
                    scope="regional_temperature_field",
                    role=role,
                    metric=metric,
                    value=value,
                    unit=temperature_metric_unit(metric),
                    training_seconds=low_rank["training_seconds"],
                    source=low_rank_path,
                )
        candidates.append(
            (
                "low_rank_residual_correction",
                low_rank["metrics"]["test"]["solid_temperature_RMSE_K"],
            )
        )
        low_rank_seconds = low_rank["metrics"]["test"]["inference_seconds_per_curve"]
        size_count, size_definition = model_size(low_rank, "low_rank_residual_correction")
        add_speed_row(
            speed_rows,
            split=split,
            model="low_rank_residual_correction",
            component_size=size_count,
            component_size_definition=size_definition,
            component_device=low_rank["compute_device"],
            component_training_seconds=float(low_rank["training_seconds"]),
            component_inference_seconds_per_curve=low_rank_seconds,
            openfoam_median_seconds=openfoam_median,
            reference_data_seconds=reference_data_seconds,
            upstream_model="graph_transformer_energy_flux",
            upstream_summary=regional_summaries["graph_transformer_energy_flux"],
        )

        diffusion_path = (
            selected_directories["diffusion_residual_correction"] / "summary.json"
        )
        diffusion = load_summary(diffusion_path)
        require_temperature_metric_definition(
            diffusion, "diffusion_residual_correction"
        )
        require_exact_split(
            diffusion, expected_ids, split_name=split, model_name="diffusion_residual_correction"
        )
        selection_rows.append(
            {
                "split_name": split,
                **require_validation_selection(
                    diffusion, "diffusion_residual_correction"
                ),
            }
        )
        if Path(diffusion.get("deterministic_prediction_dir", "")).resolve() != physics_dir.resolve():
            raise ValueError("diffusion correction does not use the physics-constrained deterministic predictions")
        diffusion_common_path, diffusion_common = load_common_energy_summary(
            diffusion_path.parent,
            split_name=split,
            model_name="diffusion_residual_correction",
        )
        for role, energy_metrics in diffusion_common["role_metrics"].items():
            for metric, value in energy_metrics.items():
                add_metric(
                    rows,
                    split=split,
                    model="diffusion_residual_correction",
                    scope="transient_energy_balance",
                    role=role,
                    metric=metric,
                    value=value,
                    unit="count" if metric == "curve_count" else (
                        "ratio" if metric.endswith("_ratio") else "dimensionless"
                    ),
                    training_seconds=diffusion["training_seconds"],
                    source=diffusion_common_path,
                )
        energy_by_model["diffusion_residual_correction"] = float(
            diffusion_common["role_metrics"]["test"][PRIMARY_ENERGY_METRIC]
        )
        require_hotspot_metrics(
            diffusion["metrics"]["test"],
            "diffusion_residual_correction",
            prefix="diffusion_refined_",
        )
        for role, metrics in diffusion["metrics"].items():
            for metric, value in metrics.items():
                if isinstance(value, dict):
                    for sample_count, nested_value in value.items():
                        add_metric(
                            rows,
                            split=split,
                            model="diffusion_residual_correction",
                            scope="regional_temperature_field",
                            role=role,
                            metric=f"{metric}_{sample_count}_samples",
                            value=nested_value,
                            unit="K",
                            training_seconds=diffusion["training_seconds"],
                            source=diffusion_path,
                        )
                    continue
                unit = diffusion_metric_unit(metric)
                add_metric(
                    rows,
                    split=split,
                    model="diffusion_residual_correction",
                    scope="regional_temperature_field",
                    role=role,
                    metric=metric,
                    value=value,
                    unit=unit,
                    training_seconds=diffusion["training_seconds"],
                    source=diffusion_path,
                )
        candidates.append(
            (
                "diffusion_residual_correction",
                diffusion["metrics"]["test"]["diffusion_refined_solid_temperature_RMSE_K"],
            )
        )
        diffusion_test = diffusion["metrics"]["test"]
        required_energy = {
            "deterministic_absolute_energy_equation_normalized_RMSE",
            "diffusion_refined_absolute_energy_equation_normalized_RMSE",
            "openfoam_reference_absolute_energy_equation_normalized_RMSE",
            "diffusion_to_deterministic_energy_residual_ratio",
            "diffusion_to_openfoam_reference_energy_residual_ratio",
            "diffusion_member_projection_aware_energy_equation_normalized_RMSE_p95",
            "diffusion_member_joint_temperature_energy_improvement_fraction",
            "diffusion_member_sample_count",
        }
        required_uncertainty = {
            "diffusion_90pct_interval_coverage_fraction",
            "diffusion_90pct_interval_mean_width_K",
        }
        missing_energy = required_energy.difference(diffusion_test)
        if missing_energy:
            raise ValueError(
                "diffusion result lacks transient energy-balance comparison: "
                f"{sorted(missing_energy)}"
            )
        missing_uncertainty = required_uncertainty.difference(diffusion_test)
        if missing_uncertainty:
            raise ValueError(
                "diffusion result lacks interval-calibration metrics: "
                f"{sorted(missing_uncertainty)}"
            )
        common_diffusion_energy = energy_by_model["diffusion_residual_correction"]
        internal_diffusion_energy = float(
            diffusion_test[
                "diffusion_refined_absolute_energy_equation_normalized_RMSE"
            ]
        )
        if not math.isfinite(internal_diffusion_energy) or internal_diffusion_energy < 0.0:
            raise ValueError("diffusion internal energy-balance result is invalid")
        member_energy_p95 = float(
            diffusion_test[
                "diffusion_member_projection_aware_energy_equation_normalized_RMSE_p95"
            ]
        )
        member_joint_fraction = float(
            diffusion_test[
                "diffusion_member_joint_temperature_energy_improvement_fraction"
            ]
        )
        member_count = int(diffusion_test["diffusion_member_sample_count"])
        interval_coverage = float(
            diffusion_test["diffusion_90pct_interval_coverage_fraction"]
        )
        interval_width = float(
            diffusion_test["diffusion_90pct_interval_mean_width_K"]
        )
        if (
            not math.isfinite(member_energy_p95)
            or member_energy_p95 < 0.0
            or not math.isfinite(member_joint_fraction)
            or not 0.0 <= member_joint_fraction <= 1.0
            or member_count < 2
            or not math.isfinite(interval_coverage)
            or not 0.0 <= interval_coverage <= 1.0
            or not math.isfinite(interval_width)
            or interval_width < 0.0
        ):
            raise ValueError(
                "diffusion member-level physical or uncertainty results are invalid"
            )
        diffusion_decision = diffusion_temperature_energy_decision(
            float(diffusion_test["deterministic_solid_temperature_RMSE_K"]),
            float(diffusion_test["diffusion_refined_solid_temperature_RMSE_K"]),
            energy_by_model["graph_transformer_energy_flux"],
            common_diffusion_energy,
        )
        diffusion_energy_effect_by_split[split] = {
            "deterministic_solid_temperature_RMSE_K": float(
                diffusion_test["deterministic_solid_temperature_RMSE_K"]
            ),
            "diffusion_refined_solid_temperature_RMSE_K": float(
                diffusion_test["diffusion_refined_solid_temperature_RMSE_K"]
            ),
            "deterministic_projection_aware_energy_normalized_RMSE": energy_by_model[
                "graph_transformer_energy_flux"
            ],
            "diffusion_refined_projection_aware_energy_normalized_RMSE": common_diffusion_energy,
            "diffusion_member_projection_aware_energy_equation_normalized_RMSE_p95": member_energy_p95,
            "diffusion_member_joint_temperature_energy_improvement_fraction": member_joint_fraction,
            "diffusion_member_sample_count": member_count,
            "diffusion_90pct_interval_coverage_fraction": interval_coverage,
            "diffusion_90pct_interval_mean_width_K": interval_width,
            "diffusion_90pct_interval_is_underdispersed": (
                interval_coverage < 0.85
            ),
            "openfoam_reference_absolute_energy_normalized_RMSE": float(
                diffusion_common["role_metrics"]["test"][REFERENCE_ENERGY_METRIC]
            ),
            "internal_gpu_to_common_cpu_diffusion_energy_ratio": (
                internal_diffusion_energy
                / max(common_diffusion_energy, np.finfo(np.float64).tiny)
            ),
            **diffusion_decision,
            "projection_aware_energy_improved": bool(
                common_diffusion_energy
                < energy_by_model["graph_transformer_energy_flux"]
            ),
        }
        diffusion_seconds = diffusion["metrics"]["test"]["inference_seconds_per_curve"]
        add_speed_row(
            speed_rows,
            split=split,
            model="diffusion_residual_correction",
            component_size=int(diffusion["model_parameter_count"]),
            component_size_definition="diffusion-refiner trainable parameters",
            component_device=diffusion["compute_device"],
            component_training_seconds=float(diffusion["training_seconds"]),
            component_inference_seconds_per_curve=diffusion_seconds,
            openfoam_median_seconds=openfoam_median,
            reference_data_seconds=reference_data_seconds,
            upstream_model="graph_transformer_energy_flux",
            upstream_summary=regional_summaries["graph_transformer_energy_flux"],
        )
        best_model, best_rmse = min(candidates, key=lambda item: item[1])
        best_by_split[split] = {
            "lowest_test_solid_temperature_RMSE_model": best_model,
            "lowest_test_solid_temperature_RMSE_K": float(best_rmse),
        }
        pareto_points = [
            {
                "model": model_name,
                "solid_temperature_RMSE_K": float(temperature_rmse),
                "projection_aware_volume_weighted_energy_normalized_RMSE": energy_by_model[
                    model_name
                ],
            }
            for model_name, temperature_rmse in candidates
            if model_name in energy_by_model
        ]
        temperature_energy_pareto_by_split[split] = pareto_temperature_energy(
            pareto_points
        )

    for row in rows:
        row["source_summary"] = project_relative(
            Path(str(row["source_summary"])), project_root
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "physical_step_model_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    speed_path = args.output_dir / "physical_step_model_speedup.csv"
    with speed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(speed_rows[0]))
        writer.writeheader()
        writer.writerows(speed_rows)
    selection_path = args.output_dir / "model_selection_evidence.csv"
    with selection_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0]))
        writer.writeheader()
        writer.writerows(selection_rows)
    strict_seed_result = None
    if args.seed_robustness_summary is not None:
        seed_summary = load_summary(args.seed_robustness_summary)
        strict_split = str(seed_summary.get("split_name", ""))
        seeds = [int(value) for value in seed_summary.get("seeds", [])]
        if strict_split != "pair_disjoint_stress_test":
            raise ValueError("seed robustness does not use pair_disjoint_stress_test")
        if strict_split not in args.split_names:
            raise ValueError("strict seed split is absent from the main model comparison")
        if len(set(seeds)) < 3:
            raise ValueError("strict split must contain at least three training seeds")
        metric_rows = seed_summary.get("metrics")
        if not isinstance(metric_rows, list) or not metric_rows:
            raise ValueError("seed robustness summary has no model metrics")
        for row in metric_rows:
            if int(row.get("seed_count", -1)) != len(seeds):
                raise ValueError("seed metric count differs from registered seeds")
        repeated_field_rows = [
            row
            for row in metric_rows
            if "solid_temperature" in str(row.get("metric", ""))
        ]
        if not repeated_field_rows:
            raise ValueError("seed robustness summary has no repeated field metric")
        best_repeated = min(
            repeated_field_rows, key=lambda row: float(row["mean_K"])
        )
        strict_seed_result = {
            "split_name": strict_split,
            "seeds": seeds,
            "metrics": metric_rows,
            "lowest_three_seed_mean_field_model": best_repeated["model"],
            "lowest_three_seed_mean_field_RMSE_K": float(best_repeated["mean_K"]),
            "corresponding_sample_std_K": float(best_repeated["sample_std_K"]),
            "source_summary": project_relative(
                args.seed_robustness_summary, project_root
            ),
        }

    summary = {
        "status": "completed_p418_physical_step_model_comparison",
        "splits": list(args.split_names),
        "comparison_order": [
            "observable_transformer",
            "initial_temperature_persistence",
            "dmdc",
            "graph_transformer_data_only",
            "graph_transformer_energy_flux",
            "graph_transformer_factorized_energy_flux",
            "low_rank_residual_correction",
            "diffusion_residual_correction",
        ],
        "lowest_test_temperature_RMSE_model_by_split": best_by_split,
        "diffusion_temperature_and_energy_effect_by_split": (
            diffusion_energy_effect_by_split
        ),
        "temperature_energy_pareto_models_by_split": (
            temperature_energy_pareto_by_split
        ),
        "energy_metrics_rejected_outside_registered_temperature_range_by_split": (
            energy_range_rejections_by_split
        ),
        "strict_split_seed_robustness": strict_seed_result,
        "selected_model_sources_by_split": selected_model_sources_by_split,
        "strict_split_loss_balancing_integration_record": (
            project_relative(strict_integration_path, project_root)
            if STRICT_SPLIT in args.split_names and strict_loss_selection_ready
            else None
        ),
        "strict_split_loss_balancing_stage": (
            "validation_selected"
            if strict_loss_selection_ready
            else "not_applicable"
        ),
        "metric_table": csv_path.name,
        "speed_table": speed_path.name,
        "selection_table": selection_path.name,
        "speed_comparison_note": (
            "OpenFOAM values are 32-MPI-rank CPU wall-clock times from each step log; model "
            "values are prediction times on the recorded device. Neural model size is the number "
            "of trainable parameters; DMDc size is the number of stored scalar coefficients. "
            "Low-rank and diffusion rows report the complete graph--Transformer-plus-correction "
            "chain rather than correction-only time or size. Training-only break-even counts "
            "include measured model-training time. Full-workflow break-even counts additionally "
            "include the measured OpenFOAM wall time for the registered training and validation "
            "curves, and use the held-out median OpenFOAM time as the future-case reference."
        ),
        "fair_comparison_checks": {
            "same_complete_curve_splits": True,
            "same_graph_transformer_architecture": True,
            "same_graph_transformer_training_normalization": True,
            "factorized_variant_changes_only_spatial_timing": True,
            "factorized_variant_uses_selected_loss_weights": (
                strict_loss_selection_ready
            ),
            "low_rank_uses_physics_constrained_prediction": True,
            "diffusion_uses_physics_constrained_prediction": True,
            "diffusion_reports_energy_balance_before_and_after_correction": True,
            "all_regional_field_models_use_common_energy_balance": (
                not energy_range_rejections_by_split
            ),
            "out_of_range_predictions_are_not_clipped_or_extrapolated_for_energy_metrics": True,
            "all_models_selected_using_validation_curves": True,
            "test_curves_used_only_for_final_metrics": True,
            "correction_speed_includes_upstream_prediction": True,
            "break_even_uses_measured_training_and_openfoam_times": True,
            "field_ranking_metric": "regional-volume-weighted solid-temperature RMSE in K",
            "physical_energy_metric": PRIMARY_ENERGY_METRIC,
            "held_out_interpretation_rule": (
                "Temperature RMSE and the common transient energy-equation residual are "
                "reported together. The held-out result is described as a joint improvement "
                "only when solid-temperature RMSE decreases and the projection-aware energy "
                "residual does not increase. This label is not used for checkpoint or model "
                "selection; all held-out outcomes remain in the comparison table."
            ),
            "pair_disjoint_split_in_main_table": (
                "pair_disjoint_stress_test" in args.split_names
            ),
            "pair_disjoint_split_has_three_seed_summary": strict_seed_result is not None,
        },
        "interpretation": (
            "The data-only and energy/flux-constrained graph--Transformers use the same "
            "architecture and curve splits. Diffusion corrects the residual after the "
            "energy/flux-constrained deterministic model. The factorized variant encodes "
            "the fixed spatial state once but retains all times and physical residuals. "
            "The training-only low-rank residual tests whether diffusion adds value beyond "
            "repeatable deterministic discrepancy. On held-out curves, diffusion is described "
            "as a joint temperature--energy improvement only when it lowers solid-temperature "
            "error without increasing the common transient fluid/solid energy-equation residual. "
            "This post-prediction description does not alter the validation-selected model."
            " If a prediction leaves the specified thermophysical temperature range, its "
            "temperature error and range violation remain reported, but no clipped or "
            "extrapolated energy-equation metric is assigned and it is omitted from the "
            "temperature--energy Pareto set."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
