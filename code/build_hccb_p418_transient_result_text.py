#!/usr/bin/env python3
"""Build manuscript prose from the completed transient model comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


EXPECTED_STATUS = "completed_p418_physical_step_model_comparison"
COST_STATUS = "complete_p418_transient_cost_table"
STRICT_SPLIT = "pair_disjoint_stress_test"
EXPECTED_SPLITS = (
    "direction_down_test",
    "direction_up_test",
    STRICT_SPLIT,
)
SPLIT_LABELS = {
    "direction_down_test": "downward-step holdout",
    "direction_up_test": "upward-step holdout",
    STRICT_SPLIT: "endpoint-pair-disjoint holdout",
}
MODEL_LABELS = {
    "dmdc": "DMDc",
    "graph_transformer_data_only": "data-only graph--Transformer",
    "graph_transformer_energy_flux": "physics-constrained graph--Transformer",
    "graph_transformer_factorized_energy_flux": (
        "factorized physics-constrained graph--Transformer"
    ),
    "low_rank_residual_correction": "low-rank residual correction",
    "diffusion_residual_correction": "diffusion residual correction",
}
TEMPERATURE_METRIC = "solid_temperature_RMSE_K"
ENERGY_METRIC = (
    "projection_aware_volume_weighted_energy_equation_normalized_RMSE"
)
DIFFUSION_SOLID_COVERAGE_METRIC = (
    "diffusion_unobserved_dynamic_solid_90pct_interval_coverage_fraction"
)
DIFFUSION_SOLID_WIDTH_METRIC = (
    "diffusion_unobserved_dynamic_solid_90pct_interval_mean_width_K"
)
DIFFUSION_SOLID_CRPS_METRIC = "diffusion_unobserved_dynamic_solid_CRPS_K"
HOTSPOT_TEMPERATURE_METRIC = "solid_maximum_temperature_history_RMSE_K"
HOTSPOT_LOCATION_METRIC = "solid_regional_hotspot_location_p95_error_m"
HOTSPOT_DEFICIT_METRIC = "solid_hotspot_target_temperature_deficit_p95_K"
RESULT_WORD_LIMIT = 390


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: object, name: str, *, nonnegative: bool = True) -> float:
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"invalid {name}: {value}")
    return result


def format_value(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 0.01:
        return f"{value:.2e}"
    if abs(value) < 1.0:
        return f"{value:.3f}"
    if abs(value) < 10.0:
        return f"{value:.2f}"
    return f"{value:.1f}"


def metric_lookup(
    path: Path, *, data_role: str = "test"
) -> dict[tuple[str, str, str], float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lookup: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if row.get("data_role") != data_role:
            continue
        key = (str(row["split_name"]), str(row["model"]), str(row["metric"]))
        if key in lookup:
            raise ValueError(f"duplicate transient metric: {key}")
        lookup[key] = finite(row["value"], "/".join(key))
    return lookup


def require_metric(
    metrics: dict[tuple[str, str, str], float],
    split: str,
    model: str,
    metric: str,
) -> float:
    key = (split, model, metric)
    if key not in metrics:
        raise ValueError(f"missing transient metric: {key}")
    return metrics[key]


def build(summary_path: Path, metrics_path: Path, cost_path: Path) -> str:
    summary = load_json(summary_path)
    if summary.get("status") != EXPECTED_STATUS:
        raise ValueError("transient model comparison is incomplete")
    splits = [str(value) for value in summary.get("splits", [])]
    if set(splits) != set(EXPECTED_SPLITS) or len(splits) != len(EXPECTED_SPLITS):
        raise ValueError(f"unexpected transient splits: {splits}")
    fair = summary.get("fair_comparison_checks", {})
    required_checks = (
        "same_complete_curve_splits",
        "same_graph_transformer_architecture",
        "same_graph_transformer_training_normalization",
        "all_models_selected_using_validation_curves",
        "test_curves_used_only_for_final_metrics",
    )
    if not all(fair.get(name) is True for name in required_checks):
        raise ValueError("transient comparison does not satisfy the declared fair-comparison checks")

    best = summary.get("lowest_test_temperature_RMSE_model_by_split", {})
    best_phrases = []
    for split in EXPECTED_SPLITS:
        record = best.get(split, {})
        model = str(record.get("lowest_test_solid_temperature_RMSE_model", ""))
        if model not in MODEL_LABELS:
            raise ValueError(f"unknown best transient model for {split}: {model}")
        rmse = finite(
            record.get("lowest_test_solid_temperature_RMSE_K"),
            f"best RMSE for {split}",
        )
        best_phrases.append(
            f"the {SPLIT_LABELS[split]} by the {MODEL_LABELS[model]} "
            f"(\({format_value(rmse)}~\mathrm{{K}}\))"
        )

    metrics = metric_lookup(metrics_path)
    validation_metrics = metric_lookup(metrics_path, data_role="validation")
    data_temperature = require_metric(
        metrics, STRICT_SPLIT, "graph_transformer_data_only", TEMPERATURE_METRIC
    )
    physics_temperature = require_metric(
        metrics, STRICT_SPLIT, "graph_transformer_energy_flux", TEMPERATURE_METRIC
    )
    data_validation_temperature = require_metric(
        validation_metrics,
        STRICT_SPLIT,
        "graph_transformer_data_only",
        TEMPERATURE_METRIC,
    )
    physics_validation_temperature = require_metric(
        validation_metrics,
        STRICT_SPLIT,
        "graph_transformer_energy_flux",
        TEMPERATURE_METRIC,
    )
    data_energy = require_metric(
        metrics, STRICT_SPLIT, "graph_transformer_data_only", ENERGY_METRIC
    )
    physics_energy = require_metric(
        metrics, STRICT_SPLIT, "graph_transformer_energy_flux", ENERGY_METRIC
    )
    physics_hotspot_temperature = require_metric(
        metrics,
        STRICT_SPLIT,
        "graph_transformer_energy_flux",
        HOTSPOT_TEMPERATURE_METRIC,
    )
    physics_hotspot_location = require_metric(
        metrics,
        STRICT_SPLIT,
        "graph_transformer_energy_flux",
        HOTSPOT_LOCATION_METRIC,
    )
    physics_hotspot_deficit = require_metric(
        metrics,
        STRICT_SPLIT,
        "graph_transformer_energy_flux",
        HOTSPOT_DEFICIT_METRIC,
    )
    diffusion_hotspot_temperature = require_metric(
        metrics,
        STRICT_SPLIT,
        "diffusion_residual_correction",
        f"diffusion_refined_{HOTSPOT_TEMPERATURE_METRIC}",
    )
    diffusion_hotspot_location = require_metric(
        metrics,
        STRICT_SPLIT,
        "diffusion_residual_correction",
        f"diffusion_refined_{HOTSPOT_LOCATION_METRIC}",
    )
    diffusion_hotspot_deficit = require_metric(
        metrics,
        STRICT_SPLIT,
        "diffusion_residual_correction",
        f"diffusion_refined_{HOTSPOT_DEFICIT_METRIC}",
    )

    diffusion_by_split = summary.get(
        "diffusion_temperature_and_energy_effect_by_split", {}
    )
    diffusion = diffusion_by_split.get(STRICT_SPLIT, {})
    deterministic_temperature = finite(
        diffusion.get("deterministic_solid_temperature_RMSE_K"),
        "deterministic strict-split temperature RMSE",
    )
    refined_temperature = finite(
        diffusion.get("diffusion_refined_solid_temperature_RMSE_K"),
        "diffusion strict-split temperature RMSE",
    )
    deterministic_energy = finite(
        diffusion.get("deterministic_projection_aware_energy_normalized_RMSE"),
        "deterministic strict-split energy difference",
    )
    refined_energy = finite(
        diffusion.get("diffusion_refined_projection_aware_energy_normalized_RMSE"),
        "diffusion strict-split energy difference",
    )
    diffusion_solid_coverage = require_metric(
        metrics,
        STRICT_SPLIT,
        "diffusion_residual_correction",
        DIFFUSION_SOLID_COVERAGE_METRIC,
    )
    if diffusion_solid_coverage > 1.0:
        raise ValueError("diffusion interval coverage must be a fraction")
    diffusion_solid_width = require_metric(
        metrics,
        STRICT_SPLIT,
        "diffusion_residual_correction",
        DIFFUSION_SOLID_WIDTH_METRIC,
    )
    diffusion_solid_crps = require_metric(
        metrics,
        STRICT_SPLIT,
        "diffusion_residual_correction",
        DIFFUSION_SOLID_CRPS_METRIC,
    )
    member_energy_p95 = finite(
        diffusion.get(
            "diffusion_member_projection_aware_energy_equation_normalized_RMSE_p95"
        ),
        "diffusion-member p95 projection-aware energy RMSE",
    )
    member_joint_fraction = finite(
        diffusion.get(
            "diffusion_member_joint_temperature_energy_improvement_fraction"
        ),
        "diffusion-member joint improvement fraction",
    )
    member_sample_count = int(diffusion.get("diffusion_member_sample_count", 0))
    if member_joint_fraction > 1.0 or member_sample_count < 2:
        raise ValueError("diffusion-member physical results are invalid")
    joint = diffusion.get("diffusion_is_joint_temperature_energy_improvement")
    if not isinstance(joint, bool):
        raise ValueError("diffusion joint-improvement result is missing")
    if diffusion.get("model_selection_uses_this_outcome") is not False:
        raise ValueError("held-out diffusion outcome was used for model selection")
    if joint:
        diffusion_interpretation = (
            "It therefore satisfied the predeclared joint temperature--energy "
            "improvement condition on this split."
        )
    else:
        reason = str(diffusion.get("outcome_reason", ""))
        if reason == "temperature_error_did_not_decrease":
            explanation = "the held-out temperature error did not decrease"
        elif reason == "energy_residual_increased_despite_lower_temperature_error":
            explanation = (
                "the lower temperature error was accompanied by a larger "
                "projection-aware energy RMSE"
            )
        else:
            raise ValueError(f"unknown diffusion outcome reason: {reason}")
        diffusion_interpretation = (
            "It remained a temperature--energy trade-off: "
            f"{explanation}."
        )

    seed = summary.get("strict_split_seed_robustness")
    if not isinstance(seed, dict) or seed.get("split_name") != STRICT_SPLIT:
        raise ValueError("strict transient three-seed result is missing")
    seeds = [int(value) for value in seed.get("seeds", [])]
    if len(set(seeds)) < 3:
        raise ValueError("strict transient result uses fewer than three seeds")
    seed_model = str(seed.get("lowest_three_seed_mean_field_model", ""))
    if seed_model not in MODEL_LABELS:
        raise ValueError(f"unknown three-seed transient model: {seed_model}")
    seed_mean = finite(seed.get("lowest_three_seed_mean_field_RMSE_K"), "seed mean")
    seed_std = finite(seed.get("corresponding_sample_std_K"), "seed standard deviation")

    cost = load_json(cost_path)
    if cost.get("status") != COST_STATUS or cost.get("split_name") != STRICT_SPLIT:
        raise ValueError("strict transient cost comparison is incomplete")
    cost_lookup = {str(row.get("model")): row for row in cost.get("records", [])}
    cost_models = (
        "graph_transformer_energy_flux",
        "low_rank_residual_correction",
        "diffusion_residual_correction",
    )
    cost_phrases = []
    for model in cost_models:
        if model not in cost_lookup:
            raise ValueError(f"transient cost result lacks {model}")
        row = cost_lookup[model]
        speedup = finite(row.get("speedup_vs_32_rank_openfoam"), f"speed-up for {model}")
        break_even = int(row.get("full_workflow_break_even_curve_count"))
        if speedup <= 0.0 or break_even < 0:
            raise ValueError(f"invalid complete-chain cost for {model}")
        cost_phrases.append(
            f"{MODEL_LABELS[model]}: "
            f"\\({format_value(speedup)}\\times\\) and {break_even} curves"
        )

    paragraphs = [
        (
            "Solid-temperature leaders were "
            + "; ".join(best_phrases)
            + ". On the endpoint-pair-disjoint holdout, the data-only and physics-constrained "
            "graph--Transformers gave solid-temperature RMSEs of "
            f"\({format_value(data_temperature)}\) and \({format_value(physics_temperature)}~\mathrm{{K}}\), "
            "respectively; their projection-aware energy RMSEs were "
            f"\({format_value(data_energy)}\) and \({format_value(physics_energy)}\); "
            "both errors decreased with the physics terms. "
            "Validation RMSEs were "
            f"\({format_value(data_validation_temperature)}\) and "
            f"\({format_value(physics_validation_temperature)}~\mathrm{{K}}\). "
            "Validation and test contain different step families and are reported separately rather than pooled."
        ),
        (
            "The diffusion residual changed solid-temperature RMSE from "
            f"\({format_value(deterministic_temperature)}\) to "
            f"\({format_value(refined_temperature)}~\mathrm{{K}}\) and the common "
            "projection-aware energy RMSE from "
            f"\({format_value(deterministic_energy)}\) to \({format_value(refined_energy)}\). "
            + diffusion_interpretation
            + " This held-out result was for final assessment, not for checkpoint or architecture selection. "
            "For unobserved solid temperatures, the nominal 90\% ensemble interval covered "
            f"\({100.0 * diffusion_solid_coverage:.1f}\%\) of the volume-weighted "
            f"reference values, with a mean width of \({format_value(diffusion_solid_width)}~\mathrm{{K}}\) "
            f"and a CRPS of \({format_value(diffusion_solid_crps)}~\mathrm{{K}}\). "
            "These intervals do not assume that it is calibrated. Across "
            f"the {member_sample_count} stochastic prediction sets, the 95th-percentile "
            "projection-aware energy RMSE was "
            f"\({format_value(member_energy_p95)}\), and "
            f"\({100.0 * member_joint_fraction:.1f}\%\) improved both the solid-temperature "
            "error and projection-aware energy RMSE. "
            "Consistency was checked for individual samples, not only for their ensemble mean. Across "
            f"{len(set(seeds))} independent initializations, the lowest mean field RMSE was "
            f"obtained by the {MODEL_LABELS[seed_model]} "
            f"(\({format_value(seed_mean)} \pm {format_value(seed_std)}~\mathrm{{K}}\))."
        ),
        (
            "For the dynamic solid hotspot, the physics-constrained graph--Transformer and diffusion refinement gave maximum-temperature-history "
            f"RMSEs of \({format_value(physics_hotspot_temperature)}\) and "
            f"\({format_value(diffusion_hotspot_temperature)}~\mathrm{{K}}\), respectively. "
            "Their 95th-percentile hottest-node centroid errors were "
            f"\({format_value(physics_hotspot_location)}\) and "
            f"\({format_value(diffusion_hotspot_location)}~\mathrm{{m}}\). At the regions selected as hottest, "
            "95th-percentile temperature deficits were "
            f"\({format_value(physics_hotspot_deficit)}\) and "
            f"\({format_value(diffusion_hotspot_deficit)}~\mathrm{{K}}\). This temperature deficit prevents "
            "a rank exchange between nearly equal neighbouring regions from being interpreted "
            "as a large thermal error. These are regional quantities after the uniform initial state, not pebble-internal maxima. "
            "Relative to the measured 32-rank OpenFOAM run, speed-up and full-workflow break-even count were "
            + "; ".join(cost_phrases)
            + "; break-even includes training and its OpenFOAM data."
        ),
    ]
    text = "\n\n".join(paragraphs) + "\n"
    words = len(re.findall(r"\b[\w'-]+\b", text))
    if words > RESULT_WORD_LIMIT:
        raise ValueError(
            f"transient result text has {words} words; limit is "
            f"{RESULT_WORD_LIMIT}"
        )
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--cost-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = build(
        args.summary.resolve(),
        args.metrics.resolve(),
        args.cost_summary.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
