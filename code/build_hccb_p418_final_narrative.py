#!/usr/bin/env python3
"""Build concise final paper sections from the accepted P418 evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


STRICT_SPLIT = "pair_disjoint_stress_test"
TEMPERATURE_METRIC = "solid_temperature_RMSE_K"
ENERGY_METRIC = (
    "projection_aware_volume_weighted_energy_equation_normalized_RMSE"
)
STEADY_METRIC = "test_solid_temperature_normalized_rmse"
TRANSIENT_MODELS = (
    "graph_transformer_data_only",
    "graph_transformer_energy_flux",
    "graph_transformer_factorized_energy_flux",
)
ALL_TRANSIENT_FIELD_MODELS = (
    "initial_temperature_persistence",
    "dmdc",
    *TRANSIENT_MODELS,
    "low_rank_residual_correction",
    "diffusion_residual_correction",
)
TRANSIENT_TEMPERATURE_METRICS = {
    model: TEMPERATURE_METRIC for model in ALL_TRANSIENT_FIELD_MODELS
}
TRANSIENT_TEMPERATURE_METRICS["diffusion_residual_correction"] = (
    "diffusion_refined_solid_temperature_RMSE_K"
)
TRANSIENT_LABELS = {
    "initial_temperature_persistence": "initial-temperature persistence",
    "dmdc": "DMDc",
    "graph_transformer_data_only": "data-only graph--Transformer",
    "graph_transformer_energy_flux": (
        "physics-constrained graph--Transformer"
    ),
    "graph_transformer_factorized_energy_flux": (
        "factorized physics-constrained graph--Transformer"
    ),
    "low_rank_residual_correction": "POD residual correction",
    "diffusion_residual_correction": "diffusion residual correction",
}
HIGH_RE_MODELS = ("data_only", "physics_constrained", "factorized")
HIGH_RE_LABELS = {
    "data_only": "data-only graph--Transformer",
    "physics_constrained": "physics-constrained graph--Transformer",
    "factorized": "factorized graph--Transformer",
}
STEADY_LABELS = {
    "response_surface": "response surface",
    "pinn_data_only": "data-only PINN",
    "pinn": "physics-informed PINN",
    "graph": "graph operator",
    "transolver": "Physics-Attention operator",
}
STEADY_SEED_METRIC = "solid_temperature_normalized_rmse"
EXPECTED_STEADY_LEARNING_COUNTS = (9, 18, 27, 36)
EXPECTED_TRANSIENT_LEARNING_COUNTS = (3, 6)
LOSS_BALANCING_LABELS = {
    "fixed_registered_5_1_1": "fixed 5:1:1 weighting",
    "relobralo_burgers_table_viii": "the Burgers ReLoBRaLo setting",
    "relobralo_kirchhoff_table_viii": "the Kirchhoff ReLoBRaLo setting",
    "relobralo_helmholtz_table_viii": "the Helmholtz ReLoBRaLo setting",
}
SECTION_WORD_LIMITS = {
    "abstract": 250,
    "discussion": 450,
    "conclusion": 170,
}


def load_json(path: Path, expected_status: str | None = None) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    if expected_status is not None and payload.get("status") != expected_status:
        raise ValueError(
            f"unexpected status in {path}: {payload.get('status')}"
        )
    return payload


def finite(value: object, name: str, *, nonnegative: bool = True) -> float:
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"invalid {name}: {value}")
    return result


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def enforce_section_lengths(sections: dict[str, str]) -> dict[str, int]:
    counts = {name: word_count(text) for name, text in sections.items()}
    for name, limit in SECTION_WORD_LIMITS.items():
        count = counts[name]
        if count > limit:
            raise ValueError(
                f"{name} has {count} words; the manuscript limit is {limit}"
            )
    discussion_paragraphs = len(
        [part for part in sections["discussion"].split("\n\n") if part.strip()]
    )
    if discussion_paragraphs > 4:
        raise ValueError(
            "discussion has more than four paragraphs; condense the argument"
        )
    return counts


def loss_balancing_result(selection_path: Path) -> dict[str, object]:
    selection = load_json(
        selection_path,
        "p418_loss_balancing_selected_on_validation_only",
    )
    records = selection.get("candidate_records")
    selected_id = str(selection.get("selected_candidate_id", ""))
    if (
        not isinstance(records, list)
        or len(records) != 4
        or selected_id not in LOSS_BALANCING_LABELS
        or selection.get("independent_test_read") is not False
        or selection.get("new_physical_parameters") != []
    ):
        raise ValueError("loss-balancing validation selection is incomplete")
    candidate_ids = {
        str(row.get("candidate_id", ""))
        for row in records
        if isinstance(row, dict)
    }
    if candidate_ids != set(LOSS_BALANCING_LABELS):
        raise ValueError("loss-balancing candidate set differs from the protocol")
    selected_score = finite(
        selection.get("selected_validation_score"),
        "selected loss-balancing validation score",
    )
    final_path = selection_path.parent / selected_id / "final_summary.json"
    final = load_json(
        final_path,
        "completed_p418_spatiotemporal_regional_operator",
    )
    if (
        final.get("evaluation_stage") != "final"
        or final.get("test_evaluated") is not True
        or final.get("new_physical_parameters") != []
        or not isinstance(final.get("loss_balancing"), dict)
        or final["loss_balancing"].get("candidate_id") != selected_id
    ):
        raise ValueError("selected loss-balancing final test is incomplete")
    test = final.get("metrics", {}).get("test", {})
    if not isinstance(test, dict):
        raise ValueError("selected loss-balancing test metrics are missing")
    return {
        "candidate_id": selected_id,
        "label": LOSS_BALANCING_LABELS[selected_id],
        "validation_selection_score": selected_score,
        "test_solid_temperature_RMSE_K": finite(
            test.get("solid_temperature_RMSE_K"),
            "loss-balancing test solid-temperature RMSE",
        ),
        "test_projection_aware_energy_normalized_RMSE": finite(
            test.get("projection_aware_energy_equation_normalized_RMSE"),
            "loss-balancing test projection-aware energy nRMSE",
        ),
    }


def fmt(value: float) -> str:
    absolute = abs(value)
    if value == 0.0:
        return "0"
    if absolute < 0.01:
        return f"{value:.2e}"
    if absolute < 1.0:
        return f"{value:.3f}"
    if absolute < 10.0:
        return f"{value:.2f}"
    return f"{value:.1f}"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def test_metric_lookup(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        if (
            row.get("split_name") != STRICT_SPLIT
            or row.get("data_role") != "test"
        ):
            continue
        model = str(row.get("model", ""))
        metric = str(row.get("metric", ""))
        if model not in ALL_TRANSIENT_FIELD_MODELS:
            continue
        quantity = None
        if metric == TRANSIENT_TEMPERATURE_METRICS[model]:
            quantity = "solid_temperature_RMSE_K"
        elif metric == ENERGY_METRIC:
            quantity = "projection_aware_energy_normalized_RMSE"
        if quantity is None:
            continue
        if quantity in result.setdefault(model, {}):
            raise ValueError(f"duplicate transient metric: {model}/{quantity}")
        result[model][quantity] = finite(
            row.get("value"), f"{model}/{quantity}"
        )
    expected = {
        model: {
            "solid_temperature_RMSE_K",
            "projection_aware_energy_normalized_RMSE",
        }
        for model in ALL_TRANSIENT_FIELD_MODELS
    }
    if set(result) != set(expected) or any(
        set(result[model]) != quantities
        for model, quantities in expected.items()
    ):
        raise ValueError(
            f"strict transient metrics are incomplete: {result.keys()}"
        )
    return result


def high_re_lookup(path: Path) -> dict[str, dict[str, float | None]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result: dict[str, dict[str, float | None]] = {}
    for row in rows:
        metric = str(row.get("metric", ""))
        if not metric or metric in result:
            raise ValueError(f"invalid or duplicate high-Re metric: {metric}")
        model_values: dict[str, float | None] = {}
        for model in HIGH_RE_MODELS:
            raw = str(row.get(model, "")).strip()
            model_values[model] = (
                None if not raw else finite(raw, f"{metric}/{model}")
            )
        result[metric] = model_values
    required = {
        "fluid_temperature_volume_weighted_RMSE_K",
        "solid_temperature_volume_weighted_RMSE_K",
        "solid_maximum_temperature_history_RMSE_K",
        "solid_regional_hotspot_location_mean_error_m",
        "solid_regional_hotspot_location_p95_error_m",
    }
    if not required.issubset(result):
        raise ValueError(f"high-Re metrics are incomplete: {set(result)}")
    for metric in required:
        if any(result[metric][model] is None for model in HIGH_RE_MODELS):
            raise ValueError(f"high-Re metric is incomplete: {metric}")
    return result


def dominant_steady_effects(completed: dict) -> dict[str, str]:
    records = completed.get("factorial_variance_decomposition")
    if (
        completed.get("complete_factorial_decomposition_available") is not True
        or not isinstance(records, list)
    ):
        raise ValueError("complete steady factorial decomposition is missing")
    observables = (
        "pressure_drop_Pa",
        "outlet_temperature_K",
        "solid_maximum_temperature_K",
        "cooling_wall_heat_into_fluid_W",
    )
    effects = (
        "inlet_velocity",
        "inlet_temperature",
        "solid_heat_source",
    )
    lookup = {
        (str(row.get("observable")), str(row.get("effect"))): finite(
            row.get("variance_fraction_percent"),
            "factorial variance fraction",
        )
        for row in records
        if isinstance(row, dict)
    }
    result = {}
    for observable in observables:
        values = {
            effect: lookup.get((observable, effect)) for effect in effects
        }
        if any(value is None for value in values.values()):
            raise ValueError(
                f"steady main effects are incomplete for {observable}"
            )
        result[observable] = max(
            values, key=lambda effect: float(values[effect])
        )
    return result


def cost_lookup(cost: dict) -> dict[str, dict]:
    if (
        cost.get("status") != "complete_p418_transient_cost_table"
        or cost.get("split_name") != STRICT_SPLIT
    ):
        raise ValueError("strict transient cost comparison is incomplete")
    records = cost.get("records")
    if not isinstance(records, list):
        raise ValueError("transient cost records are missing")
    return {str(row.get("model")): row for row in records}


def seed_and_learning_summary(
    *,
    steady_seed: dict,
    transient_seed: dict,
    steady_learning: dict,
    steady_learning_path: Path,
    transient_learning: dict,
    steady_model: str,
) -> dict[str, float | list[float]]:
    if steady_seed.get("new_physical_parameters") != []:
        raise ValueError("steady seed comparison adds physical parameters")
    steady_seed_rows = steady_seed.get("metrics")
    if (
        not isinstance(steady_seed_rows, list)
        or len(steady_seed.get("seeds", [])) != 3
    ):
        raise ValueError("steady seed comparison is incomplete")
    steady_cv = []
    for row in steady_seed_rows:
        if row.get("metric") != STEADY_SEED_METRIC:
            continue
        mean = finite(row.get("mean"), "steady seed mean")
        sample_std = finite(
            row.get("sample_std"), "steady seed sample standard deviation"
        )
        if mean <= 0.0:
            raise ValueError("steady seed mean must be positive")
        steady_cv.append(100.0 * sample_std / mean)
    if len(steady_cv) != 4:
        raise ValueError("steady seed solid-temperature metrics are incomplete")

    if (
        transient_seed.get("new_physical_parameters") != []
        or len(transient_seed.get("seeds", [])) < 3
    ):
        raise ValueError("transient seed comparison is incomplete")
    transient_seed_rows = transient_seed.get("metrics")
    if not isinstance(transient_seed_rows, list) or len(
        transient_seed_rows
    ) < 5:
        raise ValueError("transient seed metrics are incomplete")
    transient_seed_std = [
        finite(
            row.get("sample_std_K"),
            "transient seed sample standard deviation",
        )
        for row in transient_seed_rows
    ]

    if steady_learning.get("new_physical_parameters") != []:
        raise ValueError("steady learning curve adds physical parameters")
    steady_counts = tuple(
        int(value)
        for value in steady_learning.get("training_condition_counts", [])
    )
    if steady_counts != EXPECTED_STEADY_LEARNING_COUNTS:
        raise ValueError("steady learning-curve counts are incomplete")
    table_name = str(steady_learning.get("table", ""))
    table_path = (steady_learning_path.parent / table_name).resolve()
    if not table_name or not table_path.is_file():
        raise ValueError("steady learning-curve table is missing")
    with table_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    steady_learning_values: dict[int, float] = {}
    for row in rows:
        if str(row.get("architecture")) != steady_model:
            continue
        count = int(row.get("train_case_count", -1))
        if count not in (
            EXPECTED_STEADY_LEARNING_COUNTS[0],
            EXPECTED_STEADY_LEARNING_COUNTS[-1],
        ):
            continue
        if count in steady_learning_values:
            raise ValueError("duplicate steady learning-curve row")
        steady_learning_values[count] = finite(
            row.get(STEADY_METRIC),
            f"steady learning curve/{steady_model}/{count}",
        )
    if set(steady_learning_values) != {
        EXPECTED_STEADY_LEARNING_COUNTS[0],
        EXPECTED_STEADY_LEARNING_COUNTS[-1],
    }:
        raise ValueError(
            f"steady learning curve lacks {steady_model} endpoint counts"
        )

    if transient_learning.get("new_physical_parameters") != []:
        raise ValueError("transient learning curve adds physical parameters")
    transient_counts = tuple(
        int(value)
        for value in transient_learning.get(
            "training_trajectory_counts", []
        )
    )
    if transient_counts != EXPECTED_TRANSIENT_LEARNING_COUNTS:
        raise ValueError("transient learning-curve counts are incomplete")
    transient_runs = transient_learning.get("runs")
    if not isinstance(transient_runs, list):
        raise ValueError("transient learning-curve runs are missing")
    three_curve_values = [
        finite(
            row.get("test_solid_temperature_RMSE_K"),
            "three-trajectory solid-temperature RMSE",
        )
        for row in transient_runs
        if int(row.get("training_trajectory_count", -1)) == 3
    ]
    six_curve_values = [
        finite(
            row.get("test_solid_temperature_RMSE_K"),
            "six-trajectory solid-temperature RMSE",
        )
        for row in transient_runs
        if int(row.get("training_trajectory_count", -1)) == 6
    ]
    if len(three_curve_values) != 2 or len(six_curve_values) != 1:
        raise ValueError("transient learning-curve directions are incomplete")

    return {
        "maximum_steady_seed_cv_percent": max(steady_cv),
        "maximum_transient_seed_std_K": max(transient_seed_std),
        "steady_learning_low_count_nrmse": steady_learning_values[
            EXPECTED_STEADY_LEARNING_COUNTS[0]
        ],
        "steady_learning_high_count_nrmse": steady_learning_values[
            EXPECTED_STEADY_LEARNING_COUNTS[-1]
        ],
        "transient_three_curve_RMSE_range_K": [
            min(three_curve_values),
            max(three_curve_values),
        ],
        "transient_six_curve_RMSE_K": six_curve_values[0],
    }


def build(args: argparse.Namespace) -> dict:
    steady = load_json(
        args.steady_summary, "complete_p418_steady_manuscript_text"
    )
    completed = load_json(
        args.completed_physics_summary,
        "completed_p418_case_physics_summarized",
    )
    if int(completed.get("completed_case_count", -1)) != 60:
        raise ValueError("final narrative requires all 60 steady fields")

    regimes = completed.get("thermal_regime_summary", {})
    directions = regimes.get("cooling_wall_heat_direction_counts", {})
    wall_to_fluid = int(directions.get("wall_to_fluid", 0))
    fluid_to_wall = int(directions.get("fluid_to_wall", 0))
    zero = int(directions.get("zero", 0))
    if wall_to_fluid + fluid_to_wall + zero != 60:
        raise ValueError("wall-heat direction counts do not cover 60 fields")
    solid_range = regimes.get("solid_maximum_temperature_range_K")
    if not isinstance(solid_range, list) or len(solid_range) != 2:
        raise ValueError("solid-temperature range is missing")
    solid_min = finite(solid_range[0], "minimum solid temperature")
    solid_max = finite(solid_range[1], "maximum solid temperature")
    if solid_max < solid_min:
        raise ValueError("solid-temperature range is reversed")

    crossings = completed.get("wall_heat_zero_crossings")
    if not isinstance(crossings, list) or not crossings:
        raise ValueError("wall-heat reversal records are missing")
    crossing_values = [
        finite(
            row.get("interpolated_zero_wall_heat_inlet_temperature_K"),
            "wall-heat reversal temperature",
        )
        for row in crossings
    ]
    crossing_min = min(crossing_values)
    crossing_max = max(crossing_values)
    dominant_effects = dominant_steady_effects(completed)

    hotspots = load_json(
        args.steady_hotspot_summary, "p418_steady_hotspots_ready"
    )
    if int(hotspots.get("completed_case_count", -1)) != 60:
        raise ValueError("steady hotspot result does not cover 60 fields")
    hotspot_shifts = [
        finite(
            row.get("maximum_adjacent_hotspot_distance_m"),
            "steady hotspot displacement",
        )
        for row in hotspots.get("factor_summary", [])
        if isinstance(row, dict)
    ]
    if len(hotspot_shifts) < 3:
        raise ValueError("steady hotspot-factor results are incomplete")
    maximum_hotspot_shift = max(hotspot_shifts)

    leader = steady.get("leaders_by_metric", {}).get(STEADY_METRIC, {})
    steady_model = str(leader.get("method", ""))
    if steady_model not in STEADY_LABELS:
        raise ValueError(f"unknown steady model: {steady_model}")
    steady_error = finite(leader.get("value"), "steady solid-temperature nRMSE")
    robustness = seed_and_learning_summary(
        steady_seed=load_json(
            args.steady_seed_robustness,
            "completed_p418_main_steady_split_seed_robustness",
        ),
        transient_seed=load_json(
            args.transient_seed_robustness,
            "completed_p418_strict_split_seed_robustness",
        ),
        steady_learning=load_json(
            args.steady_learning_curve,
            "p418_steady_learning_curve_complete",
        ),
        steady_learning_path=args.steady_learning_curve,
        transient_learning=load_json(
            args.transient_learning_curve,
            "completed_p418_transient_learning_curve",
        ),
        steady_model=steady_model,
    )
    loss_balancing = loss_balancing_result(args.loss_balancing_selection)

    transient = load_json(
        args.transient_summary,
        "completed_p418_physical_step_model_comparison",
    )
    metrics = test_metric_lookup(args.transient_metrics)
    all_transient_temperatures = {
        model: metrics[model]["solid_temperature_RMSE_K"]
        for model in ALL_TRANSIENT_FIELD_MODELS
    }
    all_transient_energies = {
        model: metrics[model]["projection_aware_energy_normalized_RMSE"]
        for model in ALL_TRANSIENT_FIELD_MODELS
    }
    transient_temperatures = {
        model: all_transient_temperatures[model]
        for model in TRANSIENT_MODELS
    }
    transient_energies = {
        model: all_transient_energies[model]
        for model in TRANSIENT_MODELS
    }
    best_transient_model = min(
        ALL_TRANSIENT_FIELD_MODELS,
        key=lambda model: all_transient_temperatures[model],
    )
    recorded_best = transient.get(
        "lowest_test_temperature_RMSE_model_by_split", {}
    ).get(STRICT_SPLIT, {})
    if (
        recorded_best.get("lowest_test_solid_temperature_RMSE_model")
        != best_transient_model
        or not math.isclose(
            finite(
                recorded_best.get("lowest_test_solid_temperature_RMSE_K"),
                "recorded best transient temperature RMSE",
            ),
            all_transient_temperatures[best_transient_model],
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("strict transient leader differs from the metric table")

    diffusion = transient.get(
        "diffusion_temperature_and_energy_effect_by_split", {}
    ).get(STRICT_SPLIT, {})
    deterministic_temperature = finite(
        diffusion.get("deterministic_solid_temperature_RMSE_K"),
        "deterministic transient temperature RMSE",
    )
    refined_temperature = finite(
        diffusion.get("diffusion_refined_solid_temperature_RMSE_K"),
        "diffusion-refined temperature RMSE",
    )
    deterministic_energy = finite(
        diffusion.get(
            "deterministic_projection_aware_energy_normalized_RMSE"
        ),
        "deterministic transient energy difference",
    )
    refined_energy = finite(
        diffusion.get(
            "diffusion_refined_projection_aware_energy_normalized_RMSE"
        ),
        "diffusion-refined energy difference",
    )
    joint = diffusion.get(
        "diffusion_is_joint_temperature_energy_improvement"
    )
    if not isinstance(joint, bool):
        raise ValueError("diffusion joint-improvement result is missing")
    if diffusion.get("model_selection_uses_this_outcome") is not False:
        raise ValueError("held-out diffusion result was used for selection")
    diffusion_interval_coverage = finite(
        diffusion.get("diffusion_90pct_interval_coverage_fraction"),
        "diffusion 90-percent interval coverage",
    )
    diffusion_interval_width = finite(
        diffusion.get("diffusion_90pct_interval_mean_width_K"),
        "diffusion 90-percent interval mean width",
    )
    if not 0.0 <= diffusion_interval_coverage <= 1.0:
        raise ValueError("diffusion interval coverage is outside [0, 1]")
    if diffusion_interval_width < 0.0:
        raise ValueError("diffusion interval width is negative")

    costs = cost_lookup(load_json(args.transient_cost))
    if best_transient_model not in costs:
        raise ValueError("cost of the best transient model is missing")
    best_speedup = finite(
        costs[best_transient_model].get("speedup_vs_32_rank_openfoam"),
        "best transient-model speed-up",
    )
    best_break_even = int(
        costs[best_transient_model].get(
            "full_workflow_break_even_curve_count"
        )
    )
    if best_speedup <= 0.0 or best_break_even < 0:
        raise ValueError("invalid transient-model cost")

    high_re = load_json(
        args.high_re_comparison,
        "completed_p418_high_re_three_fixed_model_comparison",
    )
    if (
        int(high_re.get("curve_count", -1)) != 6
        or high_re.get("training_or_model_selection_performed") is not False
        or high_re.get("fully_coupled_model_used_for_accuracy_ranking")
        is not False
    ):
        raise ValueError("high-Re comparison is not the frozen six-curve test")
    high_re_metrics = high_re_lookup(args.high_re_aggregate)
    high_re_fluid = high_re_metrics[
        "fluid_temperature_volume_weighted_RMSE_K"
    ]
    high_re_solid = high_re_metrics[
        "solid_temperature_volume_weighted_RMSE_K"
    ]
    best_high_re_model = min(
        HIGH_RE_MODELS,
        key=lambda model: float(high_re_solid[model]),
    )
    best_high_re_fluid = float(high_re_fluid[best_high_re_model])
    best_high_re_solid = float(high_re_solid[best_high_re_model])

    packing = load_json(
        args.cross_packing_summary,
        "completed_seed101_seed202_integral_response_comparison",
    )
    if (
        packing.get("complete_nine_case_comparison") is not True
        or int(packing.get("accepted_common_case_count", -1)) != 9
        or int(packing.get("failed_seed202_case_count", -1)) != 0
    ):
        raise ValueError("independent-packing comparison is incomplete")
    packing_metrics = packing.get("metric_summary", {})
    packing_tout = finite(
        packing_metrics["outlet_temperature_K"][
            "maximum_absolute_relative_change_percent"
        ],
        "packing outlet-temperature change",
    )
    packing_tmax = finite(
        packing_metrics["maximum_solid_temperature_K"][
            "maximum_absolute_relative_change_percent"
        ],
        "packing maximum-temperature change",
    )
    packing_pressure = finite(
        packing_metrics["pressure_drop_Pa"][
            "maximum_absolute_relative_change_percent"
        ],
        "packing pressure-drop change",
    )

    external = load_json(
        args.external_evidence,
        "external_thermal_hydraulic_comparison_complete",
    )
    if (
        external.get("new_physical_parameters") != []
        or external.get("use_in_p418_training") is not False
    ):
        raise ValueError(
            "external comparisons must remain parameter-free test evidence"
        )
    hcpb = external.get("hcpb_annulus", {})
    fixed_bed = external.get("fixed_bed_pressure", {})
    premux = external.get("premux", {})
    tesomex = external.get("tesomex", {})
    if (
        int(hcpb.get("n", -1)) != 4
        or int(hcpb.get("points_inside_published_uncertainty", -1)) != 4
        or int(fixed_bed.get("n", -1)) != 6
        or int(premux.get("n", -1)) != 65
        or int(tesomex.get("snapshot_A", {}).get("n", -1)) != 7
        or int(tesomex.get("snapshot_B", {}).get("n", -1)) != 7
    ):
        raise ValueError("external thermal-hydraulic comparisons are incomplete")
    external_hcpb_error = finite(
        hcpb.get("mean_absolute_relative_error_percent"),
        "HELOKA/HCPB Nusselt mean absolute relative error",
    )
    external_pressure_error = finite(
        fixed_bed.get("median_absolute_relative_error_percent"),
        "fixed-bed pressure-gradient median absolute relative error",
    )
    external_premux_rmse = finite(
        premux.get("rmse"), "PREMUX thermocouple RMSE"
    )
    external_tesomex_rmse = [
        finite(tesomex["snapshot_A"].get("rmse"), "TESOMEX A RMSE"),
        finite(tesomex["snapshot_B"].get("rmse"), "TESOMEX B RMSE"),
    ]

    scope = load_json(
        args.scope_limits,
        "P418_SCOPE_LIMITS_EVIDENCE_SYNCED",
    )
    scope_records = scope.get("records")
    if not isinstance(scope_records, list) or len(scope_records) < 6:
        raise ValueError("scope-limit evidence is incomplete")
    mesh_failures = [
        row
        for row in scope_records
        if isinstance(row, dict)
        and row.get("status")
        == "hccb_pore_resolved_openfoam_mesh_preflight_failed"
    ]
    if not mesh_failures:
        raise ValueError("full-domain mesh limitation is missing")
    coupled_failures = [
        row
        for row in scope_records
        if isinstance(row, dict)
        and row.get("status") == "failed_solver_exit_propagated"
        and row.get("slurm_state") == "FAILED"
    ]
    coupled_job_ids = {
        str(row.get("job_id", "")) for row in coupled_failures
    }
    if len(coupled_job_ids - {""}) < 3:
        raise ValueError(
            "three independent fully coupled startup failures are required"
        )

    transient_triplet = " / ".join(
        fmt(transient_temperatures[model]) for model in TRANSIENT_MODELS
    )
    energy_triplet = " / ".join(
        fmt(transient_energies[model]) for model in TRANSIENT_MODELS
    )
    best_transient_sentence = (
        f"Across all field models, the {TRANSIENT_LABELS[best_transient_model]} "
        f"gives the lowest strict-split solid-temperature RMSE of "
        f"{fmt(all_transient_temperatures[best_transient_model])} K, with an "
        f"projection-aware energy RMSE of "
        f"{fmt(all_transient_energies[best_transient_model])}."
    )
    persistence_leads = best_transient_model == (
        "initial_temperature_persistence"
    )
    cost_sentence = (
        "This reference requires no training and therefore has no training "
        "break-even penalty. "
        if persistence_leads
        else (
            "Its complete-chain inference is "
            f"{fmt(best_speedup)} times faster than the measured 32-rank "
            "OpenFOAM runs, with "
            "training and reference-data cost recovered after "
            f"{best_break_even} predicted trajectories. "
        )
    )
    diffusion_sentence = (
        "Diffusion refinement reduced both the temperature error and the "
        "projection-aware energy RMSE."
        if joint
        else "Diffusion refinement did not improve temperature and energy "
        "simultaneously and is retained as a trade-off."
    )
    diffusion_uncertainty_sentence = (
        "The nominal 90\\% interval covered "
        f"{fmt(100.0 * diffusion_interval_coverage)}\\% "
        f"(mean width {fmt(diffusion_interval_width)} K)"
        + (
            ", indicating under-dispersion."
            if diffusion_interval_coverage < 0.85
            else "."
        )
    )
    abstract = (
        "Pore-scale heat transfer in solid-breeder pebble beds combines "
        "helium advection, conduction within ceramic pebbles, internal "
        "heating and wall cooling. We compare physics-informed and "
        "data-driven reduced models using 60 three-dimensional steady "
        "conjugate-heat-transfer fields and 12 complete fixed-flow "
        "thermal trajectories generated from literature-derived conditions. "
        "Complete conditions and endpoint pairs are withheld from fitting. "
        f"The computed maximum solid temperature spans {fmt(solid_min)}--"
        f"{fmt(solid_max)} K, and wall heat reverses direction at inlet "
        f"temperatures of {fmt(crossing_min)}--{fmt(crossing_max)} K. "
        f"Across five steady tests, the lowest worst-case solid-temperature "
        f"nRMSE is {fmt(steady_error)} for the "
        f"{STEADY_LABELS[steady_model]}. On the endpoint-pair-disjoint "
        "transient test, data-only, physics-constrained and factorized "
        f"graph--Transformers give solid-temperature RMSEs of "
        f"{transient_triplet} K and projection-aware energy RMSEs of "
        f"{energy_triplet} after projection onto the common regional basis, "
        f"respectively. {best_transient_sentence} "
        f"{diffusion_sentence} "
        f"The best frozen model on six independent high-velocity histories "
        f"is the {HIGH_RE_LABELS[best_high_re_model]}, with fluid- and "
        f"solid-temperature RMSEs of {fmt(best_high_re_fluid)} and "
        f"{fmt(best_high_re_solid)} K. Across nine matched conditions on an "
        "independent spherical-pebble arrangement, outlet and maximum-solid "
        f"temperatures change by at most {fmt(packing_tout)}\\% and "
        f"{fmt(packing_tmax)}\\%, whereas pressure drop changes by "
        f"{fmt(packing_pressure)}\\%. External HELOKA Nusselt and 1-mm "
        "fixed-bed pressure-gradient comparisons give mean and median "
        f"absolute relative errors of {fmt(external_hcpb_error)}\\% and "
        f"{fmt(external_pressure_error)}\\%, respectively."
    )

    dominant_labels = {
        "inlet_velocity": "inlet velocity",
        "inlet_temperature": "inlet temperature",
        "solid_heat_source": "pebble heat source",
    }
    discussion = (
        f"The steady matrix contains {wall_to_fluid} wall-to-fluid and "
        f"{fluid_to_wall} fluid-to-wall cases, with reversal at "
        f"{fmt(crossing_min)}--{fmt(crossing_max)} K. Factorial decomposition "
        "identifies "
        f"{dominant_labels[dominant_effects['pressure_drop_Pa']]} as the "
        "dominant main effect for pressure drop and "
        f"{dominant_labels[dominant_effects['solid_maximum_temperature_K']]} "
        "for maximum solid temperature. The largest displacement of the "
        "reference hotspot between adjacent sampled conditions is "
        f"{fmt(1000.0 * maximum_hotspot_shift)} mm. These quantities are "
        "reported separately because volume averages can hide wall-heat "
        "direction and hotspot errors.\n\n"
        "Adding physical terms does not guarantee lower temperature error. "
        "On the strict "
        "split, the three graph models give temperature RMSEs of "
        f"{transient_triplet} K and projection-aware energy RMSEs of "
        f"{energy_triplet}. {best_transient_sentence} "
        + (
            "No learned field model outperforms this zero-training baseline "
            "on the small velocity and heat-source responses in the strict "
            "split. "
            if persistence_leads
            else ""
        )
        + cost_sentence
        + f"Validation selected {loss_balancing['label']}; its independent "
        "test gave a solid-temperature RMSE of "
        f"{fmt(float(loss_balancing['test_solid_temperature_RMSE_K']))} K "
        "and energy nRMSE of "
        f"{fmt(float(loss_balancing['test_projection_aware_energy_normalized_RMSE']))}. "
        + f"Diffusion changes the deterministic temperature RMSE from "
        f"{fmt(deterministic_temperature)} to {fmt(refined_temperature)} K "
        "and the projection-aware energy RMSE from "
        f"{fmt(deterministic_energy)} to {fmt(refined_energy)}. "
        f"{diffusion_sentence} {diffusion_uncertainty_sentence} "
        "Maximum seed variation is "
        f"{fmt(float(robustness['maximum_steady_seed_cv_percent']))}\\% for "
        "steady nRMSE and "
        f"{fmt(float(robustness['maximum_transient_seed_std_K']))} K for "
        "transient temperature. For "
        f"the {STEADY_LABELS[steady_model]}, increasing steady training "
        "conditions from 9 to 36 changes test solid-temperature nRMSE from "
        f"{fmt(float(robustness['steady_learning_low_count_nrmse']))} to "
        f"{fmt(float(robustness['steady_learning_high_count_nrmse']))}; "
        "increasing transient training from three direction-specific curves "
        "to six bidirectional curves changes solid-temperature RMSE from "
        f"{fmt(float(robustness['transient_three_curve_RMSE_range_K'][0]))}"
        "--"
        f"{fmt(float(robustness['transient_three_curve_RMSE_range_K'][1]))} "
        f"to {fmt(float(robustness['transient_six_curve_RMSE_K']))} K. "
        "Held-out diffusion and data-volume results are excluded from model "
        "selection.\n\n"
        "Six high-velocity histories test the high-flow end with frozen "
        "weights; nine seed202 calculations test packing transfer. "
        "The latter shows that integral temperatures are comparatively "
        "insensitive to these two spherical-pebble arrangements, while "
        f"pressure drop changes by as much as {fmt(packing_pressure)}\\%. "
        "HELOKA Nusselt and fixed-bed pressure comparisons support the "
        "aggregate convective and hydraulic scales; PREMUX and TESOMEX "
        "remain lower-dimensional consistency checks, not cellwise "
        "validation. "
        "Transfer between these arrangements should therefore retain "
        "packing-dependent hydraulic information; bulk porosity alone did "
        "not capture the observed pressure-drop difference.\n\n"
        "Claims are limited to a local static pore-resolved domain with "
        "prescribed hydrodynamics. The full-domain mesh failed the fluid "
        "check, so no full-domain solver was run. Fully coupled startup also "
        "left the specified helium-property range. Neither failure is used "
        "to infer GCI or accuracy. Moving pebbles, evolving contacts and "
        "blanket manifolds require separate calculations."
    )

    conclusion = (
        "A finite-volume-consistent comparison of PINN, graph--Transformer "
        "and diffusion "
        "models has been completed for pore-resolved heat transfer in a "
        "solid-breeder pebble bed. The database contains 60 steady fields, "
        "12 complete thermal trajectories, six independent high-velocity "
        "histories and nine matched cases on a second spherical-pebble "
        "arrangement. The best steady and transient models are selected on "
        "complete withheld conditions rather than shuffled points, while "
        "temperature, energy, wall heat and hotspot errors remain separate. "
        f"The best strict-split transient field model is the "
        f"{TRANSIENT_LABELS[best_transient_model]} with a "
        f"{fmt(all_transient_temperatures[best_transient_model])} K "
        "solid-temperature RMSE. "
        + (
            "No learned field model improves this baseline on that split. "
            if persistence_leads
            else ""
        )
        +
        f"{diffusion_sentence} Independent-packing results show maximum "
        f"changes of {fmt(packing_tout)}\\% in outlet temperature, "
        f"{fmt(packing_tmax)}\\% in maximum solid temperature and "
        f"{fmt(packing_pressure)}\\% in pressure drop. The benchmark defines "
        "a bounded test for rapid post-adjustment thermal-field prediction "
        "within the sampled operating conditions and two packing realizations; fully coupled "
        "startup and blanket-scale prediction remain outside the evidence "
        "established here."
    )

    section_word_counts = enforce_section_lengths(
        {
            "abstract": abstract,
            "discussion": discussion,
            "conclusion": conclusion,
        }
    )
    write(args.abstract_output, abstract)
    write(args.discussion_output, discussion)
    write(args.conclusion_output, conclusion)
    return {
        "status": "complete_p418_final_manuscript_narrative",
        "steady_case_count": 60,
        "fixed_flow_trajectory_count": 12,
        "steady_solid_temperature_leader": steady_model,
        "steady_solid_temperature_worst_case_nrmse": steady_error,
        "robustness_and_learning_curve": robustness,
        "fixed_flow_loss_balancing": loss_balancing,
        "solid_maximum_temperature_range_K": [solid_min, solid_max],
        "wall_heat_direction_case_counts": {
            "wall_to_fluid": wall_to_fluid,
            "fluid_to_wall": fluid_to_wall,
            "zero": zero,
        },
        "wall_heat_reversal_temperature_range_K": [
            crossing_min,
            crossing_max,
        ],
        "factorial_dominant_main_effects": dominant_effects,
        "maximum_adjacent_reference_hotspot_distance_m": (
            maximum_hotspot_shift
        ),
        "strict_transient_temperature_RMSE_K": transient_temperatures,
        "strict_transient_projection_aware_energy_normalized_RMSE": (
            transient_energies
        ),
        "strict_transient_all_field_model_temperature_RMSE_K": (
            all_transient_temperatures
        ),
        "strict_transient_all_field_model_projection_aware_energy_"
        "normalized_RMSE": (
            all_transient_energies
        ),
        "best_strict_transient_model": best_transient_model,
        "best_strict_transient_model_is_persistence": persistence_leads,
        "best_strict_transient_speedup": best_speedup,
        "best_strict_transient_break_even_curves": best_break_even,
        "diffusion_joint_temperature_energy_improvement": joint,
        "diffusion_temperature_RMSE_K": {
            "deterministic": deterministic_temperature,
            "refined": refined_temperature,
        },
        "diffusion_projection_aware_energy_normalized_RMSE": {
            "deterministic": deterministic_energy,
            "refined": refined_energy,
        },
        "diffusion_90pct_interval_coverage_fraction": (
            diffusion_interval_coverage
        ),
        "diffusion_90pct_interval_mean_width_K": diffusion_interval_width,
        "diffusion_90pct_interval_is_underdispersed": (
            diffusion_interval_coverage < 0.85
        ),
        "high_re_curve_count": 6,
        "best_high_re_model": best_high_re_model,
        "best_high_re_fluid_temperature_RMSE_K": best_high_re_fluid,
        "best_high_re_solid_temperature_RMSE_K": best_high_re_solid,
        "independent_packing_case_count": 9,
        "independent_packing_maximum_absolute_change_percent": {
            "outlet_temperature": packing_tout,
            "maximum_solid_temperature": packing_tmax,
            "pressure_drop": packing_pressure,
        },
        "external_consistency": {
            "hcpb_annulus_mean_absolute_relative_error_percent": (
                external_hcpb_error
            ),
            "fixed_bed_pressure_median_absolute_relative_error_percent": (
                external_pressure_error
            ),
            "premux_temperature_RMSE_K": external_premux_rmse,
            "tesomex_temperature_RMSE_range_K": [
                min(external_tesomex_rmse),
                max(external_tesomex_rmse),
            ],
            "used_in_p418_training": False,
            "cellwise_validation_claimed": False,
        },
        "full_domain_solver_started": False,
        "fully_coupled_accuracy_claimed": False,
        "new_physical_parameters": [],
        "section_word_counts": section_word_counts,
        "section_word_limits": SECTION_WORD_LIMITS,
        "abstract": str(args.abstract_output),
        "discussion": str(args.discussion_output),
        "conclusions": str(args.conclusion_output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steady-summary", type=Path, required=True)
    parser.add_argument(
        "--completed-physics-summary", type=Path, required=True
    )
    parser.add_argument("--steady-hotspot-summary", type=Path, required=True)
    parser.add_argument(
        "--steady-seed-robustness", type=Path, required=True
    )
    parser.add_argument(
        "--transient-seed-robustness", type=Path, required=True
    )
    parser.add_argument("--steady-learning-curve", type=Path, required=True)
    parser.add_argument(
        "--transient-learning-curve", type=Path, required=True
    )
    parser.add_argument("--loss-balancing-selection", type=Path, required=True)
    parser.add_argument("--transient-summary", type=Path, required=True)
    parser.add_argument("--transient-metrics", type=Path, required=True)
    parser.add_argument("--transient-cost", type=Path, required=True)
    parser.add_argument("--high-re-comparison", type=Path, required=True)
    parser.add_argument("--high-re-aggregate", type=Path, required=True)
    parser.add_argument("--cross-packing-summary", type=Path, required=True)
    parser.add_argument("--external-evidence", type=Path, required=True)
    parser.add_argument("--scope-limits", type=Path, required=True)
    parser.add_argument("--abstract-output", type=Path, required=True)
    parser.add_argument("--discussion-output", type=Path, required=True)
    parser.add_argument("--conclusion-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    payload = build(args)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
