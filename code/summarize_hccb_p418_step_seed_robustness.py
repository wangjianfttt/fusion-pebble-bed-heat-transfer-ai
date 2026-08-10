#!/usr/bin/env python3
"""Summarize repeated neural-model training on the strict P418 step split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


MODEL_SPECS = (
    (
        "observable_transformer",
        "transformer",
        "seed",
        ("test_mean_rmse_by_target", "outlet_temperature_K"),
        "outlet_temperature_RMSE_K",
        "completed_p418_physical_step_response_transformer_formal",
    ),
    (
        "graph_transformer_data_only",
        "regional_graph_transformer_bounded_data_only",
        "seed",
        ("metrics", "test", "solid_temperature_RMSE_K"),
        "solid_temperature_RMSE_K",
        "completed_p418_spatiotemporal_regional_operator",
    ),
    (
        "graph_transformer_energy_flux",
        "regional_graph_transformer_bounded_physics",
        "seed",
        ("metrics", "test", "solid_temperature_RMSE_K"),
        "solid_temperature_RMSE_K",
        "completed_p418_spatiotemporal_regional_operator",
    ),
    (
        "low_rank_residual_correction",
        "low_rank_temperature_residual",
        "upstream_training_seed",
        ("metrics", "test", "solid_temperature_RMSE_K"),
        "solid_temperature_RMSE_K",
        "completed_p418_low_rank_temperature_residual",
    ),
    (
        "diffusion_residual_correction",
        "temporal_diffusion",
        "seed",
        ("metrics", "test", "diffusion_refined_solid_temperature_RMSE_K"),
        "diffusion_refined_solid_temperature_RMSE_K",
        "completed_p418_temporal_temperature_diffusion",
    ),
)


def nested_value(source: dict, keys: tuple[str, ...]) -> float:
    value: object = source
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"summary lacks metric path {'/'.join(keys)}")
        value = value[key]
    return float(value)


def result_path(
    result_dir: Path,
    base: str,
    split_name: str,
    seed: int,
    primary_seed: int,
) -> Path:
    suffix = "" if seed == primary_seed else f"_seed{seed}"
    return result_dir / f"{base}_{split_name}{suffix}" / "summary.json"


def exact_split(summary: dict, expected: dict[str, list[str]], model: str) -> None:
    recorded = summary.get("split_case_ids")
    if not isinstance(recorded, dict):
        raise ValueError(f"{model} does not record complete-curve split identifiers")
    for role, identifiers in expected.items():
        actual = [str(value) for value in recorded.get(role, [])]
        if len(actual) != len(set(actual)) or set(actual) != set(identifiers):
            raise ValueError(f"{model} {role} curves differ from the registered split")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split-name", default="pair_disjoint_stress_test")
    parser.add_argument("--primary-seed", type=int, default=20260717)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260717, 20260718, 20260719])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    seeds = list(dict.fromkeys(args.seeds))
    if len(seeds) < 3 or args.primary_seed not in seeds:
        raise ValueError("seed robustness requires at least three seeds including the primary seed")
    registered = json.loads(args.splits.read_text(encoding="utf-8"))["splits"]
    if args.split_name not in registered:
        raise ValueError(f"unknown split {args.split_name}")
    expected = {
        role: [str(value) for value in registered[args.split_name][role]]
        for role in ("train", "validation", "test")
    }

    rows: list[dict] = []
    aggregate_rows: list[dict] = []
    for model, base, seed_key, metric_path, metric_name, expected_status in MODEL_SPECS:
        values = []
        for seed in seeds:
            path = result_path(args.result_dir, base, args.split_name, seed, args.primary_seed)
            if not path.is_file():
                raise FileNotFoundError(f"missing seed result: {path}")
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("status") != expected_status:
                raise ValueError(f"{model} seed {seed} is unfinished or has the wrong result type")
            if summary.get("new_physical_parameters") != []:
                raise ValueError(f"{model} seed {seed} introduces unregistered physical parameters")
            if summary.get("split_name") not in (None, args.split_name):
                raise ValueError(f"{model} records the wrong split")
            exact_split(summary, expected, model)
            if int(summary.get(seed_key, -1)) != seed:
                raise ValueError(f"{model} result does not record seed {seed}")
            if model == "diffusion_residual_correction" and int(
                summary.get("upstream_training_seed", -1)
            ) != seed:
                raise ValueError("diffusion and deterministic graph model use different seeds")
            value = nested_value(summary, metric_path)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{model} seed {seed} records an invalid error value")
            values.append(value)
            rows.append(
                {
                    "split_name": args.split_name,
                    "model": model,
                    "seed": seed,
                    "metric": metric_name,
                    "value_K": value,
                    "source_summary": str(path),
                }
            )
        aggregate_rows.append(
            {
                "split_name": args.split_name,
                "model": model,
                "metric": metric_name,
                "seed_count": len(values),
                "mean_K": statistics.mean(values),
                "sample_std_K": statistics.stdev(values),
                "minimum_K": min(values),
                "maximum_K": max(values),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "seed_metrics.csv", rows)
    write_csv(args.output_dir / "seed_summary.csv", aggregate_rows)
    summary = {
        "status": "completed_p418_strict_split_seed_robustness",
        "split_name": args.split_name,
        "seeds": seeds,
        "complete_curve_split_ids": expected,
        "models": [row["model"] for row in aggregate_rows],
        "metrics": aggregate_rows,
        "new_physical_parameters": [],
        "scientific_scope": (
            "Three initialization/training seeds on the pair-disjoint physical-step split. "
            "This measures neural training variability; it does not add CFD cases or physical parameters."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
