#!/usr/bin/env python3
"""Summarize the deployable steady PINN--graph-Transformer--diffusion chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ENERGY_METRIC = "projection_aware_volume_weighted_energy_equation_normalized_RMSE"


def build_payload(
    *,
    chained: dict[str, object],
    chained_energy: dict[str, object],
    diffusion: dict[str, object],
    diffusion_energy: dict[str, object],
) -> dict[str, object]:
    role = "test"
    diffusion_metrics = diffusion["metrics"][role]
    timing = diffusion.get("complete_chain_timing")
    model_cost = diffusion.get("complete_chain_model_cost")
    chained_timing = chained.get("timing")
    if not isinstance(timing, dict) or not isinstance(model_cost, dict):
        raise ValueError("fused chain lacks complete inference or model-cost results")
    if not isinstance(chained_timing, dict):
        raise ValueError("fused chain lacks steady-endpoint timing")
    endpoint_timing = chained_timing.get("registered_steady_PINN_endpoint_timing")
    if not isinstance(endpoint_timing, dict):
        raise ValueError("fused chain lacks registered steady-endpoint timing")
    deterministic_temperature = float(
        chained["steady_PINN_initial_mean_solid_temperature_RMSE_K"]
    )
    refined_temperature = float(
        diffusion_metrics["ensemble_mean_solid_temperature_RMSE_K"]
    )
    deterministic_energy = float(chained_energy["role_metrics"][role][ENERGY_METRIC])
    refined_energy = float(diffusion_energy["role_metrics"][role][ENERGY_METRIC])
    interval_names = (
        "unobserved_dynamic_90pct_interval_coverage_fraction",
        "unobserved_dynamic_90pct_interval_mean_width_K",
        "unobserved_dynamic_fluid_90pct_interval_coverage_fraction",
        "unobserved_dynamic_fluid_90pct_interval_mean_width_K",
        "unobserved_dynamic_solid_90pct_interval_coverage_fraction",
        "unobserved_dynamic_solid_90pct_interval_mean_width_K",
        "unobserved_dynamic_CRPS_K",
        "unobserved_dynamic_fluid_CRPS_K",
        "unobserved_dynamic_solid_CRPS_K",
    )
    missing = [name for name in interval_names if name not in diffusion_metrics]
    if missing:
        raise ValueError(
            "chained diffusion summary lacks unobserved dynamic interval results: "
            + ", ".join(missing)
        )
    interval_results = {
        f"fused_diffusion_{name}": diffusion_metrics[name]
        for name in interval_names
    }
    deterministic_groups = chained.get("endpoint_novelty_groups")
    diffusion_groups = diffusion.get("endpoint_novelty_metrics")
    if not isinstance(deterministic_groups, dict) or not isinstance(diffusion_groups, dict):
        raise ValueError("fused chain lacks steady-endpoint subgroup results")
    if set(deterministic_groups) != set(diffusion_groups):
        raise ValueError("deterministic and diffusion endpoint groups differ")
    deterministic_energy_groups = chained_energy.get("endpoint_novelty_metrics", {}).get(
        role
    )
    diffusion_energy_groups = diffusion_energy.get("endpoint_novelty_metrics", {}).get(
        role
    )
    if not isinstance(deterministic_energy_groups, dict) or not isinstance(
        diffusion_energy_groups, dict
    ):
        raise ValueError("fused chain lacks steady-endpoint energy subgroups")
    if set(deterministic_groups) != set(deterministic_energy_groups) or set(
        deterministic_groups
    ) != set(diffusion_energy_groups):
        raise ValueError("temperature and energy endpoint groups differ")
    endpoint_groups = {
        name: {
            **deterministic_groups[name],
            "fused_diffusion_solid_temperature_RMSE_K": diffusion_groups[name][
                "ensemble_mean_solid_temperature_RMSE_K"
            ],
            "diffusion_improves_solid_temperature": (
                float(diffusion_groups[name]["ensemble_mean_solid_temperature_RMSE_K"])
                < float(
                    deterministic_groups[name][
                        "steady_PINN_initial_mean_solid_temperature_RMSE_K"
                    ]
                )
            ),
            "steady_PINN_initial_graph_transformer_energy_RMSE": deterministic_energy_groups[
                name
            ][ENERGY_METRIC],
            "fused_diffusion_energy_RMSE": diffusion_energy_groups[name][
                ENERGY_METRIC
            ],
            "diffusion_improves_energy_equation": (
                float(diffusion_energy_groups[name][ENERGY_METRIC])
                < float(deterministic_energy_groups[name][ENERGY_METRIC])
            ),
            "diffusion_improves_temperature_and_energy": (
                float(diffusion_groups[name]["ensemble_mean_solid_temperature_RMSE_K"])
                < float(
                    deterministic_groups[name][
                        "steady_PINN_initial_mean_solid_temperature_RMSE_K"
                    ]
                )
                and float(diffusion_energy_groups[name][ENERGY_METRIC])
                < float(deterministic_energy_groups[name][ENERGY_METRIC])
            ),
        }
        for name in sorted(deterministic_groups)
    }
    return {
        "status": "completed_p418_fused_chain_comparison",
        "split_name": chained["transient_split_name"],
        "test_curve_count": chained["curve_count"],
        "exact_initial_graph_transformer_solid_temperature_RMSE_K": chained[
            "exact_initial_mean_solid_temperature_RMSE_K"
        ],
        "steady_PINN_initial_graph_transformer_solid_temperature_RMSE_K": deterministic_temperature,
        "steady_PINN_initial_graph_transformer_energy_RMSE": deterministic_energy,
        "fused_diffusion_solid_temperature_RMSE_K": refined_temperature,
        "fused_diffusion_energy_RMSE": refined_energy,
        **interval_results,
        "endpoint_novelty_groups": endpoint_groups,
        "strict_end_to_end_group": endpoint_groups.get(
            "both_steady_endpoints_unseen"
        ),
        "complete_chain_timing": timing,
        "complete_chain_model_cost": model_cost,
        "registered_steady_PINN_unique_endpoint_count": endpoint_timing[
            "unique_endpoint_count"
        ],
        "registered_steady_PINN_unique_endpoint_condition_ids": endpoint_timing[
            "unique_endpoint_condition_ids"
        ],
        "diffusion_improves_temperature": refined_temperature < deterministic_temperature,
        "diffusion_improves_energy_equation": refined_energy < deterministic_energy,
        "diffusion_improves_both": (
            refined_temperature < deterministic_temperature
            and refined_energy < deterministic_energy
        ),
        "interpretation": (
            "Diffusion is considered useful for the fused model only when both the held-out "
            "solid-temperature error and the projection-aware transient energy-equation "
            "difference decrease. Temperature interval coverage is reported only for "
            "unobserved points after the initial state, separately for fluid and solid."
        ),
        "new_physical_parameters": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chained-summary", type=Path, required=True)
    parser.add_argument("--chained-energy", type=Path, required=True)
    parser.add_argument("--diffusion-summary", type=Path, required=True)
    parser.add_argument("--diffusion-energy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    chained = json.loads(args.chained_summary.resolve().read_text(encoding="utf-8"))
    chained_energy = json.loads(args.chained_energy.resolve().read_text(encoding="utf-8"))
    diffusion = json.loads(args.diffusion_summary.resolve().read_text(encoding="utf-8"))
    diffusion_energy = json.loads(args.diffusion_energy.resolve().read_text(encoding="utf-8"))
    payload = build_payload(
        chained=chained,
        chained_energy=chained_energy,
        diffusion=diffusion,
        diffusion_energy=diffusion_energy,
    )
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
