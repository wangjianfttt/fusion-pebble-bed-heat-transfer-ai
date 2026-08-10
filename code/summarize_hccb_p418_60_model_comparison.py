#!/usr/bin/env python3
"""Summarize like-for-like P418 baseline and neural-model results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from hccb_p418_comparison_contract import STEADY_METRIC_CONTRACT


ARCHITECTURES = (
    "response_surface",
    "pinn_data_only",
    "pinn",
    "graph",
    "transolver",
)
SPLITS = (
    "interleaved_all_ranges",
    "temperature_extrapolation",
    "velocity_extrapolation",
    "heat_source_interpolation",
    "heat_source_extrapolation",
)


def nested(record: dict[str, object], *keys: str) -> object | None:
    current: object = record
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def require_exact_split(
    payload: dict[str, object],
    expected: dict[str, list[str]],
    *,
    split_name: str,
    architecture: str,
) -> None:
    if payload.get("split_name") != split_name:
        raise ValueError(
            f"{architecture} reports split {payload.get('split_name')}, expected {split_name}"
        )
    recorded = payload.get("split_case_ids")
    if not isinstance(recorded, dict):
        raise ValueError(f"{architecture} does not record exact condition identifiers")
    for role, identifiers in expected.items():
        actual = [str(value) for value in recorded.get(role, [])]
        if actual != identifiers:
            raise ValueError(
                f"{architecture} {role} conditions differ from {split_name}: "
                f"actual={actual}, expected={identifiers}"
            )


def finite_metric(metrics: dict[str, object], name: str, summary: Path) -> float:
    value = metrics.get(name)
    if value is None or not np.isfinite(float(value)):
        raise ValueError(f"missing or non-finite {name} in {summary}")
    return float(value)


def mean_case_value(cases: list[dict[str, object]], name: str) -> float:
    values = [float(case[name]) for case in cases]
    if not values or not np.all(np.isfinite(values)):
        raise ValueError(f"missing or non-finite case metric {name}")
    return float(np.mean(values))


def require_paired_pinn_control(payloads: dict[str, dict[str, object]], split: str) -> None:
    if not {"pinn_data_only", "pinn"}.issubset(payloads):
        return
    data_only = payloads["pinn_data_only"]
    constrained = payloads["pinn"]
    for key in (
        "model_parameter_count",
        "settings_from_archived_source",
        "effective_batch_size",
        "training_seed",
        "optimizer_name",
        "initial_model_state_sha256",
    ):
        if data_only.get(key) != constrained.get(key):
            raise ValueError(f"paired PINNs use different {key} for {split}")
    initial_hash = str(data_only.get("initial_model_state_sha256", ""))
    if len(initial_hash) != 64:
        raise ValueError(f"paired PINNs do not record a valid initial model state for {split}")
    if data_only.get("field_architecture") != "pinn" or constrained.get("field_architecture") != "pinn":
        raise ValueError(f"paired PINNs do not use the same coordinate network for {split}")
    if data_only.get("physics_constraints_in_training") is not False:
        raise ValueError(f"data-only PINN unexpectedly uses balance constraints for {split}")
    if constrained.get("physics_constraints_in_training") is not True:
        raise ValueError(f"physics-constrained PINN omits balance constraints for {split}")
    if data_only.get("loss_group_weights") != constrained.get("loss_group_weights"):
        raise ValueError(f"paired PINNs use different loss_group_weights for {split}")
    if data_only.get("active_loss_groups") != ["state_data", "face_flux"]:
        raise ValueError(f"data-only PINN uses unexpected loss groups for {split}")
    if constrained.get("active_loss_groups") != [
        "state_data",
        "face_flux",
        "physics_balance",
    ]:
        raise ValueError(f"physics-constrained PINN uses unexpected loss groups for {split}")
    data_terms = set(data_only.get("loss_terms", []))
    constrained_terms = set(constrained.get("loss_terms", []))
    if constrained_terms - data_terms != {"continuity", "energy_balance"}:
        raise ValueError(
            f"paired PINNs differ by more than mass/energy balances for {split}: "
            f"data={sorted(data_terms)}, constrained={sorted(constrained_terms)}"
        )
    data_weights = data_only.get("effective_loss_term_weights")
    constrained_weights = constrained.get("effective_loss_term_weights")
    if not isinstance(data_weights, dict) or not isinstance(constrained_weights, dict):
        raise ValueError(f"paired PINNs do not record per-term loss weights for {split}")
    shared_weights = {
        name: constrained_weights.get(name) for name in sorted(data_terms)
    }
    if data_weights != shared_weights:
        raise ValueError(
            f"paired PINNs change shared supervised loss weights for {split}: "
            f"data={data_weights}, constrained={shared_weights}"
        )
    if set(constrained_weights) - set(data_weights) != {"continuity", "energy_balance"}:
        raise ValueError(f"paired PINN weight maps differ by unexpected terms for {split}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--result-prefix", default="hccb_p418_60")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    split_payload = json.loads(args.split_file.read_text(encoding="utf-8"))["splits"]
    checked_fingerprints: dict[str, str] = {}
    checked_normalization: dict[str, dict[str, object]] = {}
    for split in args.splits:
        payloads_by_architecture: dict[str, dict[str, object]] = {}
        expected_ids = {
            role: [str(value) for value in split_payload[split][role]]
            for role in ("train", "validation", "test")
        }
        for architecture in args.architectures:
            summary = root / (
                f"{args.result_prefix}_{architecture}_{split}_{args.epochs}epoch/summary.json"
            )
            if not summary.is_file():
                missing.append(str(summary))
                continue
            payload = json.loads(summary.read_text(encoding="utf-8"))
            payloads_by_architecture[architecture] = payload
            if payload.get("architecture") != architecture:
                raise ValueError(
                    f"{summary} reports architecture {payload.get('architecture')}, expected {architecture}"
                )
            requested_epochs = (
                payload.get("comparison_requested_epochs")
                if architecture == "response_surface"
                else payload.get("epochs")
            )
            if int(requested_epochs if requested_epochs is not None else -1) != args.epochs:
                raise ValueError(
                    f"{architecture} {split} belongs to {requested_epochs} epochs, expected {args.epochs}"
                )
            require_exact_split(
                payload,
                expected_ids,
                split_name=split,
                architecture=architecture,
            )
            if payload.get("metric_contract") != STEADY_METRIC_CONTRACT:
                raise ValueError(f"{architecture} {split} uses a different field-error definition")
            provenance = payload.get("run_provenance")
            if not isinstance(provenance, dict):
                raise ValueError(f"{architecture} {split} has no run provenance")
            fingerprint = str(provenance.get("common_comparison_fingerprint", ""))
            if not fingerprint:
                raise ValueError(f"{architecture} {split} has no common-input fingerprint")
            if split in checked_fingerprints and checked_fingerprints[split] != fingerprint:
                raise ValueError(
                    f"{architecture} {split} used different fields, split, scaling file or metric definition"
                )
            checked_fingerprints.setdefault(split, fingerprint)
            normalization = payload.get("normalization")
            if not isinstance(normalization, dict):
                raise ValueError(f"{architecture} {split} has no normalization record")
            if split in checked_normalization and checked_normalization[split] != normalization:
                raise ValueError(f"{architecture} {split} uses different flow or balance scales")
            checked_normalization.setdefault(split, normalization)
            test = nested(payload, "evaluations", "test")
            if not isinstance(test, dict):
                raise ValueError(f"missing test evaluation in {summary}")
            metrics = test.get("metrics", {})
            cases = test.get("cases", [])
            if not isinstance(metrics, dict) or not isinstance(cases, list) or not cases:
                raise ValueError(f"incomplete test metrics in {summary}")
            case_ids = [str(case.get("condition_id")) for case in cases]
            if case_ids != expected_ids["test"]:
                raise ValueError(
                    f"{architecture} {split} evaluated different test conditions: {case_ids}"
                )
            channel_rmse = metrics.get("state_channel_rmse")
            if not isinstance(channel_rmse, list) or len(channel_rmse) != 6:
                raise ValueError(f"{architecture} {split} lacks the six common state channels")
            if not np.all(np.isfinite(np.asarray(channel_rmse, dtype=float))):
                raise ValueError(f"{architecture} {split} has non-finite state-channel errors")

            def engineering_values(name: str) -> np.ndarray:
                values = np.asarray([
                    float(case["engineering_absolute_errors"][name])
                    for case in cases
                ], dtype=float)
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"non-finite engineering error {name} in {summary}")
                return values

            def engineering_statistics(name: str) -> dict[str, float]:
                values = engineering_values(name)
                return {
                    "mae": float(np.mean(values)),
                    "p95": float(np.quantile(values, 0.95)),
                    "maximum": float(np.max(values)),
                }

            engineering = {
                name: engineering_statistics(name)
                for name in STEADY_METRIC_CONTRACT["engineering_errors"]
            }

            def heat_error_over_generated_statistics(name: str) -> dict[str, float]:
                values = np.asarray(
                    [
                        100.0
                        * float(case["engineering_absolute_errors"][name])
                        / float(case["generated_power_W"])
                        for case in cases
                    ],
                    dtype=float,
                )
                if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                    raise ValueError(
                        f"invalid heat-transfer error normalized by generated power: {name}"
                    )
                return {
                    "mean": float(np.mean(values)),
                    "p95": float(np.quantile(values, 0.95)),
                    "maximum": float(np.max(values)),
                }

            wall_heat_percent = heat_error_over_generated_statistics(
                "cooling_wall_heat_into_fluid_W"
            )
            interphase_heat_percent = heat_error_over_generated_statistics(
                "solid_to_fluid_interphase_net_W"
            )

            rows.append(
                {
                    "architecture": architecture,
                    "split": split,
                    "epochs": args.epochs,
                    "training_epochs": payload.get("epochs"),
                    "best_epoch": payload.get("best_epoch"),
                    "train_case_count": len(expected_ids["train"]),
                    "validation_case_count": len(expected_ids["validation"]),
                    "test_state_normalized_rmse": finite_metric(metrics, "state_normalized_rmse", summary),
                    "test_fluid_temperature_normalized_rmse": float(channel_rmse[4]),
                    "test_solid_temperature_normalized_rmse": float(channel_rmse[5]),
                    "test_continuity_normalized_rmse": finite_metric(metrics, "continuity_normalized_rmse", summary),
                    "test_energy_balance_normalized_rmse": finite_metric(metrics, "energy_balance_normalized_rmse", summary),
                    "test_case_count": len(cases),
                    "test_pressure_drop_mae_Pa": engineering["pressure_drop_Pa"]["mae"],
                    "test_pressure_drop_p95_Pa": engineering["pressure_drop_Pa"]["p95"],
                    "test_pressure_drop_max_Pa": engineering["pressure_drop_Pa"]["maximum"],
                    "test_outlet_temperature_mae_K": engineering["outlet_temperature_K"]["mae"],
                    "test_outlet_temperature_p95_K": engineering["outlet_temperature_K"]["p95"],
                    "test_outlet_temperature_max_K": engineering["outlet_temperature_K"]["maximum"],
                    "test_solid_maximum_temperature_mae_K": engineering["solid_maximum_temperature_K"]["mae"],
                    "test_solid_maximum_temperature_p95_K": engineering["solid_maximum_temperature_K"]["p95"],
                    "test_solid_maximum_temperature_max_K": engineering["solid_maximum_temperature_K"]["maximum"],
                    "test_cooling_wall_heat_mae_W": engineering["cooling_wall_heat_into_fluid_W"]["mae"],
                    "test_cooling_wall_heat_p95_W": engineering["cooling_wall_heat_into_fluid_W"]["p95"],
                    "test_cooling_wall_heat_max_W": engineering["cooling_wall_heat_into_fluid_W"]["maximum"],
                    "test_cooling_wall_heat_over_generated_mean_percent": wall_heat_percent["mean"],
                    "test_cooling_wall_heat_over_generated_p95_percent": wall_heat_percent["p95"],
                    "test_cooling_wall_heat_over_generated_max_percent": wall_heat_percent["maximum"],
                    "test_interphase_net_heat_mae_W": engineering["solid_to_fluid_interphase_net_W"]["mae"],
                    "test_interphase_net_heat_p95_W": engineering["solid_to_fluid_interphase_net_W"]["p95"],
                    "test_interphase_net_heat_max_W": engineering["solid_to_fluid_interphase_net_W"]["maximum"],
                    "test_interphase_net_heat_over_generated_mean_percent": interphase_heat_percent["mean"],
                    "test_interphase_net_heat_over_generated_p95_percent": interphase_heat_percent["p95"],
                    "test_interphase_net_heat_over_generated_max_percent": interphase_heat_percent["maximum"],
                    "test_interphase_absolute_heat_mae_W": engineering["fluid_solid_interphase_absolute_flow_W"]["mae"],
                    "test_interphase_absolute_heat_p95_W": engineering["fluid_solid_interphase_absolute_flow_W"]["p95"],
                    "test_interphase_absolute_heat_max_W": engineering["fluid_solid_interphase_absolute_flow_W"]["maximum"],
                    "test_local_mass_l1_over_two_inlet_mean": mean_case_value(cases, "local_mass_l1_over_two_inlet"),
                    "test_global_mass_imbalance_over_inlet_mean": mean_case_value(cases, "global_mass_imbalance_over_inlet"),
                    "test_local_energy_l1_over_two_generated_power_mean": mean_case_value(cases, "local_energy_l1_over_two_generated_power"),
                    "test_global_energy_imbalance_over_generated_power_mean": mean_case_value(cases, "global_energy_imbalance_over_generated_power"),
                    "training_wall_time_s": payload.get("training_seconds"),
                    "optimization_wall_time_s": payload.get("optimization_seconds"),
                    "optimization_s_per_update": payload.get("optimization_seconds_per_update"),
                    "validation_wall_time_s": payload.get("validation_seconds"),
                    "final_evaluation_wall_time_s": payload.get("final_evaluation_seconds"),
                    "model_parameter_count": payload.get("model_parameter_count"),
                    "effective_batch_size": payload.get("effective_batch_size"),
                    "microbatch_size": payload.get("microbatch_size"),
                    "gradient_accumulation": payload.get("gradient_accumulation"),
                    "peak_gpu_memory_GB": payload.get("peak_gpu_memory_GB"),
                    "total_parameter_updates": payload.get("total_parameter_updates"),
                    "training_seed": payload.get("training_seed"),
                    "device": payload.get("device"),
                    "torch_threads": payload.get("torch_threads"),
                    "test_inference_s_per_case": test.get("inference_seconds_per_case"),
                    "summary_file": str(summary),
                }
            )
        require_paired_pinn_control(payloads_by_architecture, split)
    csv_path = output / "model_comparison.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary_payload = {
        "status": "complete" if len(rows) == len(args.splits) * len(args.architectures) else "partial",
        "completed_model_count": len(rows),
        "expected_model_count": len(args.splits) * len(args.architectures),
        "architectures": args.architectures,
        "splits": args.splits,
        "result_prefix": args.result_prefix,
        "missing": missing,
        "comparison_csv": str(csv_path),
        "comparison_checks": {
            "exact_train_validation_test_conditions": True,
            "same_target_files_and_training_statistics": True,
            "same_flow_and_energy_scales": True,
            "same_volume_weighted_six_channel_state_metric": True,
            "same_engineering_metric_definitions": True,
            "same_integrated_wall_and_interphase_heat_definitions": True,
            "test_condition_order_checked": True,
            "paired_pinn_same_network_initialization_and_optimizer": (
                {"pinn_data_only", "pinn"}.issubset(args.architectures)
            ),
            "paired_pinn_differs_only_by_mass_and_energy_balance_losses": (
                {"pinn_data_only", "pinn"}.issubset(args.architectures)
            ),
            "paired_pinn_shared_supervised_loss_weights_are_identical": (
                {"pinn_data_only", "pinn"}.issubset(args.architectures)
            ),
        },
        "common_comparison_fingerprint_by_split": checked_fingerprints,
        "interpretation": (
            "All architectures use the same P418 fields, condition splits, train-only scaling, "
            "and reported metrics. The paired PINNs use the same coordinate network and supervised "
            "outputs with identical supervised-term coefficients; only the constrained branch adds "
            "local mass and energy balance losses. "
            "Wall and fluid--solid heat-transfer errors use the same oriented regional energy flows "
            "and are also reported relative to each case's generated power."
        ),
        "new_physical_parameters": [],
    }
    (output / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
