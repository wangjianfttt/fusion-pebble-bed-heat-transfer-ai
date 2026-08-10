#!/usr/bin/env python3
"""Generate cross-packing manuscript prose from the fixed-model results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


LABELS = {
    "pinn_data_only": "data-only PINN",
    "pinn": "physics-informed PINN",
    "graph": "graph operator",
    "transolver": "Physics-Attention operator",
}

METRICS = (
    ("fluid_temperature_volume_weighted_rmse_K", "fluid-temperature RMSE", "K", 1.0),
    ("solid_temperature_volume_weighted_rmse_K", "solid-temperature RMSE", "K", 1.0),
    ("solid_hotspot_location_error_m", "hotspot displacement", "mm", 1000.0),
    ("engineering_absolute_errors.pressure_drop_Pa", "pressure-drop error", "Pa", 1.0),
    ("engineering_absolute_errors.cooling_wall_heat_into_fluid_W", "wall-heat error", "W", 1.0),
    ("local_mass_l1_over_two_inlet", "regional mass difference", "percent", 100.0),
    ("local_energy_l1_over_two_generated_power", "regional energy difference", "percent", 100.0),
)


def load(path: Path, expected_status: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != expected_status:
        raise ValueError(f"unexpected status in {path}: {payload.get('status')}")
    return payload


def runs_by_seed(summary: dict, seeds: set[int]) -> dict[tuple[int, str], dict]:
    runs = list(summary.get("runs", []))
    output = {(int(run["packing_seed"]), str(run["architecture"])): run for run in runs}
    if {seed for seed, _ in output} != seeds or len(output) != len(runs):
        raise ValueError("cross-packing result contains unexpected or duplicated runs")
    return output


def p95(run: dict, metric: str) -> float:
    value = float(run["metrics"][metric]["p95"])
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"invalid p95 value for {metric}")
    return value


def fmt(value: float) -> str:
    return f"{value:.3g}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-summary", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--final-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    development = load(
        args.development_summary.resolve(), "cross_packing_model_summary_complete"
    )
    selection = load(
        args.selection.resolve(), "seed202_architecture_fixed_before_seed303"
    )
    final = load(args.final_summary.resolve(), "cross_packing_model_summary_complete")
    if selection.get("seed303_fields_read") is not False:
        raise ValueError("architecture selection does not prove seed303 was unseen")
    if selection.get("composite_score_used") is not False:
        raise ValueError("architecture selection used a composite score")

    development_runs = runs_by_seed(development, {202})
    final_runs = runs_by_seed(final, {202, 303})
    selected = str(selection["selected_architecture"])
    if selected not in LABELS:
        raise ValueError(f"unknown selected architecture: {selected}")
    if set(architecture for _, architecture in final_runs) != {selected}:
        raise ValueError("final comparison contains an architecture other than the frozen model")
    selected_202 = development_runs[(202, selected)]
    final_202 = final_runs[(202, selected)]
    final_303 = final_runs[(303, selected)]
    if selected_202.get("source_sha256") != final_202.get("source_sha256"):
        raise ValueError("final comparison does not reuse the original seed202 result")

    pareto = [str(value) for value in selection.get("pareto_architectures", [])]
    if selected not in pareto:
        raise ValueError("selected architecture is absent from the non-dominated set")

    seed202_values = {}
    transfer = {}
    for metric, quantity, unit, scale in METRICS:
        seed202_values[metric] = p95(final_202, metric) * scale
        denominator = p95(final_202, metric)
        numerator = p95(final_303, metric)
        if denominator <= 0.0:
            raise ValueError(f"seed303/seed202 ratio is undefined for {metric}")
        transfer[metric] = numerator / denominator
    largest_metric = max(transfer, key=transfer.get)
    largest_quantity = next(quantity for metric, quantity, _, _ in METRICS if metric == largest_metric)

    metric_sentences = []
    for metric, quantity, unit, _ in METRICS:
        value = seed202_values[metric]
        rendered_unit = "\\%" if unit == "percent" else unit
        metric_sentences.append(f"{quantity} {fmt(value)}~{rendered_unit}")
    ratio_sentences = [
        f"{quantity} {fmt(transfer[metric])}"
        for metric, quantity, _, _ in METRICS
    ]
    pareto_text = ", ".join(LABELS[name] for name in pareto)
    lines = [
        (
            f"On seed202, the non-dominated set contains {pareto_text}. Following the declared "
            f"solid-temperature criterion within that set selects the {LABELS[selected]}; no composite "
            "temperature--pressure--heat-transfer score is used."
        ),
        "",
        (
            f"For this fixed model, the seed202 p95 values are "
            + ", ".join(metric_sentences)
            + ". These values keep field temperature, hotspot location, pressure, wall heat, mass and "
            "energy behaviour separate."
        ),
        "",
        (
            "After the architecture, weights and seed101 normalization are frozen, the corresponding "
            "seed303-to-seed202 p95 error ratios are "
            + ", ".join(ratio_sentences)
            + f". The largest ratio is {fmt(transfer[largest_metric])} for {largest_quantity}; a ratio "
            "above unity denotes degradation on the previously unseen packing. These local results "
            "measure sensitivity to packing realization and are not a blanket-module prediction."
        ),
        "",
    ]

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "status": "complete_p418_cross_packing_manuscript_text",
        "selected_architecture": selected,
        "pareto_architectures": pareto,
        "seed202_p95_values": seed202_values,
        "seed303_to_seed202_p95_ratios": transfer,
        "largest_transfer_ratio_metric": largest_metric,
        "largest_transfer_ratio": transfer[largest_metric],
        "tex": str(output),
        "new_physical_parameters": [],
    }
    summary_path = args.summary.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
