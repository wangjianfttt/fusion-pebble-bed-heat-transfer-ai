#!/usr/bin/env python3
"""Generate the manuscript table for the full steady-to-transient model chain."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_SPLITS = (
    "direction_down_test",
    "direction_up_test",
    "pair_disjoint_stress_test",
)

SPLIT_LABELS = {
    "direction_down_test": "Downward-step split",
    "direction_up_test": "Upward-step split",
    "pair_disjoint_stress_test": "Endpoint-pair split",
}


def load_summary(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed_p418_fused_chain_comparison":
        raise ValueError(f"incomplete fused-chain result: {path}")
    return payload


def finite_nonnegative(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"invalid {name}: {value}")
    return result


def metric_row(
    *,
    split_name: str,
    subset: str,
    curve_count: object,
    exact_temperature: object,
    pinn_temperature: object,
    diffusion_temperature: object,
    deterministic_energy: object,
    diffusion_energy: object,
) -> dict[str, object]:
    exact = finite_nonnegative(exact_temperature, "exact-initial temperature RMSE")
    pinn = finite_nonnegative(pinn_temperature, "PINN-initial temperature RMSE")
    refined = finite_nonnegative(diffusion_temperature, "diffusion temperature RMSE")
    energy = finite_nonnegative(deterministic_energy, "graph-Transformer energy RMSE")
    refined_energy = finite_nonnegative(diffusion_energy, "diffusion energy RMSE")
    count = int(curve_count)
    if count <= 0:
        raise ValueError(f"invalid curve count for {split_name}/{subset}: {curve_count}")
    return {
        "split_name": split_name,
        "split_label": SPLIT_LABELS.get(split_name, split_name.replace("_", " ")),
        "subset": subset,
        "curve_count": count,
        "exact_initial_graph_transformer_solid_temperature_RMSE_K": exact,
        "steady_PINN_initial_graph_transformer_solid_temperature_RMSE_K": pinn,
        "fused_diffusion_solid_temperature_RMSE_K": refined,
        "steady_PINN_initial_graph_transformer_energy_RMSE": energy,
        "fused_diffusion_energy_RMSE": refined_energy,
        "diffusion_improves_temperature_and_energy": refined < pinn and refined_energy < energy,
    }


def collect_records(result_root: Path, splits: tuple[str, ...]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for split_name in splits:
        payload = load_summary(result_root / split_name / "fused_chain_summary.json")
        if payload.get("split_name") != split_name:
            raise ValueError(
                f"split mismatch in {split_name}: summary reports {payload.get('split_name')}"
            )
        timing = payload.get("complete_chain_timing")
        model_cost = payload.get("complete_chain_model_cost")
        if not isinstance(timing, dict) or not isinstance(model_cost, dict):
            raise ValueError(f"complete-chain cost is missing for {split_name}")
        all_row = metric_row(
                split_name=split_name,
                subset="all held-out trajectories",
                curve_count=payload["test_curve_count"],
                exact_temperature=payload[
                    "exact_initial_graph_transformer_solid_temperature_RMSE_K"
                ],
                pinn_temperature=payload[
                    "steady_PINN_initial_graph_transformer_solid_temperature_RMSE_K"
                ],
                diffusion_temperature=payload[
                    "fused_diffusion_solid_temperature_RMSE_K"
                ],
                deterministic_energy=payload[
                    "steady_PINN_initial_graph_transformer_energy_RMSE"
                ],
                diffusion_energy=payload["fused_diffusion_energy_RMSE"],
            )
        all_row.update(
            {
                "registered_steady_PINN_unique_endpoint_count": int(
                    payload["registered_steady_PINN_unique_endpoint_count"]
                ),
                "complete_chain_model_parameter_count": int(
                    model_cost["complete_chain_model_parameter_count"]
                ),
                "complete_chain_training_seconds": finite_nonnegative(
                    model_cost["complete_chain_training_seconds"],
                    "complete-chain training time",
                ),
                "warm_start_complete_chain_inference_seconds_per_curve": finite_nonnegative(
                    timing["warm_start_complete_chain_inference_seconds_per_curve"],
                    "warm-start complete-chain inference time",
                ),
                "cold_start_complete_chain_inference_seconds_per_curve": finite_nonnegative(
                    timing["cold_start_complete_chain_inference_seconds_per_curve"],
                    "cold-start complete-chain inference time",
                ),
            }
        )
        records.append(all_row)
        strict = payload.get("strict_end_to_end_group")
        if strict is not None:
            if not isinstance(strict, dict):
                raise ValueError(f"strict group is not a mapping for {split_name}")
            records.append(
                metric_row(
                    split_name=split_name,
                    subset="both steady endpoints unseen",
                    curve_count=strict["curve_count"],
                    exact_temperature=strict[
                        "exact_initial_mean_solid_temperature_RMSE_K"
                    ],
                    pinn_temperature=strict[
                        "steady_PINN_initial_mean_solid_temperature_RMSE_K"
                    ],
                    diffusion_temperature=strict[
                        "fused_diffusion_solid_temperature_RMSE_K"
                    ],
                    deterministic_energy=strict[
                        "steady_PINN_initial_graph_transformer_energy_RMSE"
                    ],
                    diffusion_energy=strict["fused_diffusion_energy_RMSE"],
                )
            )
    if not any(record["subset"] == "both steady endpoints unseen" for record in records):
        raise ValueError("no strict end-to-end result is present in the fused-chain summaries")
    return records


def format_number(value: float) -> str:
    if value == 0.0:
        return "0"
    if value < 0.01 or value >= 1000.0:
        return f"{value:.2e}"
    return f"{value:.3g}"


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def render_table(records: list[dict[str, object]]) -> str:
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\caption{Independent prediction by the complete steady-PINN--graph--Transformer--diffusion chain. The exact-initial column supplies the first OpenFOAM temperature field and target face flow as a state-assisted reference. The PINN-initial and diffusion columns instead use the frozen steady PINN for source temperature, target velocity and pressure, and the target face flow used by the common energy calculation. Energy entries are projection-aware, volume-weighted normalized errors evaluated with the same regional finite-volume energy relation. The strict rows contain trajectories for which the complete trajectory and both steady endpoints are absent from their respective training sets. Diffusion is marked as improving the result only when both solid-temperature and energy errors decrease.}",
        r"\label{tab:fused_chain_performance}",
        r"\begin{tabular}{llrrrrrrc}",
        r"\toprule",
        r"Split & Evaluation subset & $N$ & \multicolumn{3}{c}{Solid-$T$ RMSE (K)} & \multicolumn{2}{c}{Energy nRMSE} & Both $\downarrow$ \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
        r" & & & Exact init. & PINN init. & Diffusion & Graph--Transformer & Diffusion & \\",
        r"\midrule",
    ]
    for record in records:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                latex_escape(str(record["split_label"])),
                latex_escape(str(record["subset"])),
                record["curve_count"],
                format_number(
                    float(record["exact_initial_graph_transformer_solid_temperature_RMSE_K"])
                ),
                format_number(
                    float(record["steady_PINN_initial_graph_transformer_solid_temperature_RMSE_K"])
                ),
                format_number(
                    float(record["fused_diffusion_solid_temperature_RMSE_K"])
                ),
                format_number(
                    float(record["steady_PINN_initial_graph_transformer_energy_RMSE"])
                ),
                format_number(float(record["fused_diffusion_energy_RMSE"])),
                "yes" if record["diffusion_improves_temperature_and_energy"] else "no",
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    cost_records = [
        record
        for record in records
        if record["subset"] == "all held-out trajectories"
    ]
    lines.extend(
        [
            r"\begin{table*}[htbp]",
            r"\centering",
            r"\small",
            r"\caption{Measured computational cost of the complete steady-PINN--graph--Transformer--diffusion prediction chain. Warm-start time assumes that the two steady endpoint fields are already available. Cold-start time additionally includes the registered steady-PINN inference time for every unique source or target endpoint required by the held-out trajectory set; shared endpoints are evaluated once and the total is divided by the number of trajectories. Training time and scalar count include all three models. These timings do not include plotting or calculation of evaluation metrics.}",
            r"\label{tab:fused_chain_cost}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Split & Curves & Unique endpoints & Scalars & Training (h) & Inference (s curve$^{-1}$) \\",
            r"\cmidrule(lr){6-6}",
            r" & & & & & Warm / cold \\",
            r"\midrule",
        ]
    )
    for record in cost_records:
        lines.append(
            "{} & {} & {} & {} & {} & {} / {} \\\\".format(
                latex_escape(str(record["split_label"])),
                record["curve_count"],
                record["registered_steady_PINN_unique_endpoint_count"],
                int(record["complete_chain_model_parameter_count"]),
                format_number(float(record["complete_chain_training_seconds"]) / 3600.0),
                format_number(
                    float(record["warm_start_complete_chain_inference_seconds_per_curve"])
                ),
                format_number(
                    float(record["cold_start_complete_chain_inference_seconds_per_curve"])
                ),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def value_range(records: list[dict[str, object]], key: str) -> tuple[float, float]:
    values = [finite_nonnegative(record[key], key) for record in records]
    if not values:
        raise ValueError(f"no values are available for {key}")
    return min(values), max(values)


def format_range(values: tuple[float, float]) -> str:
    low, high = values
    if math.isclose(low, high, rel_tol=1.0e-12, abs_tol=1.0e-12):
        return format_number(low)
    return f"{format_number(low)}--{format_number(high)}"


def render_text(records: list[dict[str, object]]) -> str:
    """Render a compact main-text result without adding two full tables."""
    all_rows = [
        record
        for record in records
        if record["subset"] == "all held-out trajectories"
    ]
    strict_rows = [
        record
        for record in records
        if record["subset"] == "both steady endpoints unseen"
    ]
    if not all_rows or not strict_rows:
        raise ValueError("fused-chain text requires all-split and strict rows")
    strict_count = sum(int(record["curve_count"]) for record in strict_rows)
    exact_range = value_range(
        all_rows,
        "exact_initial_graph_transformer_solid_temperature_RMSE_K",
    )
    pinn_range = value_range(
        all_rows,
        "steady_PINN_initial_graph_transformer_solid_temperature_RMSE_K",
    )
    diffusion_range = value_range(
        all_rows,
        "fused_diffusion_solid_temperature_RMSE_K",
    )
    strict_pinn = value_range(
        strict_rows,
        "steady_PINN_initial_graph_transformer_solid_temperature_RMSE_K",
    )
    strict_diffusion = value_range(
        strict_rows,
        "fused_diffusion_solid_temperature_RMSE_K",
    )
    strict_energy = value_range(
        strict_rows,
        "steady_PINN_initial_graph_transformer_energy_RMSE",
    )
    strict_diffusion_energy = value_range(
        strict_rows,
        "fused_diffusion_energy_RMSE",
    )
    joint_count = sum(
        bool(record["diffusion_improves_temperature_and_energy"])
        for record in strict_rows
    )
    return (
        "The deployable steady-PINN--graph--Transformer chain was evaluated "
        "separately from the state-assisted comparison. Across the tested "
        "trajectory splits, supplying exact OpenFOAM endpoint fields gave "
        f"solid-temperature RMSEs of {format_range(exact_range)}~K; replacing "
        "them with frozen steady-PINN endpoints changed the range to "
        f"{format_range(pinn_range)}~K, and diffusion refinement gave "
        f"{format_range(diffusion_range)}~K. For the {strict_count} trajectories "
        "whose complete histories and both steady endpoints were unseen, the "
        f"PINN-initial and refined RMSE ranges were {format_range(strict_pinn)} "
        f"and {format_range(strict_diffusion)}~K, while the corresponding "
        "projection-aware energy RMSEs were "
        f"{format_range(strict_energy)} and "
        f"{format_range(strict_diffusion_energy)}. Diffusion improved both "
        f"quantities in {joint_count} of {len(strict_rows)} strict result groups."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    splits = tuple(args.splits)
    records = collect_records(args.result_root.resolve(), splits)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_table(records), encoding="utf-8")
    text_output = args.text_output.resolve() if args.text_output else None
    if text_output is not None:
        text_output.parent.mkdir(parents=True, exist_ok=True)
        text_output.write_text(render_text(records) + "\n", encoding="utf-8")

    summary = {
        "status": "completed_p418_fused_chain_manuscript_table",
        "result_root": str(args.result_root.resolve()),
        "splits": list(splits),
        "records": records,
        "strict_end_to_end_curve_count": sum(
            int(record["curve_count"])
            for record in records
            if record["subset"] == "both steady endpoints unseen"
        ),
        "tex": str(output),
        "main_text": str(text_output) if text_output is not None else None,
        "new_physical_parameters": [],
    }
    summary_path = args.summary.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
