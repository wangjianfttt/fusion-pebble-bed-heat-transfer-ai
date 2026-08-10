#!/usr/bin/env python3
"""Build the manuscript table for complete-chain transient prediction cost."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


STRICT_SPLIT = "pair_disjoint_stress_test"
METHODS = (
    ("dmdc", "DMDc"),
    ("graph_transformer_data_only", "Data-only GT"),
    ("graph_transformer_energy_flux", "Physics GT"),
    ("graph_transformer_factorized_energy_flux", "Factorized physics GT"),
    ("low_rank_residual_correction", "Physics GT + POD residual"),
    ("diffusion_residual_correction", "Physics GT + diffusion"),
)


def finite_positive(row: dict[str, str], name: str) -> float:
    raw = row.get(name, "")
    if raw == "":
        raise ValueError(f"missing {name} for {row.get('model')}")
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"invalid {name} for {row.get('model')}: {raw}")
    return value


def integer_count(row: dict[str, str], name: str) -> int:
    raw = row.get(name, "")
    if raw == "":
        raise ValueError(f"missing {name} for {row.get('model')}")
    value = int(raw)
    if value < 0:
        raise ValueError(f"invalid {name} for {row.get('model')}: {raw}")
    return value


def format_number(value: float) -> str:
    if abs(value) >= 1000.0:
        return f"{value:.3g}"
    if abs(value) >= 10.0:
        return f"{value:.2f}"
    return f"{value:.3f}"


def build(speed_csv: Path) -> tuple[str, list[dict[str, object]]]:
    with speed_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {
        row["model"]: row
        for row in rows
        if row.get("split_name") == STRICT_SPLIT
    }
    missing = [model for model, _ in METHODS if model not in lookup]
    if missing:
        raise ValueError(f"strict transient cost table lacks models: {missing}")

    records: list[dict[str, object]] = []
    for model, label in METHODS:
        row = lookup[model]
        records.append(
            {
                "model": model,
                "label": label,
                "complete_chain_scalar_count": integer_count(
                    row, "model_size_scalar_count"
                ),
                "complete_chain_training_minutes": finite_positive(
                    row, "training_wall_time_s"
                )
                / 60.0,
                "complete_chain_inference_seconds_per_curve": finite_positive(
                    row, "model_inference_seconds_per_curve"
                ),
                "speedup_vs_32_rank_openfoam": finite_positive(
                    row, "wall_clock_speedup"
                ),
                "training_only_break_even_curve_count": integer_count(
                    row, "training_only_break_even_curve_count"
                ),
                "full_workflow_break_even_curve_count": integer_count(
                    row, "full_workflow_break_even_curve_count"
                ),
            }
        )

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Complete-chain computational cost for the strict endpoint-pair split. POD and diffusion entries include the upstream physics-constrained graph--Transformer in their parameter count, training time and prediction time. The first break-even count includes measured model training; the second additionally includes the measured 32-rank \OpenFOAM{} wall time used to generate the registered training and validation trajectories. Both estimates use the median held-out \OpenFOAM{} wall time as the cost of a future trajectory.}",
        r"\label{tab:transient_cost}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & Scalars (M) & Train (min) & Predict (s/curve) & Speed-up & Break-even curves (train/full) \\",
        r"\midrule",
    ]
    for record in records:
        lines.append(
            "{} & {} & {} & {} & {} & {}/{} \\\\".format(
                record["label"],
                format_number(float(record["complete_chain_scalar_count"]) / 1.0e6),
                format_number(float(record["complete_chain_training_minutes"])),
                format_number(
                    float(record["complete_chain_inference_seconds_per_curve"])
                ),
                format_number(float(record["speedup_vs_32_rank_openfoam"])),
                record["training_only_break_even_curve_count"],
                record["full_workflow_break_even_curve_count"],
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines), records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    text, records = build(args.speed_csv.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    payload = {
        "status": "complete_p418_transient_cost_table",
        "split_name": STRICT_SPLIT,
        "speed_csv": str(args.speed_csv.resolve()),
        "records": records,
        "tex": str(args.output.resolve()),
        "new_physical_parameters": [],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
