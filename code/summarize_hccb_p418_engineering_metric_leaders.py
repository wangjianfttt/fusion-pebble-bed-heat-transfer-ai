#!/usr/bin/env python3
"""Compare steady P418 models separately for each physical quantity."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


METRICS = (
    ("test_state_normalized_rmse", "volume-weighted 3D state", "normalized RMSE"),
    ("test_outlet_temperature_p95_K", "outlet temperature", "K"),
    ("test_solid_maximum_temperature_p95_K", "maximum solid temperature", "K"),
    (
        "test_cooling_wall_heat_over_generated_p95_percent",
        "cooling-wall heat",
        "% of generated power",
    ),
    (
        "test_interphase_net_heat_over_generated_p95_percent",
        "fluid-solid net heat transfer",
        "% of generated power",
    ),
    (
        "test_global_energy_imbalance_over_generated_power_mean",
        "global energy balance",
        "fraction of generated power",
    ),
    (
        "test_global_mass_imbalance_over_inlet_mean",
        "global mass balance",
        "fraction of inlet flow",
    ),
)


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"model comparison is empty: {path}")
    required = {"architecture", "split", *(name for name, _, _ in METRICS)}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"model comparison lacks columns: {sorted(missing)}")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["split"], row["architecture"])
        if key in seen:
            raise ValueError(f"duplicate split/architecture row: {key}")
        seen.add(key)
        for name, _, _ in METRICS:
            value = float(row[name])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid {name} for {key}: {value}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.comparison_csv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    splits = list(dict.fromkeys(row["split"] for row in rows))
    leader_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    lead_counts: dict[str, dict[str, int]] = {}

    for split in splits:
        split_rows = [row for row in rows if row["split"] == split]
        counts: Counter[str] = Counter()
        for metric, physical_quantity, unit in METRICS:
            ordered = sorted(split_rows, key=lambda row: float(row[metric]))
            best = ordered[0]
            runner_up = ordered[1] if len(ordered) > 1 else None
            best_value = float(best[metric])
            runner_up_value = float(runner_up[metric]) if runner_up else None
            counts[best["architecture"]] += 1
            leader_rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "physical_quantity": physical_quantity,
                    "unit": unit,
                    "best_architecture": best["architecture"],
                    "best_value": best_value,
                    "runner_up_architecture": (
                        runner_up["architecture"] if runner_up else ""
                    ),
                    "runner_up_value": runner_up_value if runner_up else "",
                    "best_reduction_from_runner_up_percent": (
                        100.0 * (runner_up_value - best_value) / runner_up_value
                        if runner_up_value not in (None, 0.0)
                        else 0.0
                    ),
                }
            )
            for rank, row in enumerate(ordered, start=1):
                rank_rows.append(
                    {
                        "split": split,
                        "metric": metric,
                        "physical_quantity": physical_quantity,
                        "unit": unit,
                        "rank": rank,
                        "architecture": row["architecture"],
                        "value": float(row[metric]),
                    }
                )
        lead_counts[split] = dict(counts)

    leaders_path = output / "engineering_metric_leaders.csv"
    with leaders_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(leader_rows[0]))
        writer.writeheader()
        writer.writerows(leader_rows)

    ranks_path = output / "engineering_metric_ranking.csv"
    with ranks_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rank_rows[0]))
        writer.writeheader()
        writer.writerows(rank_rows)

    metric_count = len(METRICS)
    summary = {
        "status": "engineering_metrics_compared_separately",
        "comparison_csv": portable_path(args.comparison_csv),
        "splits": splits,
        "metrics": [
            {"column": name, "physical_quantity": label, "unit": unit}
            for name, label, unit in METRICS
        ],
        "metric_lead_count_by_split": lead_counts,
        "one_architecture_leads_every_metric_by_split": {
            split: any(count == metric_count for count in counts.values())
            for split, counts in lead_counts.items()
        },
        "selection_rule": (
            "No cross-unit aggregate score is formed. Outlet temperature, solid hot spot, "
            "wall heat, fluid-solid heat transfer, energy balance, mass balance and full-field "
            "state error are compared separately on the same independent conditions."
        ),
        "leaders_csv": portable_path(leaders_path),
        "ranking_csv": portable_path(ranks_path),
        "new_physical_parameters": [],
    }
    (output / "engineering_metric_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
