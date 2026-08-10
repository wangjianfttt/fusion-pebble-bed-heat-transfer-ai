#!/usr/bin/env python3
"""Compare first-pass and source-length P418 training runs by physical quantity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from summarize_hccb_p418_engineering_metric_leaders import METRICS


def load_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    test = payload.get("evaluations", {}).get("test", {})
    if not isinstance(test, dict) or not test.get("cases"):
        raise ValueError(f"missing independent test cases: {path}")
    return payload


def summary_metrics(payload: dict[str, object]) -> dict[str, float]:
    test = payload["evaluations"]["test"]
    metrics = test["metrics"]
    cases = test["cases"]

    def engineering_p95(name: str) -> float:
        values = np.asarray(
            [float(case["engineering_absolute_errors"][name]) for case in cases],
            dtype=float,
        )
        return float(np.quantile(values, 0.95))

    def heat_p95(name: str) -> float:
        values = np.asarray(
            [
                100.0
                * float(case["engineering_absolute_errors"][name])
                / float(case["generated_power_W"])
                for case in cases
            ],
            dtype=float,
        )
        return float(np.quantile(values, 0.95))

    result = {
        "test_state_normalized_rmse": float(metrics["state_normalized_rmse"]),
        "test_outlet_temperature_p95_K": engineering_p95("outlet_temperature_K"),
        "test_solid_maximum_temperature_p95_K": engineering_p95(
            "solid_maximum_temperature_K"
        ),
        "test_cooling_wall_heat_over_generated_p95_percent": heat_p95(
            "cooling_wall_heat_into_fluid_W"
        ),
        "test_interphase_net_heat_over_generated_p95_percent": heat_p95(
            "solid_to_fluid_interphase_net_W"
        ),
        "test_global_energy_imbalance_over_generated_power_mean": float(
            np.mean(
                [
                    float(case["global_energy_imbalance_over_generated_power"])
                    for case in cases
                ]
            )
        ),
        "test_global_mass_imbalance_over_inlet_mean": float(
            np.mean(
                [float(case["global_mass_imbalance_over_inlet"]) for case in cases]
            )
        ),
    }
    if not np.all(np.isfinite(list(result.values()))) or any(
        value < 0.0 for value in result.values()
    ):
        raise ValueError("follow-up comparison contains invalid physical errors")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    root = args.project_root.resolve()
    rows: list[dict[str, object]] = []
    run_summaries: list[dict[str, object]] = []
    for run in plan.get("runs", []):
        initial_path = root / str(run["initial_result_directory"]) / "summary.json"
        followup_path = root / str(run["followup_result_directory"]) / "summary.json"
        initial = load_summary(initial_path)
        followup = load_summary(followup_path)
        if initial.get("architecture") != followup.get("architecture"):
            raise ValueError("initial and source-length runs use different architectures")
        if initial.get("split_case_ids") != followup.get("split_case_ids"):
            raise ValueError("initial and source-length runs use different condition splits")
        initial_fp = initial.get("run_provenance", {}).get(
            "common_comparison_fingerprint"
        )
        followup_fp = followup.get("run_provenance", {}).get(
            "common_comparison_fingerprint"
        )
        if not initial_fp or initial_fp != followup_fp:
            raise ValueError("initial and source-length runs use different physical data")
        initial_values = summary_metrics(initial)
        followup_values = summary_metrics(followup)
        improved = 0
        worsened = 0
        unchanged = 0
        for name, physical_quantity, unit in METRICS:
            before = initial_values[name]
            after = followup_values[name]
            if np.isclose(before, after, rtol=1.0e-9, atol=0.0):
                change = "unchanged"
                unchanged += 1
            elif after < before:
                change = "improved"
                improved += 1
            else:
                change = "worsened"
                worsened += 1
            rows.append(
                {
                    "architecture": run["architecture"],
                    "split": run["split"],
                    "initial_epochs": run["initial_epochs"],
                    "followup_epochs": run["followup_epochs"],
                    "metric": name,
                    "physical_quantity": physical_quantity,
                    "unit": unit,
                    "initial_value": before,
                    "followup_value": after,
                    "followup_change": change,
                    "followup_reduction_percent": (
                        100.0 * (before - after) / before if before > 0.0 else 0.0
                    ),
                }
            )
        run_summaries.append(
            {
                "architecture": run["architecture"],
                "split": run["split"],
                "initial_epochs": run["initial_epochs"],
                "followup_epochs": run["followup_epochs"],
                "improved_metric_count": improved,
                "worsened_metric_count": worsened,
                "unchanged_metric_count": unchanged,
                "all_engineering_metrics_improved": worsened == 0 and improved > 0,
            }
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "epoch_followup_physical_comparison.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "status": "source_epoch_followup_compared" if rows else "no_followup_required",
        "followup_run_count": len(run_summaries),
        "runs": run_summaries,
        "comparison_rule": (
            "The longer source schedule is judged separately for seven physical quantities. "
            "A lower validation total loss alone is not treated as an improvement."
        ),
        "comparison_csv": str(csv_path),
        "new_physical_parameters": [],
    }
    (output / "epoch_followup_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
