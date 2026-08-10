from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_engineering_metrics_are_ranked_separately(tmp_path: Path) -> None:
    comparison = tmp_path / "model_comparison.csv"
    fields = [
        "architecture",
        "split",
        "test_state_normalized_rmse",
        "test_outlet_temperature_p95_K",
        "test_solid_maximum_temperature_p95_K",
        "test_cooling_wall_heat_over_generated_p95_percent",
        "test_interphase_net_heat_over_generated_p95_percent",
        "test_global_energy_imbalance_over_generated_power_mean",
        "test_global_mass_imbalance_over_inlet_mean",
    ]
    rows = [
        {
            "architecture": "response_surface",
            "split": "independent",
            "test_state_normalized_rmse": 0.2,
            "test_outlet_temperature_p95_K": 2.0,
            "test_solid_maximum_temperature_p95_K": 8.0,
            "test_cooling_wall_heat_over_generated_p95_percent": 4.0,
            "test_interphase_net_heat_over_generated_p95_percent": 3.0,
            "test_global_energy_imbalance_over_generated_power_mean": 0.04,
            "test_global_mass_imbalance_over_inlet_mean": 0.001,
        },
        {
            "architecture": "pinn",
            "split": "independent",
            "test_state_normalized_rmse": 0.1,
            "test_outlet_temperature_p95_K": 4.0,
            "test_solid_maximum_temperature_p95_K": 5.0,
            "test_cooling_wall_heat_over_generated_p95_percent": 2.0,
            "test_interphase_net_heat_over_generated_p95_percent": 1.0,
            "test_global_energy_imbalance_over_generated_power_mean": 0.01,
            "test_global_mass_imbalance_over_inlet_mean": 0.002,
        },
    ]
    with comparison.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "code/summarize_hccb_p418_engineering_metric_leaders.py"),
            "--comparison-csv",
            str(comparison),
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    summary = json.loads(
        (output / "engineering_metric_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "engineering_metrics_compared_separately"
    assert not summary["one_architecture_leads_every_metric_by_split"]["independent"]
    assert summary["metric_lead_count_by_split"]["independent"] == {
        "pinn": 5,
        "response_surface": 2,
    }
    with (output / "engineering_metric_leaders.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        leaders = {row["metric"]: row for row in csv.DictReader(handle)}
    assert leaders["test_outlet_temperature_p95_K"]["best_architecture"] == (
        "response_surface"
    )
    assert leaders["test_global_energy_imbalance_over_generated_power_mean"][
        "best_architecture"
    ] == "pinn"
