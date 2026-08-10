from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_summary(path: Path, *, scale: float, wall_scale: float) -> None:
    path.parent.mkdir(parents=True)
    case = {
        "condition_id": "case1",
        "generated_power_W": 100.0,
        "global_energy_imbalance_over_generated_power": 0.05 * scale,
        "global_mass_imbalance_over_inlet": 0.01 * scale,
        "engineering_absolute_errors": {
            "outlet_temperature_K": 4.0 * scale,
            "solid_maximum_temperature_K": 8.0 * scale,
            "cooling_wall_heat_into_fluid_W": 3.0 * wall_scale,
            "solid_to_fluid_interphase_net_W": 2.0 * scale,
        },
    }
    path.write_text(
        json.dumps(
            {
                "architecture": "graph",
                "split_case_ids": {
                    "train": ["case0"],
                    "validation": ["case2"],
                    "test": ["case1"],
                },
                "run_provenance": {"common_comparison_fingerprint": "same-data"},
                "evaluations": {
                    "test": {
                        "metrics": {"state_normalized_rmse": 0.2 * scale},
                        "cases": [case],
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_longer_training_can_improve_and_worsen_different_quantities(
    tmp_path: Path,
) -> None:
    write_summary(
        tmp_path / "results/base/summary.json", scale=1.0, wall_scale=1.0
    )
    write_summary(
        tmp_path / "results/followup/summary.json", scale=0.5, wall_scale=2.0
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "architecture": "graph",
                        "split": "formal",
                        "initial_epochs": 100,
                        "followup_epochs": 2000,
                        "initial_result_directory": "results/base",
                        "followup_result_directory": "results/followup",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "code/compare_hccb_p418_epoch_followup.py"),
            "--plan",
            str(plan),
            "--project-root",
            str(tmp_path),
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    summary = json.loads(
        (output / "epoch_followup_summary.json").read_text(encoding="utf-8")
    )
    run = summary["runs"][0]
    assert run["improved_metric_count"] == 6
    assert run["worsened_metric_count"] == 1
    assert not run["all_engineering_metrics_improved"]
