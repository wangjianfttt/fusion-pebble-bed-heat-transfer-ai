#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/summarize_hccb_p418_high_re_model_comparison.py"
RUNNER = ROOT / "code/run_hccb_p418_high_re_independent_evaluation.sh"
SEQUENCE_IDS = [f"high_re_{index}" for index in range(6)]
METRICS = {
    "fluid_temperature_volume_weighted_RMSE_K": 2.0,
    "solid_temperature_volume_weighted_RMSE_K": 3.0,
    "solid_maximum_temperature_history_RMSE_K": 4.0,
    "solid_regional_hotspot_location_mean_error_m": 0.002,
    "solid_regional_hotspot_location_p95_error_m": 0.004,
}
ENERGY_METRICS = {
    "projection_aware_volume_weighted_energy_equation_normalized_RMSE": 0.05,
    "prediction_global_energy_closure_normalized_RMSE": 0.02,
}


def evaluation(scale: float, sequence_ids: list[str]) -> dict[str, object]:
    aggregate = {name: value * scale for name, value in METRICS.items()}
    rows = [
        {
            "sequence_id": sequence_id,
            "fluid_temperature_volume_weighted_RMSE_K": (2.0 + index) * scale,
            "solid_temperature_volume_weighted_RMSE_K": (3.0 + index) * scale,
        }
        for index, sequence_id in enumerate(sequence_ids)
    ]
    return {
        "status": "completed_p418_frozen_high_re_independent_evaluation",
        "mode": "fixed",
        "training_or_model_selection_performed": False,
        "independent_reference_curve_count": 6,
        "independent_sequence_ids": sequence_ids,
        "aggregate_metrics": aggregate,
        "per_curve_metrics": rows,
    }


def energy(scale: float) -> dict[str, object]:
    return {
        "status": "completed_p418_common_transient_energy_balance",
        "evaluated_roles": ["test"],
        "role_metrics": {
            "test": {
                name: value * scale for name, value in ENERGY_METRICS.items()
            }
        },
    }


def write_inputs(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for label, scale in (
        ("data_only", 1.3),
        ("physics", 0.9),
        ("factorized", 1.0),
    ):
        summary = tmp_path / f"{label}_summary.json"
        energy_path = tmp_path / f"{label}_energy.json"
        summary.write_text(
            json.dumps(evaluation(scale, SEQUENCE_IDS)), encoding="utf-8"
        )
        energy_path.write_text(json.dumps(energy(scale)), encoding="utf-8")
        paths[f"{label}_summary"] = summary
        paths[f"{label}_energy"] = energy_path
    return paths


def test_builds_three_fixed_model_tables_on_same_six_curves(
    tmp_path: Path,
) -> None:
    paths = write_inputs(tmp_path)
    output = tmp_path / "comparison"
    manuscript_table = tmp_path / "manuscript/generated_high_re_comparison.tex"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data-only-summary",
            str(paths["data_only_summary"]),
            "--physics-summary",
            str(paths["physics_summary"]),
            "--factorized-summary",
            str(paths["factorized_summary"]),
            "--data-only-energy",
            str(paths["data_only_energy"]),
            "--physics-energy",
            str(paths["physics_energy"]),
            "--factorized-energy",
            str(paths["factorized_energy"]),
            "--output-dir",
            str(output),
            "--latex-output",
            str(manuscript_table),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed_p418_high_re_three_fixed_model_comparison"
    assert result["curve_count"] == 6
    assert result["same_ordered_independent_curves"] is True
    assert result["fully_coupled_model_used_for_accuracy_ranking"] is False
    assert result["model_order"] == [
        "data_only",
        "physics_constrained",
        "factorized",
    ]
    with (output / "aggregate_fixed_model_comparison.csv").open(
        encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(METRICS) + len(ENERGY_METRICS)
    assert set(rows[0]) == {
        "metric",
        "quantity",
        "data_only",
        "physics_constrained",
        "factorized",
    }
    report = (output / result["chinese_report_file"]).read_text(encoding="utf-8")
    assert "同一组6条高流速OpenFOAM独立曲线" in report
    assert "不压缩成一个人为总分" in report
    assert "不参加精度排名" in report
    tex = (output / result["latex_table_file"]).read_text(encoding="utf-8")
    assert r"\begin{table*}" in tex
    assert "Data only & Physics constrained & Factorized" in tex
    assert manuscript_table.read_text(encoding="utf-8") == tex


def test_rejects_different_curve_order(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    paths["factorized_summary"].write_text(
        json.dumps(evaluation(1.0, list(reversed(SEQUENCE_IDS)))),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data-only-summary",
            str(paths["data_only_summary"]),
            "--physics-summary",
            str(paths["physics_summary"]),
            "--factorized-summary",
            str(paths["factorized_summary"]),
            "--data-only-energy",
            str(paths["data_only_energy"]),
            "--physics-energy",
            str(paths["physics_energy"]),
            "--factorized-energy",
            str(paths["factorized_energy"]),
            "--output-dir",
            str(tmp_path / "comparison"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "same ordered curves" in completed.stderr


def test_reports_unavailable_energy_without_extrapolating_temperature(
    tmp_path: Path,
) -> None:
    paths = write_inputs(tmp_path)
    paths["data_only_energy"].write_text(
        json.dumps(
            {
                "status": (
                    "p418_energy_evaluation_unavailable_outside_registered_temperature_range"
                ),
                "evaluated_roles": ["test"],
                "registered_fluid_temperature_range_K": [300.0, 1000.0],
                "predicted_fluid_temperature_minimum_K": 280.0,
                "predicted_fluid_temperature_maximum_K": 900.0,
                "predicted_fluid_temperature_outside_registered_range_fraction": 0.01,
                "registered_solid_temperature_range_K": [298.0, 1300.0],
                "predicted_solid_temperature_minimum_K": 320.0,
                "predicted_solid_temperature_maximum_K": 900.0,
                "predicted_solid_temperature_outside_registered_range_fraction": 0.0,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "comparison"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data-only-summary",
            str(paths["data_only_summary"]),
            "--physics-summary",
            str(paths["physics_summary"]),
            "--factorized-summary",
            str(paths["factorized_summary"]),
            "--data-only-energy",
            str(paths["data_only_energy"]),
            "--physics-energy",
            str(paths["physics_energy"]),
            "--factorized-energy",
            str(paths["factorized_energy"]),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    with (output / "aggregate_fixed_model_comparison.csv").open(
        encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    energy_rows = [
        row
        for row in rows
        if row["metric"] in ENERGY_METRICS
    ]
    assert energy_rows
    assert all(row["data_only"] == "" for row in energy_rows)
    report = (output / "P418_高速端三种固定流场模型比较_CN.md").read_text(
        encoding="utf-8"
    )
    assert "| -- |" in report


def test_runner_plan_excludes_fully_coupled_accuracy_claims() -> None:
    completed = subprocess.run(
        ["bash", str(RUNNER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "models=data_only physics_constrained factorized" in completed.stdout
    assert "全耦合启动短算不参加模型精度排名" in completed.stdout
    source = RUNNER.read_text(encoding="utf-8")
    assert (
        "regional_sequences/merged/dataset_index.json"
        in source
    )
    assert "MODE=fully_coupled" not in source
    assert "fully_coupled_model_evaluation" not in source
