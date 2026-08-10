from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_transient_performance_table.py"
MODELS = (
    ("dmdc", ""),
    ("graph_transformer_data_only", ""),
    ("graph_transformer_energy_flux", ""),
    ("graph_transformer_factorized_energy_flux", ""),
    ("low_rank_residual_correction", ""),
    ("diffusion_residual_correction", "diffusion_refined_"),
)


def write_metric(
    writer: csv.DictWriter,
    *,
    model: str,
    scope: str,
    metric: str,
    value: float,
    unit: str,
) -> None:
    writer.writerow(
        {
            "split_name": "pair_disjoint_stress_test",
            "model": model,
            "result_scope": scope,
            "data_role": "test",
            "metric": metric,
            "value": value,
            "unit": unit,
            "training_seconds": 12.0,
            "source_summary": "/tmp/source.json",
        }
    )


def make_metrics(path: Path) -> None:
    fields = (
        "split_name",
        "model",
        "result_scope",
        "data_role",
        "metric",
        "value",
        "unit",
        "training_seconds",
        "source_summary",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (model, prefix) in enumerate(MODELS, 1):
            write_metric(
                writer,
                model=model,
                scope="regional_temperature_field",
                metric=f"{prefix}solid_temperature_RMSE_K",
                value=index * 0.5,
                unit="K",
            )
            write_metric(
                writer,
                model=model,
                scope="regional_temperature_field",
                metric=f"{prefix}solid_maximum_temperature_history_RMSE_K",
                value=index * 0.25,
                unit="K",
            )
            write_metric(
                writer,
                model=model,
                scope="regional_temperature_field",
                metric=f"{prefix}solid_regional_hotspot_location_p95_error_m",
                value=index * 0.001,
                unit="m",
            )
            write_metric(
                writer,
                model=model,
                scope="regional_temperature_field",
                metric=f"{prefix}solid_hotspot_target_temperature_deficit_p95_K",
                value=index * 0.2,
                unit="K",
            )
            write_metric(
                writer,
                model=model,
                scope="transient_energy_balance",
                metric=(
                    "projection_aware_volume_weighted_energy_equation_normalized_RMSE"
                ),
                value=index * 0.01,
                unit="dimensionless",
            )


def test_builds_common_transient_performance_table(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.csv"
    make_metrics(metrics)
    output = tmp_path / "performance.tex"
    summary = tmp_path / "summary.json"
    subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--metrics-csv",
            str(metrics),
            "--output",
            str(output),
            "--summary",
            str(summary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    text = output.read_text(encoding="utf-8")
    assert "Physics graph--Transformer + diffusion" in text
    assert "Hotspot p95 (mm)" in text
    assert "Hotspot deficit p95 (K)" in text
    assert text.count("\\\\") >= len(MODELS) + 1
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["status"] == "complete_p418_transient_performance_table"
    assert payload["new_physical_parameters"] == []
    assert len(payload["records"]) == len(MODELS)
    assert payload["records"][0]["regional_hotspot_p95_distance_mm"] == 1.0
    assert abs(
        payload["records"][-1]["regional_hotspot_target_temperature_deficit_p95_K"]
        - 1.2
    ) < 1.0e-12


def test_rejects_missing_common_metric(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.csv"
    make_metrics(metrics)
    rows = list(csv.DictReader(metrics.open(encoding="utf-8")))
    rows = [
        row
        for row in rows
        if not (
            row["model"] == "dmdc"
            and row["metric"] == "solid_maximum_temperature_history_RMSE_K"
        )
    ]
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    completed = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--metrics-csv",
            str(metrics),
            "--output",
            str(tmp_path / "performance.tex"),
            "--summary",
            str(tmp_path / "summary.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "expected one" in completed.stderr
