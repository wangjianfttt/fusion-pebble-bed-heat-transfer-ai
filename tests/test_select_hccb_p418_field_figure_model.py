from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/select_hccb_p418_field_figure_model.py"
SPLIT = "pair_disjoint_stress_test"
MODELS = {
    "graph_transformer_data_only": (
        f"regional_graph_transformer_bounded_data_only_{SPLIT}",
        "test_temporal_temperature_predictions.npz",
        "solid_temperature_RMSE_K",
        "completed_p418_spatiotemporal_regional_operator",
        "temporal_temperature_prediction_files",
    ),
    "graph_transformer_energy_flux": (
        f"regional_graph_transformer_bounded_physics_{SPLIT}",
        "test_temporal_temperature_predictions.npz",
        "solid_temperature_RMSE_K",
        "completed_p418_spatiotemporal_regional_operator",
        "temporal_temperature_prediction_files",
    ),
    "graph_transformer_factorized_energy_flux": (
        f"regional_graph_transformer_bounded_factorized_{SPLIT}",
        "test_temporal_temperature_predictions.npz",
        "solid_temperature_RMSE_K",
        "completed_p418_spatiotemporal_regional_operator",
        "temporal_temperature_prediction_files",
    ),
    "low_rank_residual_correction": (
        f"low_rank_temperature_residual_{SPLIT}",
        "test_low_rank_temperature_predictions.npz",
        "solid_temperature_RMSE_K",
        "completed_p418_low_rank_temperature_residual",
        "prediction_files",
    ),
    "diffusion_residual_correction": (
        f"temporal_diffusion_{SPLIT}",
        "test_refined_temperature.npz",
        "diffusion_refined_solid_temperature_RMSE_K",
        "completed_p418_temporal_temperature_diffusion",
        "prediction_files",
    ),
}


def prepare(tmp_path: Path, winner: str) -> tuple[Path, Path, Path]:
    result = tmp_path / "results"
    comparison = result / "model_comparison"
    comparison.mkdir(parents=True)
    summary = comparison / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "completed_p418_physical_step_model_comparison",
                "splits": [SPLIT],
            }
        ),
        encoding="utf-8",
    )
    metrics = comparison / "physical_step_model_metrics.csv"
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split_name",
                "model",
                "result_scope",
                "data_role",
                "metric",
                "value",
            ],
        )
        writer.writeheader()
        for index, (model, (_, _, metric, _, _)) in enumerate(MODELS.items()):
            writer.writerow(
                {
                    "split_name": SPLIT,
                    "model": model,
                    "result_scope": "regional_temperature_field",
                    "data_role": "validation",
                    "metric": metric,
                    "value": 0.5 if model == winner else 2.0 + index,
                }
            )
    for model, (directory, prediction, _, status, file_key) in MODELS.items():
        folder = result / directory
        folder.mkdir()
        (folder / prediction).write_bytes(f"prediction:{model}".encode())
        (folder / "summary.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "split_name": SPLIT,
                    file_key: {"test": prediction},
                    "deterministic_prediction_dir": str(
                        result
                        / f"regional_graph_transformer_bounded_physics_{SPLIT}"
                    ),
                    "evaluation_stage": "final",
                    "test_evaluated": True,
                    "loss_balancing": {
                        "candidate_id": "fixed_registered_5_1_1"
                    },
                    "loss_weights": {
                        "temperature_data": 5.0,
                        "reference_edge_energy_flux": 1.0,
                        "projection_aware_transient_energy": 1.0,
                    },
                    "new_physical_parameters": [],
                }
            ),
            encoding="utf-8",
        )
    integration_root = result / "fixed_flow_loss_balancing_pair_disjoint_stress_test"
    integration_root.mkdir()
    selection_path = integration_root / "selected_loss_balancing_method.json"
    selection_path.write_text(
        json.dumps(
            {
                "status": "p418_loss_balancing_selected_on_validation_only",
                "selected_candidate_id": "fixed_registered_5_1_1",
                "independent_test_read": False,
            }
        ),
        encoding="utf-8",
    )
    selected_names = (
        "graph_transformer_energy_flux",
        "graph_transformer_factorized_energy_flux",
        "low_rank_residual_correction",
        "diffusion_residual_correction",
    )
    model_paths = {}
    for model in selected_names:
        directory = result / MODELS[model][0]
        summary_path = directory / "summary.json"
        model_paths[model] = {
            "directory_relative_to_result_root": str(directory.relative_to(result)),
            "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        }
    (integration_root / "selected_downstream_integration.json").write_text(
        json.dumps(
            {
                "status": "completed_p418_selected_loss_balancing_downstream",
                "split_name": SPLIT,
                "selected_candidate_id": "fixed_registered_5_1_1",
                "selection_record_sha256": hashlib.sha256(
                    selection_path.read_bytes()
                ).hexdigest(),
                "independent_test_read_after_validation_selection": True,
                "model_paths": model_paths,
                "new_physical_parameters": [],
            }
        ),
        encoding="utf-8",
    )
    return result, summary, metrics


@pytest.mark.parametrize(
    "winner",
    [
        "graph_transformer_data_only",
        "graph_transformer_factorized_energy_flux",
        "low_rank_residual_correction",
        "diffusion_residual_correction",
    ],
)
def test_selects_lowest_rmse_learned_model(tmp_path: Path, winner: str) -> None:
    result, summary, metrics = prepare(tmp_path, winner)
    output = tmp_path / "selection.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--result-dir",
            str(result),
            "--comparison-summary",
            str(summary),
            "--metrics-csv",
            str(metrics),
            "--output",
            str(output),
        ],
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["selected_model"] == winner
    assert payload["selected_metric_value_K"] == 0.5
    assert payload["selection_data_role"] == "validation"
    assert payload["display_data_role"] == "test"
    assert payload["strict_split_loss_balancing_stage"] == "validation_selected"
    assert Path(payload["prediction_file"]).is_file()
    assert payload["excluded_reference_models"] == [
        "initial_temperature_persistence",
        "dmdc",
    ]


def test_rejects_incomplete_comparison(tmp_path: Path) -> None:
    result, summary, metrics = prepare(tmp_path, "graph_transformer_data_only")
    summary.write_text('{"status":"incomplete"}\n', encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--result-dir",
            str(result),
            "--comparison-summary",
            str(summary),
            "--metrics-csv",
            str(metrics),
            "--output",
            str(tmp_path / "selection.json"),
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "not complete" in completed.stderr
