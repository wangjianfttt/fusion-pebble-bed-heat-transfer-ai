#!/usr/bin/env python3
"""Select the best learned regional-temperature model for the paper field figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from hccb_p418_selected_fixed_flow_chain import (
    STRICT_SPLIT as SELECTED_STRICT_SPLIT,
    selected_chain_record_path,
    selected_model_directories,
)


STRICT_SPLIT = "pair_disjoint_stress_test"
MODEL_SPECS = {
    "graph_transformer_data_only": {
        "directory": "regional_graph_transformer_bounded_data_only_{split}",
        "prediction": "test_temporal_temperature_predictions.npz",
        "metric": "solid_temperature_RMSE_K",
        "label": "data-only graph--Transformer",
        "status": "completed_p418_spatiotemporal_regional_operator",
        "summary_file_key": "temporal_temperature_prediction_files",
    },
    "graph_transformer_energy_flux": {
        "directory": "regional_graph_transformer_bounded_physics_{split}",
        "prediction": "test_temporal_temperature_predictions.npz",
        "metric": "solid_temperature_RMSE_K",
        "label": "physics-constrained graph--Transformer",
        "status": "completed_p418_spatiotemporal_regional_operator",
        "summary_file_key": "temporal_temperature_prediction_files",
    },
    "graph_transformer_factorized_energy_flux": {
        "directory": "regional_graph_transformer_bounded_factorized_{split}",
        "prediction": "test_temporal_temperature_predictions.npz",
        "metric": "solid_temperature_RMSE_K",
        "label": "factorized physics graph--Transformer",
        "status": "completed_p418_spatiotemporal_regional_operator",
        "summary_file_key": "temporal_temperature_prediction_files",
    },
    "low_rank_residual_correction": {
        "directory": "low_rank_temperature_residual_{split}",
        "prediction": "test_low_rank_temperature_predictions.npz",
        "metric": "solid_temperature_RMSE_K",
        "label": "low-rank residual correction",
        "status": "completed_p418_low_rank_temperature_residual",
        "summary_file_key": "prediction_files",
    },
    "diffusion_residual_correction": {
        "directory": "temporal_diffusion_{split}",
        "prediction": "test_refined_temperature.npz",
        "metric": "diffusion_refined_solid_temperature_RMSE_K",
        "label": "diffusion residual correction",
        "status": "completed_p418_temporal_temperature_diffusion",
        "summary_file_key": "prediction_files",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def project_relative(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"formal field-selection file is outside the project root: {path}") from error


def infer_project_root(result_dir: Path) -> Path:
    result_dir = result_dir.resolve()
    if result_dir.name == "hccb_p418_physical_steps_12" and result_dir.parent.name == "results":
        return result_dir.parents[1]
    return result_dir.parent


def metric_values(metrics_csv: Path, split_name: str) -> dict[str, float]:
    values: dict[str, float] = {}
    with metrics_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for model, spec in MODEL_SPECS.items():
        matches = [
            row
            for row in rows
            if row.get("split_name") == split_name
            and row.get("model") == model
            and row.get("result_scope") == "regional_temperature_field"
            and row.get("data_role") == "validation"
            and row.get("metric") == spec["metric"]
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one {spec['metric']} row for {model} on {split_name}, "
                f"found {len(matches)}"
            )
        value = float(matches[0]["value"])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid field RMSE for {model}: {value}")
        values[model] = value
    return values


def select_model(
    *,
    result_dir: Path,
    comparison_summary: Path,
    metrics_csv: Path,
    split_name: str,
) -> dict:
    result_dir = result_dir.resolve()
    project_root = infer_project_root(result_dir)
    comparison = load_json(comparison_summary)
    if comparison.get("status") != "completed_p418_physical_step_model_comparison":
        raise ValueError("formal transient model comparison is not complete")
    if split_name not in comparison.get("splits", []):
        raise ValueError(f"comparison does not contain {split_name}")
    values = metric_values(metrics_csv, split_name)
    selected_model = min(MODEL_SPECS, key=lambda name: (values[name], name))
    spec = MODEL_SPECS[selected_model]
    integration_path = selected_chain_record_path(result_dir)
    if split_name == STRICT_SPLIT and not integration_path.is_file():
        raise ValueError(
            "validation-selected loss-balancing chain is incomplete; field-model "
            "selection cannot fall back to registered preselection"
        )
    selected_directories = selected_model_directories(
        result_dir,
        split_name,
    )
    model_dir = selected_directories.get(
        selected_model,
        result_dir / str(spec["directory"]).format(split=split_name),
    )
    summary_path = model_dir / "summary.json"
    summary = load_json(summary_path)
    if summary.get("status") != spec["status"]:
        raise ValueError(f"selected model has unexpected status: {summary.get('status')}")
    if summary.get("split_name") != split_name:
        raise ValueError("selected model summary uses a different split")
    prediction_path = model_dir / str(spec["prediction"])
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    recorded_files = summary.get(str(spec["summary_file_key"]), {})
    if recorded_files.get("test") != prediction_path.name:
        raise ValueError("selected model summary records a different test prediction file")
    if summary.get("new_physical_parameters") != []:
        raise ValueError("selected model introduced an unregistered physical parameter")
    result = {
        "status": "selected_p418_field_figure_learned_model",
        "split_name": split_name,
        "selection_rule": (
            "lowest validation regional-volume-weighted solid-temperature RMSE among "
            "the five formally trained regional field models; the displayed field remains "
            "a held-out test trajectory"
        ),
        "selection_data_role": "validation",
        "display_data_role": "test",
        "selected_model": selected_model,
        "selected_model_label": spec["label"],
        "selected_metric": spec["metric"],
        "selected_metric_value_K": values[selected_model],
        "eligible_model_metric_K": values,
        "excluded_reference_models": [
            "initial_temperature_persistence",
            "dmdc",
        ],
        "prediction_file": project_relative(prediction_path, project_root),
        "prediction_file_sha256": sha256(prediction_path),
        "model_summary": project_relative(summary_path, project_root),
        "model_summary_sha256": sha256(summary_path),
        "comparison_summary": project_relative(comparison_summary, project_root),
        "comparison_summary_sha256": sha256(comparison_summary),
        "metrics_csv": project_relative(metrics_csv, project_root),
        "metrics_csv_sha256": sha256(metrics_csv),
        "new_physical_parameters": [],
        "strict_split_loss_balancing_stage": (
            "validation_selected" if split_name == STRICT_SPLIT else "not_applicable"
        ),
    }
    if split_name == SELECTED_STRICT_SPLIT:
        result.update(
            {
                "selected_loss_balancing_integration_record": project_relative(
                    integration_path, project_root
                ),
                "selected_loss_balancing_integration_record_sha256": sha256(
                    integration_path
                ),
            }
        )
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=root / "results/hccb_p418_physical_steps_12",
    )
    parser.add_argument("--comparison-summary", type=Path)
    parser.add_argument("--metrics-csv", type=Path)
    parser.add_argument("--split-name", default=STRICT_SPLIT)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "figures/hccb_p418_openfoam_model_field_selection.json",
    )
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    comparison_summary = (
        args.comparison_summary.resolve()
        if args.comparison_summary is not None
        else result_dir / "model_comparison/summary.json"
    )
    metrics_csv = (
        args.metrics_csv.resolve()
        if args.metrics_csv is not None
        else result_dir / "model_comparison/physical_step_model_metrics.csv"
    )
    result = select_model(
        result_dir=result_dir,
        comparison_summary=comparison_summary,
        metrics_csv=metrics_csv,
        split_name=args.split_name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
