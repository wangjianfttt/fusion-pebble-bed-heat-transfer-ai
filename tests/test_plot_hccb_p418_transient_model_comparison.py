from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/plot_hccb_p418_transient_model_comparison.py"
SPLIT = "pair_disjoint_stress_test"
SEQUENCES = [
    "velocity_up_T900_q6p85",
    "velocity_down_T900_q6p85",
    "source_up_u0p15_T700",
    "source_down_u0p15_T700",
]
MODELS = [
    "initial_temperature_persistence",
    "dmdc",
    "graph_transformer_data_only",
    "graph_transformer_energy_flux",
    "graph_transformer_factorized_energy_flux",
    "low_rank_residual_correction",
    "diffusion_residual_correction",
]


def write_prediction(path: Path, *, data_only: bool = False) -> None:
    time = np.tile(np.linspace(0.0, 300.0, 5), (4, 1))
    node_type = np.asarray([0, 1, 1])
    target = np.zeros((4, 5, 3, 1), dtype=np.float32)
    for case in range(4):
        target[case, :, :, 0] = np.linspace(0.0, 1.0 + 0.1 * case, 5)[:, None]
    prediction = target + (0.10 if data_only else 0.04)
    np.savez_compressed(
        path,
        sequence_id=np.asarray(SEQUENCES),
        time_s=time,
        baseline_temperature_normalized=prediction,
        target_temperature_normalized=target,
        node_type=node_type,
        node_volume_m3=np.asarray([1.0, 2.0, 3.0]),
        temperature_mean_K_by_node_type=np.asarray([500.0, 600.0]),
        temperature_std_K_by_node_type=np.asarray([50.0, 80.0]),
    )


def write_persistence(path: Path, physics_path: Path) -> None:
    with np.load(physics_path, allow_pickle=False) as source:
        node_type = source["node_type"].astype(int)
        mean = source["temperature_mean_K_by_node_type"][node_type]
        scale = source["temperature_std_K_by_node_type"][node_type]
        target = (
            source["target_temperature_normalized"][..., 0]
            * scale[None, None, :]
            + mean[None, None, :]
        )
        prediction = np.repeat(target[:, :1, :], target.shape[1], axis=1)
        np.savez_compressed(
            path,
            sequence_id=source["sequence_id"],
            time_s=source["time_s"],
            temperature_prediction_K=prediction,
            temperature_target_K=target,
            node_type=source["node_type"],
            node_volume_m3=source["node_volume_m3"],
        )


def write_fixture(root: Path) -> None:
    result = root / "results"
    comparison = result / "model_comparison"
    comparison.mkdir(parents=True)
    (comparison / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed_p418_physical_step_model_comparison",
                "splits": [SPLIT],
            }
        ),
        encoding="utf-8",
    )
    splits = root / "splits.json"
    splits.write_text(
        json.dumps({"splits": {SPLIT: {"train": [], "validation": [], "test": SEQUENCES}}}),
        encoding="utf-8",
    )
    physics_name = f"regional_graph_transformer_bounded_physics_{SPLIT}"
    diffusion_name = f"temporal_diffusion_{SPLIT}"
    for directory in (
        f"regional_persistence_{SPLIT}",
        physics_name,
        f"regional_graph_transformer_bounded_data_only_{SPLIT}",
        diffusion_name,
    ):
        folder = result / directory
        folder.mkdir()
        (folder / "summary.json").write_text(
            json.dumps(
                {
                    "status": (
                        "completed_p418_temporal_temperature_diffusion"
                        if directory == diffusion_name
                        else "completed_p418_spatiotemporal_regional_operator"
                        if directory == physics_name
                        else "completed_fixture"
                    ),
                    "split_name": SPLIT,
                    "split_case_ids": {"test": SEQUENCES},
                    "metrics": {
                        role: {
                            "fluid_temperature_RMSE_K": 2.0 + index,
                            "solid_temperature_RMSE_K": 3.0 + index,
                        }
                        for index, role in enumerate(("train", "validation", "test"))
                    },
                    "deterministic_prediction_dir": str(result / physics_name),
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
    factorized = result / f"regional_graph_transformer_bounded_factorized_{SPLIT}"
    low_rank = result / f"low_rank_temperature_residual_{SPLIT}"
    factorized.mkdir()
    low_rank.mkdir()
    (factorized / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed_p418_spatiotemporal_regional_operator",
                "split_name": SPLIT,
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
    (low_rank / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed_p418_low_rank_temperature_residual",
                "split_name": SPLIT,
                "deterministic_prediction_dir": str(result / physics_name),
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
    model_dirs = {
        "graph_transformer_energy_flux": result / physics_name,
        "graph_transformer_factorized_energy_flux": factorized,
        "low_rank_residual_correction": low_rank,
        "diffusion_residual_correction": result / diffusion_name,
    }
    model_paths = {}
    for model_name, directory in model_dirs.items():
        summary_path = directory / "summary.json"
        model_paths[model_name] = {
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
    write_prediction(
        result
        / f"regional_graph_transformer_bounded_physics_{SPLIT}"
        / "test_temporal_temperature_predictions.npz"
    )
    write_persistence(
        result / f"regional_persistence_{SPLIT}/test_temperature_predictions.npz",
        result
        / f"regional_graph_transformer_bounded_physics_{SPLIT}"
        / "test_temporal_temperature_predictions.npz",
    )
    write_prediction(
        result
        / f"regional_graph_transformer_bounded_data_only_{SPLIT}"
        / "test_temporal_temperature_predictions.npz",
        data_only=True,
    )
    with np.load(
        result
        / f"regional_graph_transformer_bounded_physics_{SPLIT}"
        / "test_temporal_temperature_predictions.npz",
        allow_pickle=False,
    ) as source:
        target = source["target_temperature_normalized"].copy()
        np.savez_compressed(
            result / f"temporal_diffusion_{SPLIT}/test_refined_temperature.npz",
            sequence_id=source["sequence_id"],
            time_s=source["time_s"],
            refined_temperature_normalized=target + 0.02,
            refined_temperature_q05_normalized=target + 0.01,
            refined_temperature_q95_normalized=target + 0.03,
            target_temperature_normalized=target,
            node_type=source["node_type"],
            node_volume_m3=source["node_volume_m3"],
            temperature_mean_K_by_node_type=source["temperature_mean_K_by_node_type"],
            temperature_std_K_by_node_type=source["temperature_std_K_by_node_type"],
        )
    metric_fields = [
        "split_name",
        "model",
        "result_scope",
        "data_role",
        "metric",
        "value",
        "unit",
        "training_seconds",
        "source_summary",
    ]
    with (comparison / "physical_step_model_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields)
        writer.writeheader()
        for index, model in enumerate(MODELS):
            temperature_metric = (
                "diffusion_refined_solid_temperature_RMSE_K"
                if model == "diffusion_residual_correction"
                else "solid_temperature_RMSE_K"
            )
            for scope, metric, value in (
                ("regional_temperature_field", temperature_metric, 2.0 + index),
                (
                    "transient_energy_balance",
                    "projection_aware_volume_weighted_energy_equation_normalized_RMSE",
                    0.001 * (index + 1),
                ),
            ):
                writer.writerow(
                    {
                        "split_name": SPLIT,
                        "model": model,
                        "result_scope": scope,
                        "data_role": "test",
                        "metric": metric,
                        "value": value,
                        "unit": "K" if "temperature" in metric else "dimensionless",
                        "training_seconds": 1.0,
                        "source_summary": "fixture",
                    }
                )
    speed_fields = [
        "split_name",
        "model",
        "model_size_scalar_count",
        "model_size_definition",
        "compute_device",
        "openfoam_median_clock_time_s",
        "model_inference_seconds_per_curve",
        "wall_clock_speedup",
    ]
    with (comparison / "physical_step_model_speedup.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=speed_fields)
        writer.writeheader()
        for index, model in enumerate(MODELS):
            speedup = 10.0 ** (index + 1)
            writer.writerow(
                {
                    "split_name": SPLIT,
                    "model": model,
                    "model_size_scalar_count": 100,
                    "model_size_definition": "fixture",
                    "compute_device": "cpu",
                    "openfoam_median_clock_time_s": 1000.0,
                    "model_inference_seconds_per_curve": 1000.0 / speedup,
                    "wall_clock_speedup": speedup,
                }
            )
    return splits


def test_renders_complete_formal_transient_figure(tmp_path: Path) -> None:
    splits = write_fixture(tmp_path)
    output = tmp_path / "figures"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--result-dir",
            str(tmp_path / "results"),
            "--splits",
            str(splits),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "hccb_p418_transient_model_comparison.pdf").is_file()
    assert (output / "hccb_p418_transient_model_comparison.svg").is_file()
    assert (output / "hccb_p418_transient_model_comparison.png").is_file()
    assert (
        tmp_path
        / "manuscript/generated_transient_model_comparison_validated.tex"
    ).is_file()
    svg_text = (
        output / "hccb_p418_transient_model_comparison.svg"
    ).read_text(encoding="utf-8")
    assert "<text" in svg_text
    assert "<image" not in svg_text
    tree = ET.parse(output / "hccb_p418_transient_model_comparison.svg")
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    rectangles = tree.findall(".//svg:clipPath/svg:rect", namespace)
    assert len(rectangles) == 6
    widths = [float(rect.attrib["width"]) for rect in rectangles]
    heights = [float(rect.attrib["height"]) for rect in rectangles]
    assert max(widths) - min(widths) < 1.0e-6
    assert max(heights) - min(heights) < 1.0e-6
    summary = json.loads(
        (output / "hccb_p418_transient_model_comparison.json").read_text(encoding="utf-8")
    )
    assert summary["held_out_trajectory_count"] == 4
    assert summary["figure_size_inch"] == [5.4, 6.7]
    assert 1.15 <= summary["panel_width_to_height_ratio"] <= 1.35
    assert "regional_persistence" in summary["persistence_prediction"]
    assert summary["split_regional_field_RMSE_K"][
        "data_only_graph_transformer"
    ]["validation"]["solid_temperature_RMSE_K"] == 4.0
    casewise = summary["held_out_volume_averaged_solid_curve_RMSE_K_by_case"]
    assert set(casewise["physics_constrained_graph_transformer"]) == set(SEQUENCES)
    assert all(
        abs(value - 3.2) < 1.0e-5
        for value in casewise["physics_constrained_graph_transformer"].values()
    )
    assert all(
        abs(value - 8.0) < 1.0e-5
        for value in casewise["data_only_graph_transformer"].values()
    )
    assert summary["new_physical_parameter_values_added"] == []
    assert summary["energy_axis_metric"] == (
        "projection_aware_volume_weighted_energy_equation_normalized_RMSE"
    )
    assert summary["energy_axis_metric_unit"] == "dimensionless"
    assert summary["validation_marker"].endswith(
        "generated_transient_model_comparison_validated.tex"
    )


def test_preselection_requires_explicit_diagnostic_mode(tmp_path: Path) -> None:
    splits = write_fixture(tmp_path)
    (
        tmp_path
        / "results/fixed_flow_loss_balancing_pair_disjoint_stress_test"
        / "selected_downstream_integration.json"
    ).unlink()
    output = tmp_path / "figures"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--result-dir",
            str(tmp_path / "results"),
            "--splits",
            str(splits),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "requires validation-selected loss balancing" in result.stderr
    assert not (output / "hccb_p418_transient_model_comparison.pdf").exists()


def test_explicit_preselection_diagnostic_does_not_write_formal_files(
    tmp_path: Path,
) -> None:
    splits = write_fixture(tmp_path)
    (
        tmp_path
        / "results/fixed_flow_loss_balancing_pair_disjoint_stress_test"
        / "selected_downstream_integration.json"
    ).unlink()
    output = tmp_path / "figures"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--result-dir",
            str(tmp_path / "results"),
            "--splits",
            str(splits),
            "--output-dir",
            str(output),
            "--allow-preselection-diagnostic",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    diagnostic_stem = "hccb_p418_transient_model_comparison_preselection_diagnostic"
    summary = json.loads(
        (output / f"{diagnostic_stem}.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "diagnostic_p418_transient_model_comparison_preselection"
    assert summary["strict_split_loss_balancing_stage"] == "registered_preselection"
    assert summary["validation_marker"] is None
    assert (output / f"{diagnostic_stem}.pdf").is_file()
    assert not (output / "hccb_p418_transient_model_comparison.pdf").exists()
    assert not (output / "hccb_p418_transient_model_comparison.json").exists()
    assert not (
        tmp_path
        / "manuscript/generated_transient_model_comparison_validated.tex"
    ).exists()


def test_preselection_rejects_explicit_formal_marker(tmp_path: Path) -> None:
    splits = write_fixture(tmp_path)
    (
        tmp_path
        / "results/fixed_flow_loss_balancing_pair_disjoint_stress_test"
        / "selected_downstream_integration.json"
    ).unlink()
    marker = tmp_path / "manuscript/formal_marker.tex"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--result-dir",
            str(tmp_path / "results"),
            "--splits",
            str(splits),
            "--output-dir",
            str(tmp_path / "figures"),
            "--validation-marker",
            str(marker),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "requires validation-selected loss balancing" in result.stderr
    assert not marker.exists()


def test_subplots_do_not_have_titles_above_axes() -> None:
    assert ".set_title(" not in SCRIPT.read_text(encoding="utf-8")


def test_speed_axis_is_explicitly_inference_only() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "Inference speed-up vs 32-rank OpenFOAM" in source
    assert "training and reference-data" in source


def test_rejects_speedup_inconsistent_with_recorded_wall_times(tmp_path: Path) -> None:
    splits = write_fixture(tmp_path)
    speed_table = tmp_path / "results/model_comparison/physical_step_model_speedup.csv"
    rows = list(csv.DictReader(speed_table.open(encoding="utf-8")))
    rows[0]["wall_clock_speedup"] = "999.0"
    with speed_table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--result-dir",
            str(tmp_path / "results"),
            "--splits",
            str(splits),
            "--output-dir",
            str(tmp_path / "figures"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "inconsistent with its recorded" in result.stderr


def test_energy_axis_names_the_common_metric() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'error_axis.set_ylabel("Projection-aware energy RMSE")' in source
    assert '"energy_axis_metric": ENERGY_METRIC' in source


def test_rejects_incomplete_formal_time_axis(tmp_path: Path) -> None:
    splits = write_fixture(tmp_path)
    predictions = (
        tmp_path
        / f"results/regional_graph_transformer_bounded_physics_{SPLIT}"
        / "test_temporal_temperature_predictions.npz",
        tmp_path
        / f"results/regional_graph_transformer_bounded_data_only_{SPLIT}"
        / "test_temporal_temperature_predictions.npz",
        tmp_path
        / f"results/regional_persistence_{SPLIT}"
        / "test_temperature_predictions.npz",
        tmp_path
        / f"results/temporal_diffusion_{SPLIT}"
        / "test_refined_temperature.npz",
    )
    for prediction in predictions:
        with np.load(prediction, allow_pickle=False) as source:
            arrays = {name: source[name].copy() for name in source.files}
        arrays["time_s"][:, -1] = 299.0
        np.savez_compressed(prediction, **arrays)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--result-dir",
            str(tmp_path / "results"),
            "--splits",
            str(splits),
            "--output-dir",
            str(tmp_path / "figures"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "must reach 300 s" in result.stderr


def test_rejects_reversed_diffusion_interval(tmp_path: Path) -> None:
    splits = write_fixture(tmp_path)
    prediction = (
        tmp_path
        / f"results/temporal_diffusion_{SPLIT}/test_refined_temperature.npz"
    )
    with np.load(prediction, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    arrays["refined_temperature_q05_normalized"] = (
        arrays["refined_temperature_q95_normalized"] + 1.0
    )
    np.savez_compressed(prediction, **arrays)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--result-dir",
            str(tmp_path / "results"),
            "--splits",
            str(splits),
            "--output-dir",
            str(tmp_path / "figures"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "reversed 5--95% intervals" in result.stderr


def test_rejects_incomplete_strict_prediction(tmp_path: Path) -> None:
    splits = write_fixture(tmp_path)
    physics = (
        tmp_path
        / f"results/regional_graph_transformer_bounded_physics_{SPLIT}/summary.json"
    )
    payload = json.loads(physics.read_text(encoding="utf-8"))
    payload["split_case_ids"]["test"] = SEQUENCES[:3]
    physics.write_text(json.dumps(payload), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--result-dir",
            str(tmp_path / "results"),
            "--splits",
            str(splits),
            "--output-dir",
            str(tmp_path / "figures"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "selected-model summary changed" in result.stderr
