#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
torch = pytest.importorskip("torch")

from hccb_p418_comparison_contract import sha256_file  # noqa: E402
from hccb_p418_fully_coupled_dataset import (  # noqa: E402
    load_sequence as load_full_sequence,
    sequence_records as full_records,
    training_statistics as full_statistics,
)
from hccb_p418_fully_coupled_spatiotemporal_operator import (  # noqa: E402
    FULLY_COUPLED_ARCHITECTURE_REVISION,
    HCCBP418FullyCoupledRegionalOperator,
    build_p418_fully_coupled_flux_graph,
)
from hccb_p418_fully_coupled_training import training_equation_scales  # noqa: E402
from hccb_p418_fully_coupled_transient_physics import (  # noqa: E402
    assemble_p418_fully_coupled_transient_residual,
)
from hccb_p418_regional_cht_adapter import load_p418_subface_geometry  # noqa: E402
from hccb_p418_spatiotemporal_regional_operator import (  # noqa: E402
    HCCBP418SpatiotemporalRegionalOperator,
    P418ThermalStepRegionalGraph,
)
from train_hccb_p418_spatiotemporal_regional_operator import (  # noqa: E402
    sequence_records as fixed_records,
    training_statistics as fixed_statistics,
)


EVALUATOR = ROOT / "code/evaluate_hccb_p418_frozen_independent_operator.py"
FULL_HELPER_PATH = ROOT / "tests/test_hccb_p418_fully_coupled_trainer.py"
ARCHITECTURE = {
    "hidden_dim": 8,
    "local_pre_iterations": 1,
    "physics_attention_blocks": 1,
    "local_post_iterations": 1,
    "physics_attention_heads": 1,
    "physics_slices": 2,
    "temporal_layers": 1,
    "temporal_heads": 1,
    "spatial_time_chunk_size": 1,
    "temporal_node_chunk_size": 16,
}
CONDITION_NAMES = [
    "source_inlet_velocity_m_s",
    "source_inlet_temperature_K",
    "source_solid_heat_source_MW_m3",
    "target_inlet_velocity_m_s",
    "target_inlet_temperature_K",
    "target_solid_heat_source_MW_m3",
    "target_outlet_pressure_Pa",
    "target_cooling_wall_temperature_K",
]
STATE_NAMES = ["Ux_m_s", "Uy_m_s", "Uz_m_s", "pressure_Pa", "temperature_K"]


def load_full_helper():
    spec = importlib.util.spec_from_file_location("p418_full_test_helper", FULL_HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dataset_payload(
    *,
    graph: Path,
    records: list[dict[str, object]],
    fully_coupled: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "sequence_count": len(records),
        "condition_names": CONDITION_NAMES,
        "state_names": STATE_NAMES,
        "regional_geometry_file": graph.name,
        "boundary_patch_names": {
            "fluid": [
                "inlet",
                "outlet",
                "coolingWall",
                "symmetryWalls",
                "fluid_to_solid",
            ],
            "solid": [
                "inlet",
                "outlet",
                "coolingWall",
                "symmetryWalls",
                "solid_to_fluid",
            ],
        },
        "sequences": records,
    }
    if fully_coupled:
        payload["history_mode"] = "fully_coupled_flow_heat"
    return payload


def save_statistics(path: Path, statistics: dict[str, object], full: bool) -> None:
    values = {
        "condition_mean": statistics["condition_mean"],
        "condition_std": statistics["condition_std"],
        "state_mean": statistics["state_mean"],
        "state_std": statistics["state_std"],
        "maximum_time_s": np.asarray(statistics["maximum_time_s"]),
    }
    if full:
        values.update(
            {
                "internal_mass_flux_mean_kg_s": np.asarray(
                    statistics["internal_mass_flux_mean_kg_s"]
                ),
                "internal_mass_flux_std_kg_s": np.asarray(
                    statistics["internal_mass_flux_std_kg_s"]
                ),
                "boundary_mass_flux_mean_kg_s": np.asarray(
                    statistics["boundary_mass_flux_mean_kg_s"]
                ),
                "boundary_mass_flux_std_kg_s": np.asarray(
                    statistics["boundary_mass_flux_std_kg_s"]
                ),
            }
        )
    np.savez_compressed(path, **values)


def make_common_data(tmp_path: Path):
    helper = load_full_helper()
    graph_path, residual_path = helper.write_geometry(tmp_path)
    training_records = [
        helper.write_sequence(tmp_path, f"development_{index}", float(index))
        for index in range(3)
    ]
    test_records = [
        helper.write_sequence(tmp_path, f"high_re_{index}", 20.0 + index)
        for index in range(6)
    ]
    return helper, graph_path, residual_path, training_records, test_records


def test_fixed_frozen_model_reads_six_curves_without_training(tmp_path: Path) -> None:
    _, graph_path, _, training_records, test_records = make_common_data(tmp_path)
    training_index = dataset_payload(
        graph=graph_path, records=training_records, fully_coupled=False
    )
    test_index = dataset_payload(
        graph=graph_path, records=test_records, fully_coupled=False
    )
    training_index_path = tmp_path / "fixed_training_index.json"
    test_index_path = tmp_path / "fixed_test_index.json"
    training_index_path.write_text(json.dumps(training_index), encoding="utf-8")
    test_index_path.write_text(json.dumps(test_index), encoding="utf-8")

    graph = P418ThermalStepRegionalGraph.from_npz(graph_path)
    statistics = fixed_statistics(
        tmp_path,
        fixed_records(training_index),
        ["development_0"],
        graph.node_type.numpy(),
    )
    model_dir = tmp_path / "fixed_model"
    model_dir.mkdir()
    statistics_path = model_dir / "training_statistics.npz"
    save_statistics(statistics_path, statistics, full=False)
    model = HCCBP418SpatiotemporalRegionalOperator(
        condition_dim=8,
        boundary_role_count=graph.boundary_role_count,
        spatial_temporal_mode="factorized_static_spatial",
        temperature_output_mode="literature_bounded_logit",
        temperature_mean_k_by_node_type=torch.as_tensor(
            np.asarray(statistics["state_mean"])[:, 4]
        ),
        temperature_std_k_by_node_type=torch.as_tensor(
            np.asarray(statistics["state_std"])[:, 4]
        ),
        temperature_bounds_k_by_node_type=torch.tensor(
            ((300.0, 1000.0), (298.0, 1300.0))
        ),
        **ARCHITECTURE,
    )
    state_path = model_dir / "model_state.pt"
    torch.save(model.state_dict(), state_path)
    summary = {
        "status": "completed_p418_spatiotemporal_regional_operator",
        "architecture": {
            **ARCHITECTURE,
            "spatial_temporal_mode": "factorized_static_spatial",
            "temperature_output_mode": "literature_bounded_logit",
            "temperature_output_bounds_K_by_node_type": {
                "fluid": [300.0, 1000.0],
                "solid": [298.0, 1300.0],
            },
        },
        "split_case_ids": {
            "train": ["development_0"],
            "validation": ["development_1"],
            "test": ["development_2"],
        },
    }
    summary_path = model_dir / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    output = tmp_path / "fixed_output"
    completed = subprocess.run(
        [
            sys.executable,
            str(EVALUATOR),
            "--mode",
            "fixed",
            "--training-summary",
            str(summary_path),
            "--training-dataset-index",
            str(training_index_path),
            "--test-dataset-index",
            str(test_index_path),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert result["independent_reference_curve_count"] == 6
    assert result["training_or_model_selection_performed"] is False
    assert result["new_physical_parameters"] == []
    assert set(result["prediction_files"]) == {"test"}
    assert len(result["per_curve_metrics"]) == 6
    assert (
        result["aggregate_metrics"][
            "predicted_fluid_temperature_outside_registered_range_fraction"
        ]
        == 0.0
    )
    assert (
        result["aggregate_metrics"][
            "predicted_solid_temperature_outside_registered_range_fraction"
        ]
        == 0.0
    )


def test_fully_coupled_frozen_model_predicts_full_state_and_flux(
    tmp_path: Path,
) -> None:
    _, graph_path, residual_path, training_records, test_records = make_common_data(
        tmp_path
    )
    training_index = dataset_payload(
        graph=graph_path, records=training_records, fully_coupled=True
    )
    test_index = dataset_payload(
        graph=graph_path, records=test_records, fully_coupled=True
    )
    training_index_path = tmp_path / "full_training_index.json"
    test_index_path = tmp_path / "full_test_index.json"
    training_index_path.write_text(json.dumps(training_index), encoding="utf-8")
    test_index_path.write_text(json.dumps(test_index), encoding="utf-8")

    graph = P418ThermalStepRegionalGraph.from_npz(graph_path)
    statistics = full_statistics(
        tmp_path,
        full_records(training_index),
        ["development_0"],
        graph.node_type.numpy(),
    )
    geometry = load_p418_subface_geometry(
        residual_path,
        fluid_patch_names=training_index["boundary_patch_names"]["fluid"],
        solid_patch_names=training_index["boundary_patch_names"]["solid"],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    flux_graph = build_p418_fully_coupled_flux_graph(
        geometry=geometry, graph=graph
    )
    model_dir = tmp_path / "full_model"
    model_dir.mkdir()
    statistics_path = model_dir / "training_statistics.npz"
    save_statistics(statistics_path, statistics, full=True)
    model = HCCBP418FullyCoupledRegionalOperator(
        condition_dim=8,
        boundary_role_count=graph.boundary_role_count,
        internal_face_feature_dim=flux_graph.internal_features.shape[1],
        boundary_face_feature_dim=flux_graph.boundary_features.shape[1],
        **ARCHITECTURE,
    )
    state_path = model_dir / "model_state.pt"
    torch.save(model.state_dict(), state_path)
    records = full_records(training_index)
    time_s, condition, state, internal, boundary = load_full_sequence(
        tmp_path, records["development_0"]
    )
    reference_state = torch.as_tensor(state[None], dtype=torch.float32)
    reference_residual = assemble_p418_fully_coupled_transient_residual(
        geometry=geometry,
        step_condition=torch.as_tensor(condition[None], dtype=torch.float32),
        state_physical=reference_state,
        time_s=torch.as_tensor(time_s, dtype=torch.float32),
        fluid_internal_mass_flux_kg_s=torch.as_tensor(
            internal[None], dtype=torch.float32
        ),
        fluid_boundary_mass_flux_kg_s=torch.as_tensor(
            boundary[None], dtype=torch.float32
        ),
    )
    equation_scales = training_equation_scales(
        [reference_residual], [reference_state]
    )
    summary = {
        "status": "completed_p418_fully_coupled_spatiotemporal_operator",
        "architecture": {
            **ARCHITECTURE,
            "revision": FULLY_COUPLED_ARCHITECTURE_REVISION,
        },
        "split_sequence_ids": {
            "train": ["development_0"],
            "validation": ["development_1"],
            "test": ["development_2"],
        },
        "model_state_sha256": sha256_file(state_path),
        "equation_scales_from_training_curves": {
            field.name: float(getattr(equation_scales, field.name))
            for field in fields(equation_scales)
        },
    }
    summary_path = model_dir / "final_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    output = tmp_path / "full_output"
    completed = subprocess.run(
        [
            sys.executable,
            str(EVALUATOR),
            "--mode",
            "fully_coupled",
            "--training-summary",
            str(summary_path),
            "--training-dataset-index",
            str(training_index_path),
            "--test-dataset-index",
            str(test_index_path),
            "--residual-geometry",
            str(residual_path),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert result["independent_reference_curve_count"] == 6
    assert result["training_or_model_selection_performed"] is False
    assert result["new_physical_parameters"] == []
    assert len(result["prediction_files"]["test"]) == 6
    assert len(result["per_curve_metrics"]) == 6
    assert (
        result["aggregate_metrics"]["equation_scale_source"]
        == "training_curves_only"
    )
    for name in (
        "continuity",
        "momentum",
        "fluid_energy",
        "solid_energy",
        "interface_flux",
        "interface_temperature",
        "internal_mass_flux_consistency",
        "boundary_mass_flux_consistency",
    ):
        assert (
            f"physics_difference_{name}_normalized_RMSE"
            in result["aggregate_metrics"]
        )

    summary["architecture"]["revision"] = "obsolete_structure"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    stale = subprocess.run(
        [
            sys.executable,
            str(EVALUATOR),
            "--mode",
            "fully_coupled",
            "--training-summary",
            str(summary_path),
            "--training-dataset-index",
            str(training_index_path),
            "--test-dataset-index",
            str(test_index_path),
            "--residual-geometry",
            str(residual_path),
            "--output-dir",
            str(tmp_path / "stale_output"),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert stale.returncode != 0
    assert "architecture revision differs" in stale.stderr


def test_new_runners_default_to_plan_only_and_obey_pause(tmp_path: Path) -> None:
    full_runner = ROOT / "code/run_hccb_p418_fully_coupled_model_stage.sh"
    evaluation_runner = (
        ROOT / "code/run_hccb_p418_high_re_independent_evaluation.sh"
    )
    for runner in (full_runner, evaluation_runner):
        subprocess.run(["bash", "-n", str(runner)], check=True)
    planned = subprocess.run(
        ["bash", str(evaluation_runner)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert planned.returncode == 0
    assert "没有启动训练或推理" in planned.stdout
    marker = tmp_path / "pause"
    marker.write_text("pause\n", encoding="utf-8")
    paused = subprocess.run(
        ["bash", str(full_runner)],
        cwd=ROOT,
        env={
            "ROOT": str(ROOT),
            "EXECUTE": "1",
            "DEVICE": "cpu",
            "PAUSE_MARKER": str(marker),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert paused.returncode == 3
    assert "paused for cloud migration" in paused.stderr
