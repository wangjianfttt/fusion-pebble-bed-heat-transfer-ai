#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def write_geometry(root: Path) -> tuple[Path, Path]:
    graph = root / "regional_sequence_geometry.npz"
    np.savez_compressed(
        graph,
        node_centroid_m=np.asarray(
            [[0.0, 0.0, 0.5], [0.0, 0.0, 1.5], [0.0, 1.0, 0.5]]
        ),
        node_volume_m3=np.ones(3),
        node_type=np.asarray([0, 0, 1], dtype=np.int8),
        fluid_global_region=np.asarray([0, 1]),
        solid_global_region=np.asarray([2]),
        regional_graph_level=np.asarray(1),
        edge_source=np.asarray([0, 1, 0, 2]),
        edge_target=np.asarray([1, 0, 2, 0]),
        edge_kind=np.asarray([0, 0, 2, 2]),
        edge_area_m2=np.ones(4),
        edge_area_vector_m2=np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
        ),
    )
    residual = root / "residual_geometry.npz"
    fluid_boundary_centroid = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [-0.5, 0.0, 0.5],
            [0.5, 0.0, 1.5],
            [0.0, 0.5, 0.5],
        ]
    )
    solid_boundary_centroid = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.5, 1.0, 0.5],
            [-0.5, 1.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
    )
    np.savez_compressed(
        residual,
        fluid_cell_centroid_m=np.asarray([[0.0, 0.0, 0.5], [0.0, 0.0, 1.5]]),
        fluid_cell_volume_m3=np.ones(2),
        fluid_internal_subface_centroid_m=np.asarray([[0.0, 0.0, 1.0]]),
        fluid_internal_subface_area_vector_m2=np.asarray([[0.0, 0.0, 1.0]]),
        fluid_internal_subface_owner=np.asarray([0]),
        fluid_internal_subface_neighbour=np.asarray([1]),
        fluid_boundary_face_centroid_m=fluid_boundary_centroid,
        fluid_boundary_face_area_vector_m2=np.asarray(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        fluid_boundary_face_owner=np.asarray([0, 1, 0, 1, 0]),
        solid_cell_centroid_m=np.asarray([[0.0, 1.0, 0.5]]),
        solid_cell_volume_m3=np.ones(1),
        solid_internal_subface_centroid_m=np.empty((0, 3)),
        solid_internal_subface_area_vector_m2=np.empty((0, 3)),
        solid_internal_subface_owner=np.empty(0, dtype=np.int64),
        solid_internal_subface_neighbour=np.empty(0, dtype=np.int64),
        solid_boundary_face_centroid_m=solid_boundary_centroid,
        solid_boundary_face_area_vector_m2=np.asarray(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        ),
        solid_boundary_face_owner=np.zeros(5, dtype=np.int64),
        interface_fluid_boundary_face=np.asarray([4]),
        interface_solid_boundary_face=np.asarray([4]),
        fluid_boundary_face_patch=np.arange(5),
        solid_boundary_face_patch=np.arange(5),
        fine_to_regional_global=np.asarray([0, 1, 2]),
        fluid_global_region=np.asarray([0, 1]),
        solid_global_region=np.asarray([2]),
    )
    return graph, residual


def write_sequence(root: Path, sequence_id: str, offset: float) -> dict[str, object]:
    time = np.asarray([0.0, 1.0, 3.0], dtype=np.float32)
    state = np.zeros((3, 3, 5), dtype=np.float32)
    for index, value in enumerate(time):
        state[index, :2, 0] = 0.01 + 0.002 * value + 0.001 * offset
        state[index, :2, 1] = 0.005 + 0.001 * value
        state[index, :2, 2] = 0.20 + 0.01 * value + 0.002 * offset
        state[index, :2, 3] = 120000.0 + 100.0 * value + offset
        state[index, :2, 4] = 700.0 + 5.0 * value + offset
        state[index, 2, 4] = 650.0 + 3.0 * value + offset
    internal = (0.01 + 0.001 * time + 0.0001 * offset)[:, None]
    boundary = np.zeros((3, 5), dtype=np.float32)
    boundary[:, 0] = -internal[:, 0]
    boundary[:, 1] = internal[:, 0]
    path = root / f"{sequence_id}.npz"
    np.savez_compressed(
        path,
        sequence_id=np.asarray(sequence_id),
        time_s=time,
        condition_physical=np.asarray(
            [0.2, 700.0 + offset, 4.85, 0.25, 900.0, 8.85, 120000.0, 635.0],
            dtype=np.float32,
        ),
        state_physical=state,
        fluid_internal_mass_flux_kg_s=internal,
        fluid_boundary_mass_flux_kg_s=boundary,
    )
    return {
        "sequence_id": sequence_id,
        "sequence_file": path.name,
        "complete": True,
    }


def test_full_trainer_uses_complete_curve_roles_and_writes_restart(tmp_path: Path) -> None:
    graph, residual = write_geometry(tmp_path)
    records = [
        write_sequence(tmp_path, "train_curve", 0.0),
        write_sequence(tmp_path, "validation_curve", 20.0),
        write_sequence(tmp_path, "test_curve", 40.0),
    ]
    index = {
        "history_mode": "fully_coupled_flow_heat",
        "sequence_count": 3,
        "state_names": ["Ux_m_s", "Uy_m_s", "Uz_m_s", "pressure_Pa", "temperature_K"],
        "condition_names": [
            "source_inlet_velocity_m_s",
            "source_inlet_temperature_K",
            "source_solid_heat_source_MW_m3",
            "target_inlet_velocity_m_s",
            "target_inlet_temperature_K",
            "target_solid_heat_source_MW_m3",
            "target_outlet_pressure_Pa",
            "target_cooling_wall_temperature_K",
        ],
        "regional_geometry_file": graph.name,
        "boundary_patch_names": {
            "fluid": ["inlet", "outlet", "coolingWall", "symmetryWalls", "fluid_to_solid"],
            "solid": ["inlet", "outlet", "coolingWall", "symmetryWalls", "solid_to_fluid"],
        },
        "sequences": records,
    }
    index_path = tmp_path / "dataset_index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    split_path = tmp_path / "splits.json"
    split_path.write_text(
        json.dumps(
            {
                "splits": {
                    "smoke": {
                        "train": ["train_curve"],
                        "validation": ["validation_curve"],
                        "test": ["test_curve"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    command = [
        sys.executable,
        str(ROOT / "code/train_hccb_p418_fully_coupled_spatiotemporal_operator.py"),
        "--dataset-index", str(index_path),
        "--splits", str(split_path),
        "--split-name", "smoke",
        "--residual-geometry", str(residual),
        "--output-dir", str(output),
        "--run-role", "smoke",
        "--epochs", "1",
        "--learning-rate", "1e-7",
        "--weight-decay", "0",
        "--state-weight", "1",
        "--face-flux-weight", "1",
        "--physics-weight", "1",
        "--hidden-dim", "8",
        "--local-pre-iterations", "1",
        "--physics-attention-blocks", "1",
        "--local-post-iterations", "1",
        "--physics-attention-heads", "1",
        "--physics-slices", "2",
        "--temporal-layers", "1",
        "--temporal-heads", "1",
        "--temporal-node-chunk-size", "16",
        "--device", "cpu",
        "--torch-threads", "1",
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["training_normalization_sequence_ids"] == ["train_curve"]
    assert summary["selection_split"] == "validation"
    assert summary["test_used_after_model_selection_only"] is False
    assert set(summary["metrics"]) == {"train", "validation", "test"}
    assert summary["loss_balancing"]["method"] == "fixed"
    assert "selection_score" in summary["metrics"]["validation"]
    assert "best_validation_selection_score" in summary
    assert summary["new_physical_parameters"] == []
    assert (
        summary["architecture"]["revision"]
        == "p418_fully_coupled_oriented_initial_face_flux_context_v2"
    )
    assert len(summary["architecture"]["model_implementation_sha256"]) == 64
    assert (output / "training_checkpoint.pt").is_file()
    assert (output / "model_state.pt").is_file()
    assert (output / "final_summary.json").is_file()
    assert len(summary["equation_scales_from_training_curves"]) == 8
    subprocess.run(
        command + ["--resume"], cwd=ROOT, check=True, capture_output=True, text=True
    )

    adaptive_output = tmp_path / "adaptive_output"
    adaptive_command = command.copy()
    adaptive_command[adaptive_command.index(str(output))] = str(adaptive_output)
    adaptive_command.extend(
        [
            "--loss-balance-method",
            "relobralo",
            "--relobralo-temperature",
            "0.1",
            "--relobralo-alpha",
            "0.999",
            "--relobralo-rho",
            "0.9999",
            "--evaluation-stage",
            "selection",
        ]
    )
    subprocess.run(
        adaptive_command, cwd=ROOT, check=True, capture_output=True, text=True
    )
    adaptive_summary = json.loads(
        (adaptive_output / "summary.json").read_text(encoding="utf-8")
    )
    assert adaptive_summary["loss_balancing"]["method"] == "relobralo"
    assert adaptive_summary["evaluation_stage"] == "selection"
    assert adaptive_summary["test_evaluated"] is False
    assert adaptive_summary["test_used_after_model_selection_only"] is False
    assert set(adaptive_summary["metrics"]) == {"train", "validation"}
    assert set(adaptive_summary["prediction_files"]) == {"train", "validation"}
    assert not list(adaptive_output.glob("test_*_prediction.npz"))
    assert (adaptive_output / "selection_summary.json").is_file()
    selected = adaptive_summary["loss_balancing"]["selected_checkpoint_state"]
    assert selected["step"] == 1
    assert np.isclose(sum(selected["weights"]), 3.0)
    subprocess.run(
        adaptive_command + ["--resume"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
