from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/train_hccb_p418_low_rank_temperature_residual.py"


def load_module():
    sys.path.insert(0, str(ROOT / "code"))
    spec = importlib.util.spec_from_file_location(
        "train_hccb_p418_low_rank_temperature_residual", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_role(path: Path, role: str, conditions: list[float]) -> list[str]:
    times, nodes = 3, 4
    node_type = np.asarray([0, 0, 1, 1], dtype=np.int64)
    volume = np.asarray([1.0, 2.0, 1.5, 2.5], dtype=np.float32)
    baseline = np.zeros((len(conditions), times, nodes, 1), dtype=np.float32)
    target = baseline.copy()
    mean_shape = np.asarray([[0.0] * nodes, [1.0, -1.0, 2.0, -2.0], [2.0, -2.0, 4.0, -4.0]])
    mode_shape = np.asarray([[0.0] * nodes, [1.0, 0.0, -1.0, 0.0], [2.0, 0.0, -2.0, 0.0]])
    for index, condition in enumerate(conditions):
        target[index, ..., 0] = mean_shape + condition * mode_shape
    identifiers = [f"{role}_{index}" for index in range(len(conditions))]
    np.savez_compressed(
        path / f"{role}_temporal_temperature_predictions.npz",
        sequence_id=np.asarray(identifiers),
        time_s=np.broadcast_to(
            np.arange(times, dtype=np.float32), (len(conditions), times)
        ),
        condition_physical=np.zeros((len(conditions), 8), dtype=np.float32),
        condition_normalized=np.asarray(conditions, dtype=np.float32)[:, None],
        fixed_hydrodynamics_physical=np.zeros(
            (len(conditions), nodes, 4), dtype=np.float32
        ),
        fluid_internal_mass_flux_kg_s=np.zeros(
            (len(conditions), 1), dtype=np.float32
        ),
        fluid_boundary_mass_flux_kg_s=np.zeros(
            (len(conditions), 2), dtype=np.float32
        ),
        baseline_temperature_normalized=baseline,
        target_temperature_normalized=target,
        node_type=node_type,
        node_volume_m3=volume,
        node_centroid_m=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        temperature_mean_K_by_node_type=np.asarray([300.0, 600.0], dtype=np.float32),
        temperature_std_K_by_node_type=np.asarray([10.0, 20.0], dtype=np.float32),
    )
    return identifiers


def test_training_only_rank_selection_and_exact_initial_state(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction"
    prediction.mkdir()
    split_ids = {
        "train": write_role(prediction, "train", [-2.0, -1.0, 0.0, 1.0]),
        "validation": write_role(prediction, "validation", [0.5, 1.5]),
        "test": write_role(prediction, "test", [-1.5, 2.0]),
    }
    (prediction / "summary.json").write_text(
        json.dumps({"split_case_ids": split_ids}), encoding="utf-8"
    )
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--prediction-dir",
            str(prediction),
            "--output-dir",
            str(output),
            "--split-name",
            "synthetic_split",
            "--run-role",
            "software_smoke",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["selected_rank"] == 1
    assert summary["temperature_metric_definition"] == (
        "regional-volume-weighted RMSE, reported separately for fluid and solid"
    )
    assert summary["metrics"]["test"]["solid_temperature_RMSE_K"] < 1.0e-10
    assert summary["metrics"]["test"]["solid_maximum_temperature_history_RMSE_K"] < 1.0e-10
    with np.load(output / "test_low_rank_temperature_predictions.npz") as data:
        np.testing.assert_allclose(
            data["corrected_temperature_normalized"][:, 0],
            data["baseline_temperature_normalized"][:, 0],
            atol=0.0,
        )
        assert "fixed_hydrodynamics_physical" in data.files
        assert "condition_physical" in data.files
        assert "fluid_internal_mass_flux_kg_s" in data.files
        assert "fluid_boundary_mass_flux_kg_s" in data.files
        assert "node_centroid_m" in data.files


def test_formal_low_rank_contract_requires_twelve_disjoint_physical_curves(
    tmp_path: Path,
) -> None:
    module = load_module()
    identifiers = {
        "train": np.asarray([f"train_{index}" for index in range(8)]),
        "validation": np.asarray(["validation_0", "validation_1"]),
        "test": np.asarray(["test_0", "test_1"]),
    }
    for role in identifiers:
        (tmp_path / f"{role}_temporal_temperature_predictions.npz").write_bytes(
            b"fixture"
        )
    summary = {
        "status": "completed_p418_spatiotemporal_regional_operator",
        "run_role": "formal",
        "physics_mode": "energy_and_flux",
        "selection_split": "validation",
        "new_physical_parameters": [],
        "split_case_ids": {
            role: values.tolist() for role, values in identifiers.items()
        },
        "temporal_temperature_prediction_files": {
            role: f"{role}_temporal_temperature_predictions.npz"
            for role in identifiers
        },
    }
    splits = {
        role: {"sequence_id": values} for role, values in identifiers.items()
    }
    module.validate_deterministic_prediction_contract(
        summary=summary,
        splits=splits,
        prediction_dir=tmp_path,
        run_role="formal",
    )

    summary["run_role"] = "smoke"
    try:
        module.validate_deterministic_prediction_contract(
            summary=summary,
            splits=splits,
            prediction_dir=tmp_path,
            run_role="formal",
        )
    except ValueError as error:
        assert "formal deterministic run" in str(error)
    else:
        raise AssertionError("formal POD correction accepted a smoke-run predictor")
