from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from summarize_hccb_p418_transient_learning_curve import summarize, write_tex  # noqa: E402


RUNS = (
    ("transient_learning_n03_up", ["up_a", "up_b", "up_c"]),
    ("transient_learning_n03_down", ["down_a", "down_b", "down_c"]),
    (
        "transient_learning_n06_both",
        ["up_a", "up_b", "up_c", "down_a", "down_b", "down_c"],
    ),
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def fixture(tmp_path: Path) -> tuple[Path, Path]:
    result_root = tmp_path / "results"
    splits = {}
    for index, (name, train) in enumerate(RUNS):
        split = {
            "train": train,
            "validation": ["validation_up", "validation_down"],
            "test": ["test_1", "test_2", "test_3", "test_4"],
            "unused": [],
        }
        splits[name] = split
        write_json(
            result_root / name / "summary.json",
            {
                "status": "completed_p418_spatiotemporal_regional_operator",
                "split_name": name,
                "split_case_ids": {
                    role: split[role] for role in ("train", "validation", "test")
                },
                "new_physical_parameters": [],
                "physics_mode": "energy_and_flux",
                "architecture": {
                    "spatial_temporal_mode": "factorized_static_spatial"
                },
                "seed": 20260717,
                "selected_epoch": 50 + index,
                "training_seconds": 100.0 + index,
                "metrics": {
                    "test": {
                        "solid_temperature_RMSE_K": 10.0 - index,
                        "fluid_temperature_RMSE_K": 8.0 - index,
                        "projection_aware_energy_normalized_RMSE": 0.2 - 0.05 * index,
                    }
                },
            },
        )
    split_path = tmp_path / "splits.json"
    write_json(split_path, {"splits": splits})
    return result_root, split_path


def test_summarizes_only_complete_trajectory_runs(tmp_path: Path) -> None:
    result_root, splits = fixture(tmp_path)
    summary, rows = summarize(result_root, splits)
    assert summary["status"] == "completed_p418_transient_learning_curve"
    assert summary["training_trajectory_counts"] == [3, 6]
    assert summary["fixed_validation_trajectory_count"] == 2
    assert summary["fixed_test_trajectory_count"] == 4
    assert [row["training_direction"] for row in rows] == ["up", "down", "both"]
    tex = tmp_path / "table.tex"
    write_tex(tex, rows)
    assert "saved time points are not counted as independent samples" in tex.read_text(
        encoding="utf-8"
    )


def test_rejects_changed_test_split_or_unfinished_run(tmp_path: Path) -> None:
    result_root, splits = fixture(tmp_path)
    source = result_root / "transient_learning_n03_down/summary.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["split_case_ids"]["test"] = ["different_test"]
    write_json(source, payload)
    with pytest.raises(ValueError, match="test trajectories differ"):
        summarize(result_root, splits)

    result_root, splits = fixture(tmp_path / "unfinished")
    source = result_root / "transient_learning_n06_both/summary.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["status"] = "training_in_progress"
    write_json(source, payload)
    with pytest.raises(ValueError, match="unfinished"):
        summarize(result_root, splits)


def test_runner_generates_the_declared_summary_and_table() -> None:
    runner = (ROOT / "code/run_hccb_p418_transient_learning_curve.sh").read_text(
        encoding="utf-8"
    )
    assert "summarize_hccb_p418_transient_learning_curve.py" in runner
    assert "hccb_p418_physical_steps_12/regional_sequences/dataset_index.json" in runner
    assert "hccb_p418_subface_residual_geometry_r2/subface_residual_geometry.npz" in runner
    assert "hccb_p418_step_responses/regional_sequences" not in runner
    assert "transient_learning_curve.csv" not in runner
    assert "generated_transient_learning_curve.tex" in runner
