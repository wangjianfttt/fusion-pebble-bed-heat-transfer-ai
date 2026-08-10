from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/estimate_hccb_p418_transient_training_wall_time.py"


def measured(path: Path, seconds: float, *, time_points: int = 56) -> None:
    path.write_text(
        json.dumps(
            {
                "elapsed_seconds": seconds,
                "nodes": 100,
                "edges": 200,
                "time_points": time_points,
                "peak_gpu_GB": 10.0,
            }
        ),
        encoding="utf-8",
    )


def test_runtime_projection_counts_primary_splits_and_extra_seeds(tmp_path: Path) -> None:
    splits = tmp_path / "splits.json"
    split = {"train": [f"c{i}" for i in range(6)], "validation": [], "test": []}
    splits.write_text(json.dumps({"splits": {"a": split, "b": split, "c": split}}))
    repeated = tmp_path / "repeated.json"
    factorized = tmp_path / "factorized.json"
    diffusion = tmp_path / "diffusion.json"
    data_only = tmp_path / "data_only.json"
    measured(data_only, 4.0)
    measured(repeated, 10.0)
    measured(factorized, 6.0)
    measured(diffusion, 9.0)
    output = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--splits",
            str(splits),
            "--repeated-graph-summary",
            str(repeated),
            "--data-only-graph-summary",
            str(data_only),
            "--factorized-graph-summary",
            str(factorized),
            "--diffusion-summary",
            str(diffusion),
            "--epochs",
            "500",
            "--seed-count",
            "3",
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    # Five repeated energy runs, five data-only runs, three factorized runs and
    # five diffusion runs; every run uses six full curves for 500 epochs.
    assert summary["projected_training_hours_low"] == pytest.approx(110.83333333333334)
    assert summary["projected_training_hours_high"] == pytest.approx(110.83333333333334)
    assert summary["data_only_timing_measured"] is True
    assert summary["measured_time_points"] == 56
    assert summary["measured_seconds_per_full_curve_update"] == {
        "data_only_graph": 4.0,
        "energy_flux_graph": 10.0,
        "factorized_energy_flux_graph": 6.0,
        "diffusion_residual": 9.0,
    }
    assert summary["new_physical_parameters"] == []
    assert (output / "training_components.csv").is_file()
    assert "完整模型阶段应按至少7--9天安排" in (
        output / "README_CN.md"
    ).read_text(encoding="utf-8")


def test_runtime_projection_rejects_inconsistent_curve_size(tmp_path: Path) -> None:
    splits = tmp_path / "splits.json"
    split = {"train": ["c0"], "validation": [], "test": []}
    splits.write_text(json.dumps({"splits": {"a": split}}))
    repeated = tmp_path / "repeated.json"
    factorized = tmp_path / "factorized.json"
    diffusion = tmp_path / "diffusion.json"
    measured(repeated, 10.0)
    measured(factorized, 6.0, time_points=37)
    measured(diffusion, 9.0)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--splits",
            str(splits),
            "--repeated-graph-summary",
            str(repeated),
            "--data-only-graph-summary",
            str(tmp_path / "missing.json"),
            "--factorized-graph-summary",
            str(factorized),
            "--diffusion-summary",
            str(diffusion),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "does not match" in result.stderr
