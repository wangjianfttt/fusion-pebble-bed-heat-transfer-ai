from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/verify_hccb_p418_steady_seed_repeatability.py"
ARCHITECTURES = ("pinn_data_only", "pinn", "graph", "transolver")


def write_result(path: Path, seed: int, prediction: np.ndarray) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "training_seed": seed,
                "split_case_ids": {"train": ["a"], "validation": ["b"], "test": ["c"]},
                "run_provenance": {"common_comparison_fingerprint": "same"},
            }
        ),
        encoding="utf-8",
    )
    np.savez(path / "test_regional_predictions.npz", baseline_state_normalized=prediction)
    (path / "best.pt").write_bytes(f"seed={seed};value={prediction.tolist()}".encode())


def build_fixture(tmp_path: Path, repeat_shift: float = 0.0) -> Path:
    results = tmp_path / "results"
    primary = np.asarray([[[0.1, 0.2]]], dtype=np.float32)
    for architecture in ARCHITECTURES:
        base = f"primary_{architecture}_completed_smoke_1epoch"
        write_result(results / base, 20260717, primary)
        write_result(
            results / f"repeat_{architecture}_completed_smoke_1epoch",
            20260717,
            primary + np.float32(repeat_shift),
        )
        write_result(results / f"{base}_seed20260718", 20260718, primary + 0.1)
        write_result(results / f"{base}_seed20260719", 20260719, primary - 0.1)
    return results


def command(tmp_path: Path, results: Path) -> list[str]:
    return [
        "python3",
        str(SCRIPT),
        "--results-root",
        str(results),
        "--primary-prefix",
        "primary",
        "--repeat-prefix",
        "repeat",
        "--output-dir",
        str(tmp_path / "output"),
    ]


def test_realistic_same_seed_roundoff_and_different_seed_change_are_accepted(tmp_path: Path) -> None:
    results = build_fixture(tmp_path, repeat_shift=1.0e-7)
    completed = subprocess.run(command(tmp_path, results), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads((tmp_path / "output/summary.json").read_text(encoding="utf-8"))
    assert payload["status"] == "steady_neural_seed_repeatability_verified"
    assert len(payload["results"]) == 4
    assert (tmp_path / "output/README_CN.md").is_file()


def test_large_same_seed_difference_is_rejected(tmp_path: Path) -> None:
    results = build_fixture(tmp_path, repeat_shift=0.01)
    completed = subprocess.run(command(tmp_path, results), capture_output=True, text=True)
    assert completed.returncode != 0
    assert "exceeds the float32 accumulation tolerance" in completed.stderr
