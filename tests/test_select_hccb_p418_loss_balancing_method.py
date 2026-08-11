from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summary(candidate: dict[str, object], score: float) -> dict[str, object]:
    candidate_id = str(candidate["candidate_id"])
    weights = {
        "temperature_data": float(candidate["state_weight"]),
        "reference_edge_energy_flux": float(candidate["face_flux_weight"]),
        "projection_aware_transient_energy": float(candidate["physics_weight"]),
    }
    checkpoint_state: dict[str, object] = {
        "method": candidate["method"],
        "weights": list(weights.values()),
    }
    for name in ("temperature", "alpha", "expected_rho"):
        if name in candidate:
            checkpoint_state[name] = candidate[name]
    return {
        "evaluation_stage": "selection",
        "test_evaluated": False,
        "dataset_index": "/data/dataset_index.json",
        "input_file_sha256": {
            "dataset_index": "dataset-sha",
            "split_file": "split-sha",
            "residual_geometry": "geometry-sha",
            "loss_balancing_sources": "sources-sha",
        },
        "split_name": "direction_up_test",
        "split_sequence_ids": {
            "train": ["curve_01"],
            "validation": ["curve_02"],
            "test": ["curve_03"],
        },
        "seed": 20260723,
        "architecture": {"hidden_dim": 96},
        "physics_terms": ["continuity", "momentum"],
        "training_normalization_sequence_ids": ["curve_01"],
        "loss_weights": weights,
        "loss_balancing": {
            "candidate_id": candidate_id,
            "method": candidate["method"],
            "selected_checkpoint_state": checkpoint_state,
        },
        "best_validation_selection_score": score,
        "best_epoch": 12,
        "metrics": {
            "train": {"selection_score": score * 0.8},
            "validation": {"selection_score": score},
        },
    }


def write_protocol(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    sources = json.loads(
        (ROOT / "parameters/hccb_p418_loss_balancing_sources.json").read_text(
            encoding="utf-8"
        )
    )
    source_path = tmp_path / "sources.json"
    source_path.write_text(json.dumps(sources), encoding="utf-8")
    candidate_root = tmp_path / "candidates"
    candidates = sources["formal_candidates"]
    candidate_ids = [row["candidate_id"] for row in candidates]
    for index, candidate in enumerate(candidates):
        candidate_id = candidate["candidate_id"]
        output = candidate_root / candidate_id
        output.mkdir(parents=True)
        (output / "selection_summary.json").write_text(
            json.dumps(summary(candidate, 4.0 - index)),
            encoding="utf-8",
        )
    return source_path, candidate_root, candidate_ids


def test_selector_uses_validation_only_and_preserves_summary_hash(
    tmp_path: Path,
) -> None:
    source_path, candidate_root, candidate_ids = write_protocol(tmp_path)
    output = tmp_path / "selected_method.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "code/select_hccb_p418_loss_balancing_method.py"),
            "--candidate-root",
            str(candidate_root),
            "--sources",
            str(source_path),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    selected = json.loads(output.read_text(encoding="utf-8"))
    expected_id = candidate_ids[-1]
    expected_summary = candidate_root / expected_id / "selection_summary.json"
    assert selected["selected_candidate_id"] == expected_id
    assert selected["independent_test_read"] is False
    assert selected["selected_summary_path"] == str(expected_summary)
    assert selected["selected_summary_sha256"] == hashlib.sha256(
        expected_summary.read_bytes()
    ).hexdigest()
    assert len(selected["candidate_records"]) == 4


def test_selector_rejects_candidate_that_already_read_test(
    tmp_path: Path,
) -> None:
    source_path, candidate_root, candidate_ids = write_protocol(tmp_path)
    bad_path = candidate_root / candidate_ids[1] / "selection_summary.json"
    bad = json.loads(bad_path.read_text(encoding="utf-8"))
    bad["test_evaluated"] = True
    bad["metrics"]["test"] = {"selection_score": 0.0}
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "code/select_hccb_p418_loss_balancing_method.py"),
            "--candidate-root",
            str(candidate_root),
            "--sources",
            str(source_path),
            "--output",
            str(tmp_path / "selected_method.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    assert "read the independent test curves too early" in process.stderr


def test_selector_rejects_noncomparable_candidate(tmp_path: Path) -> None:
    source_path, candidate_root, candidate_ids = write_protocol(tmp_path)
    bad_path = candidate_root / candidate_ids[2] / "selection_summary.json"
    bad = copy.deepcopy(json.loads(bad_path.read_text(encoding="utf-8")))
    bad["seed"] = 9
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "code/select_hccb_p418_loss_balancing_method.py"),
            "--candidate-root",
            str(candidate_root),
            "--sources",
            str(source_path),
            "--output",
            str(tmp_path / "selected_method.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    assert "do not share one data/model setting" in process.stderr


def test_selector_rejects_candidate_with_unregistered_loss_setting(
    tmp_path: Path,
) -> None:
    source_path, candidate_root, candidate_ids = write_protocol(tmp_path)
    bad_path = candidate_root / candidate_ids[1] / "selection_summary.json"
    bad = json.loads(bad_path.read_text(encoding="utf-8"))
    bad["loss_balancing"]["selected_checkpoint_state"]["temperature"] = 0.25
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "code/select_hccb_p418_loss_balancing_method.py"),
            "--candidate-root",
            str(candidate_root),
            "--sources",
            str(source_path),
            "--output",
            str(tmp_path / "selected_method.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    assert "temperature differs from its source JSON" in process.stderr
