from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/run_hccb_p418_selected_loss_downstream.py"
SPLIT = "pair_disjoint_stress_test"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_selected_final(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    result_dir = tmp_path / "results"
    loss_root = result_dir / "fixed_flow_loss_balancing_pair_disjoint_stress_test"
    selected_id = "fixed_registered_5_1_1"
    selected_dir = loss_root / selected_id
    selected_dir.mkdir(parents=True)
    selection_path = loss_root / "selected_loss_balancing_method.json"
    selection_path.write_text(
        json.dumps(
            {
                "status": "p418_loss_balancing_selected_on_validation_only",
                "independent_test_read": False,
                "selected_candidate_id": selected_id,
            }
        ),
        encoding="utf-8",
    )
    final = {
        "status": "completed_p418_spatiotemporal_regional_operator",
        "evaluation_stage": "final",
        "test_evaluated": True,
        "split_name": SPLIT,
        "loss_balancing": {"candidate_id": selected_id},
        "selected_method_record_sha256": sha256(selection_path),
    }
    text = json.dumps(final)
    (selected_dir / "summary.json").write_text(text, encoding="utf-8")
    (selected_dir / "final_summary.json").write_text(text, encoding="utf-8")
    dataset = tmp_path / "dataset.json"
    splits = tmp_path / "splits.json"
    geometry = tmp_path / "geometry.npz"
    dataset.write_text("{}", encoding="utf-8")
    splits.write_text("{}", encoding="utf-8")
    geometry.write_bytes(b"fixture")
    return result_dir, dataset, splits, geometry


def test_plan_uses_selected_model_for_all_downstream_results(tmp_path: Path) -> None:
    result_dir, dataset, splits, geometry = prepare_selected_final(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset-index",
            str(dataset),
            "--splits",
            str(splits),
            "--residual-geometry",
            str(geometry),
            "--result-dir",
            str(result_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)
    assert plan["status"] == "plan_only_no_training_started"
    commands = "\n".join(plan["commands"])
    selected_dir = (
        result_dir
        / "fixed_flow_loss_balancing_pair_disjoint_stress_test"
        / "fixed_registered_5_1_1"
    )
    assert commands.count(str(selected_dir)) >= 3
    assert "formal_factorized" in commands
    assert "train_hccb_p418_low_rank_temperature_residual.py" in commands
    assert "train_hccb_p418_temporal_temperature_diffusion.py" in commands
    assert not (
        result_dir
        / "fixed_flow_loss_balancing_pair_disjoint_stress_test"
        / "selected_downstream_integration.json"
    ).exists()

