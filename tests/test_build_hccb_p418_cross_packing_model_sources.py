from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_cross_packing_model_sources.py"


def load_module():
    spec = importlib.util.spec_from_file_location("cross_packing_sources", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_run(
    root: Path,
    architecture: str,
    epochs: int,
    validation_loss: float,
    *,
    split: dict[str, list[str]] | None = None,
) -> Path:
    directory = (
        root
        / "results"
        / f"hccb_p418_60_{architecture}_interleaved_all_ranges_{epochs}epoch"
    )
    directory.mkdir(parents=True)
    (directory / "best.pt").write_bytes(f"{architecture}-{epochs}".encode())
    summary = {
        "status": "conservative_mixed_operator_training_complete",
        "architecture": architecture,
        "split_name": "interleaved_all_ranges",
        "split_case_ids": split
        or {"train": ["a", "b"], "validation": ["c"], "test": ["d"]},
        "epochs": epochs,
        "best_epoch": max(1, epochs // 2),
        "best_validation_total_loss": validation_loss,
        "run_provenance": {"common_comparison_fingerprint": "same-fields"},
    }
    path = directory / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def test_each_architecture_uses_validation_selected_seed101_checkpoint(tmp_path: Path):
    module = load_module()
    for architecture, loss in {
        "pinn_data_only": 0.8,
        "pinn": 0.7,
        "graph": 0.6,
        "transolver": 0.5,
    }.items():
        write_run(tmp_path, architecture, 100, loss)
    write_run(tmp_path, "pinn", 3000, 0.4)
    write_run(tmp_path, "graph", 2000, 0.9)
    plan = {
        "status": "source_epoch_followup_plan_ready",
        "runs": [
            {
                "architecture": "pinn",
                "split": "interleaved_all_ranges",
                "followup_result_directory": "results/hccb_p418_60_pinn_interleaved_all_ranges_3000epoch",
            },
            {
                "architecture": "graph",
                "split": "interleaved_all_ranges",
                "followup_result_directory": "results/hccb_p418_60_graph_interleaved_all_ranges_2000epoch",
            },
        ],
    }
    plan_path = tmp_path / "followup.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    result = module.build(
        project_root=tmp_path,
        initial_epochs=100,
        split_name="interleaved_all_ranges",
        followup_plan=plan_path,
    )
    assert result["status"] == "cross_packing_seed101_model_sources_selected"
    assert result["independent_test_used_for_selection"] is False
    assert result["seed202_fields_read"] is False
    assert result["seed303_fields_read"] is False
    assert result["models"]["pinn"]["selected_epochs"] == 3000
    assert result["models"]["graph"]["selected_epochs"] == 100
    assert result["models"]["transolver"]["selected_epochs"] == 100
    assert all(
        record["selection_data"] == "seed101 validation conditions only"
        for record in result["models"].values()
    )


def test_rejects_followup_with_different_condition_split(tmp_path: Path):
    module = load_module()
    for architecture in module.ARCHITECTURES:
        write_run(tmp_path, architecture, 100, 0.5)
    write_run(
        tmp_path,
        "pinn",
        3000,
        0.4,
        split={"train": ["a"], "validation": ["b"], "test": ["different"]},
    )
    plan_path = tmp_path / "followup.json"
    plan_path.write_text(
        json.dumps(
            {
                "status": "source_epoch_followup_plan_ready",
                "runs": [
                    {
                        "architecture": "pinn",
                        "split": "interleaved_all_ranges",
                        "followup_result_directory": "results/hccb_p418_60_pinn_interleaved_all_ranges_3000epoch",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    try:
        module.build(
            project_root=tmp_path,
            initial_epochs=100,
            split_name="interleaved_all_ranges",
            followup_plan=plan_path,
        )
    except ValueError as exc:
        assert "different condition splits" in str(exc)
    else:
        raise AssertionError("a mismatched condition split was accepted")
