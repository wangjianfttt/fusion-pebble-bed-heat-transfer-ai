from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/verify_hccb_p418_cross_packing_fixed_model.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fixed_cross_packing", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path):
    architecture = "graph"
    checkpoint = tmp_path / "results/model/best.pt"
    summary = tmp_path / "results/model/summary.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixed weights")
    summary.write_text('{"status":"training"}', encoding="utf-8")
    sources = tmp_path / "results/model_sources.json"
    sources.write_text(
        json.dumps(
            {
                "status": "cross_packing_seed101_model_sources_selected",
                "models": {
                    architecture: {
                        "selected_checkpoint": str(checkpoint.relative_to(tmp_path)),
                        "selected_checkpoint_sha256": digest(checkpoint),
                        "selected_summary": str(summary.relative_to(tmp_path)),
                        "selected_summary_sha256": digest(summary),
                        "selected_epochs": 2000,
                        "selected_epoch": 1450,
                        "selected_validation_total_loss": 0.12,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    selection = tmp_path / "results/selection.json"
    selection.write_text(
        json.dumps(
            {
                "status": "seed202_architecture_fixed_before_seed303",
                "selected_architecture": architecture,
                "seed303_fields_read": False,
                "seed101_model_sources_sha256": digest(sources),
                "seed101_checkpoint_selection": {
                    architecture: {
                        "selected_epochs": 2000,
                        "selected_epoch": 1450,
                        "selected_validation_total_loss": 0.12,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    seed202 = tmp_path / "results/seed202.json"
    seed202.write_text(
        json.dumps(
            {
                "status": "cross_packing_conservative_evaluation_complete",
                "packing_seed": 202,
                "architecture": architecture,
                "checkpoint_sha256": digest(checkpoint),
                "training_summary_sha256": digest(summary),
            }
        ),
        encoding="utf-8",
    )
    return architecture, checkpoint, summary, sources, selection, seed202


def verify(module, tmp_path: Path, values):
    architecture, _, _, sources, selection, seed202 = values
    return module.verify(
        selection_path=selection,
        model_sources_path=sources,
        project_root=tmp_path,
        architecture=architecture,
        seed202_result_path=seed202,
    )


def test_seed303_uses_exact_files_recorded_for_seed202(tmp_path: Path):
    module = load_module()
    result = verify(module, tmp_path, fixture(tmp_path))
    assert result["status"] == "seed303_uses_exact_seed202_model"
    assert result["seed303_fields_read"] is False
    assert result["new_physical_parameter_values_added"] == []


def test_rejects_checkpoint_replaced_after_seed202(tmp_path: Path):
    module = load_module()
    values = fixture(tmp_path)
    values[1].write_bytes(b"different weights")
    with pytest.raises(ValueError, match="checkpoint changed before seed303"):
        verify(module, tmp_path, values)


def test_rejects_model_source_map_rewritten_after_seed202(tmp_path: Path):
    module = load_module()
    values = fixture(tmp_path)
    with values[3].open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="model-source map changed"):
        verify(module, tmp_path, values)


def test_rejects_seed202_result_from_another_checkpoint(tmp_path: Path):
    module = load_module()
    values = fixture(tmp_path)
    payload = json.loads(values[5].read_text(encoding="utf-8"))
    payload["checkpoint_sha256"] = "0" * 64
    values[5].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the one used on seed202"):
        verify(module, tmp_path, values)
