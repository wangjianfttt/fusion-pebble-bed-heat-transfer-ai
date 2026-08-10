from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "code/handoff_hccb_p418_training_manifest.py"
SPEC = importlib.util.spec_from_file_location("handoff_training_manifest", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_revised_manifest_requires_cuda_for_every_physics_job(tmp_path: Path) -> None:
    good = {
        "jobs": [
            {
                "job_id": "physics",
                "command": "train --physics-mode energy_and_flux --physics-device cuda",
            },
            {"job_id": "data", "command": "train --physics-mode data_only"},
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    payload = MODULE.validate_revised_manifest(path, MODULE.sha256(path))
    assert len(payload["jobs"]) == 2

    good["jobs"][0]["command"] = "train --physics-mode energy_and_flux"
    path.write_text(json.dumps(good), encoding="utf-8")
    try:
        MODULE.validate_revised_manifest(path, MODULE.sha256(path))
    except ValueError as error:
        assert "without explicit CUDA" in str(error)
    else:
        raise AssertionError("a physics job without CUDA was accepted")


def test_completion_rejects_running_or_failed_status(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    for status in ("running", "failed_training"):
        path.write_text(json.dumps({"status": status}), encoding="utf-8")
        assert not MODULE.valid_completion(path)
    path.write_text(json.dumps({"status": "completed_model"}), encoding="utf-8")
    assert MODULE.valid_completion(path)


def test_zombie_process_no_longer_holds_execution_resources() -> None:
    assert MODULE.process_holds_execution_resources("R")
    assert MODULE.process_holds_execution_resources("S")
    assert MODULE.process_holds_execution_resources("T")
    assert not MODULE.process_holds_execution_resources("Z")
    assert not MODULE.process_holds_execution_resources(None)
