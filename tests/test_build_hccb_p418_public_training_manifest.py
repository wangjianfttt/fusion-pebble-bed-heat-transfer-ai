from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_public_training_manifest.py"


def load_module():
    spec = importlib.util.spec_from_file_location("public_training_manifest", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_public_manifest_preserves_jobs_and_removes_machine_paths(tmp_path: Path) -> None:
    module = load_module()
    source_root = "/" + "data2/CodexWork/fusion_pebble_heat_ai"
    source = tmp_path / "formal.json"
    source.write_text(
        json.dumps(
            {
                "status": "running",
                "source_runner": f"{source_root}/code/run_hccb_p418_step_responses.sh",
                "job_count": 1,
                "completed_job_count": 0,
                "remaining_job_count": 1,
                "jobs": [
                    {
                        "job_id": "model_a",
                        "output_dir": f"{source_root}/results/model_a",
                        "completion_file": f"{source_root}/results/model_a/summary.json",
                        "command": f"python3 {source_root}/code/train.py --seed 17",
                        "seed": 17,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "public.json"
    payload = module.build(source, output, "/different/local/project")
    stored = output.read_text(encoding="utf-8")
    assert source_root not in stored
    assert "${PROJECT_ROOT}/code/train.py --seed 17" in stored
    assert payload["jobs"][0]["seed"] == 17
    assert len(payload["source_manifest_sha256"]) == 64


def test_public_manifest_rejects_unrelated_private_path(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "formal.json"
    private_path = "/" + "Users/private/file"
    source.write_text(
        json.dumps(
            {"job_count": 1, "jobs": [{"command": f"cat {private_path}"}]}
        ),
        encoding="utf-8",
    )
    try:
        module.build(source, tmp_path / "public.json", "/" + "data2/project")
    except ValueError as exc:
        assert "private machine path" in str(exc)
    else:
        raise AssertionError("private path was not rejected")


def test_public_manifest_refreshes_completion_snapshot(tmp_path: Path) -> None:
    module = load_module()
    machine_root = "/" + "data2/CodexWork/fusion_pebble_heat_ai"
    local_root = tmp_path / "local_project"
    completion = local_root / "results/model_a/summary.json"
    completion.parent.mkdir(parents=True)
    completion.write_text("{}\n", encoding="utf-8")
    source = tmp_path / "formal.json"
    source.write_text(
        json.dumps(
            {
                "source_runner": f"{machine_root}/code/run_hccb_p418_step_responses.sh",
                "job_count": 2,
                "completed_job_count": 0,
                "remaining_job_count": 2,
                "completed_job_ids": [],
                "jobs": [
                    {
                        "job_id": "model_a",
                        "completion_file": f"{machine_root}/results/model_a/summary.json",
                    },
                    {
                        "job_id": "model_b",
                        "completion_file": f"{machine_root}/results/model_b/summary.json",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = module.build(source, tmp_path / "public.json", str(local_root))

    assert payload["completed_job_ids"] == ["model_a"]
    assert payload["completed_job_count"] == 1
    assert payload["remaining_job_count"] == 1
    assert "${PROJECT_ROOT}" in payload["completion_state_basis"]


def test_public_manifest_keeps_registered_remote_completion(tmp_path: Path) -> None:
    module = load_module()
    machine_root = "/" + "data2/CodexWork/fusion_pebble_heat_ai"
    source = tmp_path / "formal.json"
    source.write_text(
        json.dumps(
            {
                "source_runner": f"{machine_root}/code/run_hccb_p418_step_responses.sh",
                "job_count": 1,
                "completed_job_count": 1,
                "remaining_job_count": 0,
                "completed_job_ids": ["remote_only_model"],
                "jobs": [
                    {
                        "job_id": "remote_only_model",
                        "completion_file": (
                            f"{machine_root}/results/remote_only_model/summary.json"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = module.build(source, tmp_path / "public.json", str(tmp_path))

    assert payload["completed_job_ids"] == ["remote_only_model"]
    assert payload["completed_job_count"] == 1
    assert payload["remaining_job_count"] == 0
