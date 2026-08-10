import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "check_hccb_p418_formal_training_manifest_readiness.py"
)
SPEC = importlib.util.spec_from_file_location("manifest_readiness", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_manifest(path: Path, jobs: list[dict]) -> None:
    path.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")


def test_manifest_accepts_declared_future_input(tmp_path: Path) -> None:
    script = tmp_path / "job.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    data = tmp_path / "data.json"
    data.write_text("{}\n", encoding="utf-8")
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    manifest = tmp_path / "manifest.json"
    write_manifest(
        manifest,
        [
            {
                "job_id": "first",
                "stage": "training",
                "depends_on": [],
                "output_dir": str(first_dir),
                "completion_file": str(first_dir / "summary.json"),
                "command": (
                    f"python3 {script} --data {data} --output-dir {first_dir}"
                ),
            },
            {
                "job_id": "second",
                "stage": "evaluation",
                "depends_on": ["first"],
                "output_dir": str(second_dir),
                "completion_file": str(second_dir / "summary.json"),
                "command": (
                    f"python3 {script} --model-summary {first_dir / 'summary.json'} "
                    f"--output-dir {second_dir}"
                ),
            },
        ],
    )
    result = MODULE.check_manifest(manifest)
    assert result["status"].endswith("passed")
    assert result["errors"] == []
    assert result["deferred_input_path_count"] == 1
    assert result["deferred_inputs"][0]["declared_producer"] == "first"


def test_manifest_rejects_undeclared_future_input(tmp_path: Path) -> None:
    script = tmp_path / "job.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    manifest = tmp_path / "manifest.json"
    write_manifest(
        manifest,
        [
            {
                "job_id": "first",
                "stage": "training",
                "depends_on": [],
                "output_dir": str(first_dir),
                "completion_file": str(first_dir / "summary.json"),
                "command": f"python3 {script} --output-dir {first_dir}",
            },
            {
                "job_id": "second",
                "stage": "evaluation",
                "depends_on": [],
                "output_dir": str(second_dir),
                "completion_file": str(second_dir / "summary.json"),
                "command": (
                    f"python3 {script} --model-summary {first_dir / 'summary.json'} "
                    f"--output-dir {second_dir}"
                ),
            },
        ],
    )
    result = MODULE.check_manifest(manifest)
    assert result["status"].endswith("failed")
    assert "not a declared dependency" in result["errors"][0]


def test_manifest_rejects_missing_script(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    manifest = tmp_path / "manifest.json"
    write_manifest(
        manifest,
        [
            {
                "job_id": "missing",
                "stage": "training",
                "depends_on": [],
                "output_dir": str(output_dir),
                "completion_file": str(output_dir / "summary.json"),
                "command": f"python3 {tmp_path / 'missing.py'} --output-dir {output_dir}",
            }
        ],
    )
    result = MODULE.check_manifest(manifest)
    assert result["status"].endswith("failed")
    assert "command script is missing" in result["errors"][0]
