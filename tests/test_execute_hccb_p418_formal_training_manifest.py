from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/execute_hccb_p418_formal_training_manifest.py"


class FormalTrainingManifestExecutorTests(unittest.TestCase):
    def manifest(self, root: Path, *, bad_order: bool = False) -> Path:
        first_output = root / "first.json"
        second_output = root / "second.json"
        first = {
            "job_id": "first",
            "depends_on": [],
            "device": "cpu",
            "completion_file": str(first_output),
            "command": (
                f"{sys.executable} -c "
                + repr(
                    "import json,pathlib;"
                    f"pathlib.Path({str(first_output)!r}).write_text("
                    "json.dumps({'status':'complete'}))"
                )
            ),
        }
        second = {
            "job_id": "second",
            "depends_on": ["first"],
            "device": "cpu",
            "completion_file": str(second_output),
            "command": (
                f"{sys.executable} -c "
                + repr(
                    "import json,pathlib;"
                    f"pathlib.Path({str(second_output)!r}).write_text("
                    "json.dumps({'status':'complete'}))"
                )
            ),
        }
        path = root / "manifest.json"
        path.write_text(
            json.dumps({"jobs": [second, first] if bad_order else [first, second]}),
            encoding="utf-8",
        )
        return path

    def command(self, root: Path, manifest: Path, *extra: str) -> list[str]:
        return [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--root",
            str(root),
            "--state-file",
            str(root / "state.json"),
            "--log-dir",
            str(root / "logs"),
            "--lock-file",
            str(root / "lock"),
            *extra,
        ]

    def test_executes_dependencies_and_writes_completion_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = subprocess.run(
                self.command(root, self.manifest(root), "--execute"),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state = json.loads((root / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "completed_p418_formal_model_chain")
            self.assertEqual(
                state["completed_job_ids_this_run"], ["first", "second"]
            )

    def test_existing_completion_is_retained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.manifest(root)
            (root / "first.json").write_text('{"status":"existing"}', encoding="utf-8")
            result = subprocess.run(
                self.command(root, manifest, "--execute"),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state = json.loads((root / "state.json").read_text(encoding="utf-8"))
            self.assertIn("first", state["existing_job_ids_retained"])
            self.assertEqual(state["completed_job_ids_this_run"], ["second"])

    def test_failed_status_is_not_treated_as_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.manifest(root)
            (root / "first.json").write_text(
                '{"status":"failed_model_training"}', encoding="utf-8"
            )
            result = subprocess.run(
                self.command(root, manifest, "--execute"),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            first = json.loads((root / "first.json").read_text(encoding="utf-8"))
            self.assertEqual(first["status"], "complete")

    def test_rejects_manifest_with_dependency_after_consumer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = subprocess.run(
                self.command(root, self.manifest(root, bad_order=True)),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("appears before its dependencies", result.stderr)


if __name__ == "__main__":
    unittest.main()
