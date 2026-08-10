from __future__ import annotations

import importlib.util
import json
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/package_hccb_p418_processed_data_release.py"


def load_module():
    spec = importlib.util.spec_from_file_location("processed_release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_current_release_refuses_missing_final_outputs(tmp_path: Path) -> None:
    module = load_module()
    with pytest.raises(RuntimeError, match="processed-data release is incomplete"):
        module.build_archive(
            ROOT,
            tmp_path / "preflight",
            tmp_path / "processed.tar.gz",
        )


def test_complete_fixture_builds_deterministic_safe_archive(tmp_path: Path) -> None:
    module = load_module()
    project = tmp_path / "project"
    (project / "manuscript").mkdir(parents=True)
    main = (ROOT / "manuscript/main.tex").read_text(encoding="utf-8")
    (project / "manuscript/main.tex").write_text(main, encoding="utf-8")
    (project / "manuscript/generated_final_abstract.tex").write_text(
        "Final evidence-based abstract for the processed data release.\n",
        encoding="utf-8",
    )
    choice = project / "submission/data_release_license_choice.json"
    choice.parent.mkdir(parents=True)
    choice.write_text(
        json.dumps(
            {"software_license": "MIT", "data_license": "CC-BY-4.0"}
        )
        + "\n",
        encoding="utf-8",
    )
    for index, relative in enumerate(
        (*module.COMPACT_FILES, *module.FINAL_PROCESSED_FILES), start=1
    ):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture-{index}\n".encode("ascii"))

    release = project / "release"
    output = project / "processed.tar.gz"
    first = module.build_archive(project, release, output)
    first_bytes = output.read_bytes()
    second = module.build_archive(project, release, output)
    assert output.read_bytes() == first_bytes
    assert first["archive_sha256"] == second["archive_sha256"]
    assert first["repository_metadata_ready"] is True
    with tarfile.open(output, "r:gz") as tar:
        names = tar.getnames()
        assert len(names) == len(set(names))
        assert names[-1].endswith("/SHA256SUMS.txt")
        assert all(not member.issym() and not member.islnk() for member in tar)
