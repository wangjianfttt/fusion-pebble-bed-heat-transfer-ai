from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/finalize_hccb_p418_cross_packing_preprocess.py"


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_stage(tmp_path: Path, solid_ok: bool) -> tuple[Path, str]:
    stage = tmp_path / "seed.building"
    stage.mkdir()
    packing_sha = hashlib.sha256(b"packing").hexdigest()
    write_json(
        stage / "case_manifest.json",
        {"source_packing_sha256": packing_sha, "new_physical_parameters": []},
    )
    write_json(
        stage / "mesh_check_summary.json",
        {
            "cell_volume_porosity": 0.39,
            "triangulated_porosity": 0.40,
            "fluid": {
                "failed_check_count": 3,
                "maximum_non_orthogonality_deg": 61.0,
                "maximum_skewness": 3.0,
                "small_determinant_cells": 10,
                "concave_cells": 20,
            },
            "solid": {
                "failed_check_count": 3 if solid_ok else 4,
                "maximum_non_orthogonality_deg": 64.0,
                "maximum_skewness": 3.8 if solid_ok else 4.02,
                "small_determinant_cells": 5,
                "concave_cells": 15,
            },
            "checks": {"fluid_is_one_connected_region": True},
        },
    )
    for name in ("Allmesh", "checkMesh.fluid", "checkMesh.solid"):
        write_json(stage / f"resource.{name}.json", {"return_code": 0})
    (stage / "log.checkMesh.fluid").write_text(
        "Failed 3 mesh checks.\n", encoding="utf-8"
    )
    (stage / "log.checkMesh.solid").write_text(
        "Failed 3 mesh checks.\n", encoding="utf-8"
    )
    (stage / "log.checkMesh.fluid.basic_diagnostic_20260725").write_text(
        "Mesh OK.\n", encoding="utf-8"
    )
    (stage / "log.checkMesh.solid.basic_diagnostic_20260725").write_text(
        "Mesh OK.\n" if solid_ok else "Failed 1 mesh checks.\n",
        encoding="utf-8",
    )
    return stage, packing_sha


def test_promotes_basic_pass_and_retains_strict_diagnostics(tmp_path: Path) -> None:
    stage, packing_sha = make_stage(tmp_path, solid_ok=True)
    run_root = tmp_path / "seed202"
    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--stage",
            str(stage),
            "--run-root",
            str(run_root),
            "--seed",
            "202",
            "--expected-packing-sha256",
            packing_sha,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert run_root.is_dir()
    assert not stage.exists()
    payload = json.loads(
        (run_root / "cloud_mesh_completion.json").read_text(encoding="utf-8")
    )
    assert payload["basic_check"]["fluid"]["mesh_ok"] is True
    assert payload["basic_check"]["solid"]["mesh_ok"] is True
    assert payload["strict_check"]["fluid"]["failed_check_count"] == 3
    assert payload["strict_diagnostics_retained"] is True
    assert payload["heat_transfer_solver_started"] is False


def test_preserves_failed_stage_without_relaxing_threshold(tmp_path: Path) -> None:
    stage, packing_sha = make_stage(tmp_path, solid_ok=False)
    run_root = tmp_path / "seed303"
    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--stage",
            str(stage),
            "--run-root",
            str(run_root),
            "--seed",
            "303",
            "--expected-packing-sha256",
            packing_sha,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3
    assert stage.is_dir()
    assert not run_root.exists()
    payload = json.loads(
        (stage / "mesh_preprocess_failure.json").read_text(encoding="utf-8")
    )
    assert payload["basic_check"]["solid"]["mesh_ok"] is False
    assert payload["strict_check"]["solid"]["maximum_skewness"] == 4.02
    assert payload["mesh_parameters_changed"] is False
    assert payload["physical_parameters_changed"] is False
