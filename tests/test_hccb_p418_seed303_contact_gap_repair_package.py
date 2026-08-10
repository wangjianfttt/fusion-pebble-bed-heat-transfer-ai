from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "code/build_hccb_p418_seed303_contact_gap_repair_package.sh"
RUNNER = ROOT / "cloud_migration/run_seed303_contact_gap_mesh_repair.sh"
PREFLIGHT = ROOT / "code/prepare_hccb_p418_seed303_contact_gap_repair.py"
CANDIDATE = ROOT / "parameters/hccb_p418_seed303_contact_gap_repair_candidate.json"


def test_seed303_repair_scripts_have_valid_syntax() -> None:
    for script in (BUILDER, RUNNER):
        subprocess.run(["bash", "-n", str(script)], check=True)
    subprocess.run(
        ["python3", "-m", "py_compile", str(PREFLIGHT)],
        check=True,
    )


def test_seed303_repair_is_disabled_by_default_and_mesh_only() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "EXECUTE=${EXECUTE:-0}" in text
    assert "SEED303_LOCAL_REPAIR_APPROVED=${SEED303_LOCAL_REPAIR_APPROVED:-0}" in text
    assert "if [[ ${EXECUTE} != 1 ]]" in text
    assert "if [[ ${SEED303_LOCAL_REPAIR_APPROVED} != 1 ]]" in text
    assert "--local-refinement-particle-ids 1595 951" in text
    assert "--local-refinement-level" in text
    assert "--local-refinement-region contact-gap" in text
    assert "./Allmesh" in text
    for forbidden in ("foamMultiRun", "mpirun", "decomposePar", "reconstructPar"):
        assert forbidden not in text


def test_candidate_changes_only_the_local_numerical_mesh() -> None:
    payload = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    assert payload["execution_approved"] is False
    assert payload["source_packing_sha256"] == (
        "2737dcadde1b506049aa81f478ca6e0d69f4e0e48eab77c267beebe524578b83"
    )
    assert payload["repair_controls"]["particle_ids"] == [1595, 951]
    assert payload["repair_controls"]["global_surface_refinement_level"] == 2
    assert payload["repair_controls"]["local_refinement_level"] == 3
    assert payload["repair_controls"]["local_refinement_region"] == "contact-gap"
    assert payload["repair_controls"]["local_refinement_padding_cells"] == 4.0
    assert payload["new_physical_parameters"] == []


def test_package_build_runs_only_zero_solver_preflight(tmp_path: Path) -> None:
    output = tmp_path / "output"
    env = os.environ.copy()
    env.update(
        {
            "ROOT": str(ROOT),
            "OUTPUT_ROOT": str(output),
            "CREATE_ARCHIVE": "0",
        }
    )
    subprocess.run(
        ["bash", str(BUILDER)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    package = output / "p418_seed303_contact_gap_repair_candidate"
    status = json.loads((package / "PACKAGE_STATUS.json").read_text(encoding="utf-8"))
    preflight = json.loads(
        (package / "seed303_contact_gap_repair_preflight.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["execution_default"] == 0
    assert status["execution_approved"] is False
    assert preflight["status"].endswith("passed_no_mesh_started")
    assert preflight["mesh_generator_started"] is False
    assert preflight["heat_transfer_solver_started"] is False
    assert preflight["particle_ids"] == [1595, 951]
    assert not list(package.rglob("polyMesh"))
    assert not list(package.rglob("processor*"))
    assert not list(package.rglob("log.Allmesh"))
    checksums = package / "SHA256SUMS"
    assert checksums.is_file()
    packing = package / "packings/seed303_packing.npz"
    assert hashlib.sha256(packing.read_bytes()).hexdigest() == (
        "2737dcadde1b506049aa81f478ca6e0d69f4e0e48eab77c267beebe524578b83"
    )


def test_runner_refuses_execution_without_second_approval_flag(tmp_path: Path) -> None:
    output = tmp_path / "output"
    env = os.environ.copy()
    env.update(
        {
            "ROOT": str(ROOT),
            "OUTPUT_ROOT": str(output),
            "CREATE_ARCHIVE": "0",
        }
    )
    subprocess.run(["bash", str(BUILDER)], env=env, check=True, capture_output=True)
    package = output / "p418_seed303_contact_gap_repair_candidate"
    run_env = os.environ.copy()
    run_env.update(
        {
            "EXECUTE": "1",
            "SEED303_LOCAL_REPAIR_APPROVED": "0",
            "RUN_ROOT": str(tmp_path / "must_not_exist"),
        }
    )
    result = subprocess.run(
        ["bash", str(package / "scripts/run_seed303_contact_gap_mesh_repair.sh")],
        env=run_env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "not approved" in result.stderr
    assert not (tmp_path / "must_not_exist").exists()
