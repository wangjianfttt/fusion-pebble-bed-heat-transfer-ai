from pathlib import Path
import json
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
GENERIC = ROOT / "code" / "run_hccb_dense_cht_p418_matrix_parallel.sh"
RUNNER = ROOT / "code" / "run_hccb_p418_cross_packing_matrix.sh"
VERIFY = ROOT / "code" / "verify_hccb_p418_cross_packing_matrix.py"


def test_generic_runner_allows_packing_specific_interface_pairs():
    text = GENERIC.read_text(encoding="utf-8")
    assert "INTERFACE_PAIRS=${INTERFACE_PAIRS:-" in text


def test_cross_packing_runner_uses_own_interface_and_postprocesses_same_seed():
    text = RUNNER.read_text(encoding="utf-8")
    assert 'SEED=${SEED:-202}' in text
    assert 'EXECUTE=${EXECUTE:-0}' in text
    assert 'SEED must be 202 or 303' in text
    assert 'CASE="${reference_case}"' in text
    assert 'INTERFACE_PAIRS=${reference_case}/interface_pairs/interface_face_pairs.npz' in text
    assert 'INTERFACE_PAIRS="${INTERFACE_PAIRS}"' in text
    assert 'SEED="${SEED}" EXECUTE=1' in text
    assert "condition_count\": 9" in text
    assert "verify_hccb_p418_cross_packing_matrix.py" in text


def test_cross_packing_runner_dry_run_starts_nothing(tmp_path: Path):
    result = subprocess.run(
        ["bash", str(RUNNER)],
        env={"ROOT": str(tmp_path), "SEED": "303", "EXECUTE": "0"},
        check=True,
        capture_output=True,
        text=True,
    )
    assert "dry run only" in result.stdout
    assert list(tmp_path.iterdir()) == []


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_matrix(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    condition_ids = [
        "u0p05_T300_q4p85",
        "u0p05_T300_q8p85",
        "u0p05_T900_q4p85",
        "u0p05_T900_q8p85",
        "u0p25_T300_q4p85",
        "u0p25_T300_q8p85",
        "u0p25_T900_q4p85",
        "u0p25_T900_q8p85",
        "u0p15_T700_q6p85",
    ]
    values = {
        identifier: {
            "condition_id": identifier,
            "inlet_velocity_m_s": float(identifier[1:5].replace("p", ".")),
            "inlet_temperature_K": float(identifier.split("_T")[1].split("_")[0]),
            "solid_heat_source_MW_m3": float(
                identifier.split("_q")[1].replace("p", ".")
            ),
        }
        for identifier in condition_ids
    }
    packing_hash = "abc123"
    plan = tmp_path / "plan.json"
    matrix_root = tmp_path / "matrix"
    matrix_manifest = matrix_root / "matrix_manifest.json"
    mesh_manifest = tmp_path / "mesh_manifest.json"
    write_json(
        plan,
        {
            "screening_design": {"conditions": list(values.values())},
            "packing_realisations": [
                {"seed": 202, "packing_npz_sha256": packing_hash}
            ],
        },
    )
    records = []
    for identifier, record in values.items():
        (matrix_root / identifier).mkdir(parents=True)
        write_json(
            matrix_root / identifier / "cht_smoke_metadata.json",
            {"mesh_source_packing_sha256": packing_hash},
        )
        records.append({**record, "mesh_source_packing_sha256": packing_hash})
    write_json(
        matrix_manifest,
        {"mode": "selected", "selected_case_count": 9, "cases": records},
    )
    write_json(mesh_manifest, {"source_packing_sha256": packing_hash})
    return plan, matrix_root, matrix_manifest, mesh_manifest


def run_verifier(
    plan: Path, matrix_root: Path, matrix_manifest: Path, mesh_manifest: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VERIFY),
            "--seed",
            "202",
            "--plan",
            str(plan),
            "--matrix-manifest",
            str(matrix_manifest),
            "--matrix-root",
            str(matrix_root),
            "--mesh-manifest",
            str(mesh_manifest),
        ],
        capture_output=True,
        text=True,
    )


def test_matrix_verifier_accepts_exact_nine_condition_plan(tmp_path: Path):
    result = run_verifier(*make_matrix(tmp_path))
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["condition_count"] == 9
    assert payload["packing_seed"] == 202


def test_matrix_verifier_rejects_replaced_condition(tmp_path: Path):
    plan, matrix_root, matrix_manifest, mesh_manifest = make_matrix(tmp_path)
    missing = matrix_root / "u0p05_T300_q4p85"
    shutil.rmtree(missing)
    (matrix_root / "u0p10_T500_q6p85").mkdir()
    result = run_verifier(plan, matrix_root, matrix_manifest, mesh_manifest)
    assert result.returncode != 0
    assert "case directories differ" in result.stderr
