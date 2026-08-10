from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "code/build_hccb_p418_openfoam13_cloud_package.sh"


def make_fake_source(root: Path) -> Path:
    source = root / "source_case"
    for relative in ("0/fluid", "0/solid", "constant/fluid/polyMesh", "constant/solid/polyMesh", "system"):
        (source / relative).mkdir(parents=True, exist_ok=True)
    (source / "0/fluid/T").write_text("T", encoding="utf-8")
    (source / "0/solid/T").write_text("T", encoding="utf-8")
    (source / "constant/fluid/polyMesh/points").write_text("mesh", encoding="utf-8")
    (source / "constant/solid/polyMesh/points").write_text("mesh", encoding="utf-8")
    (source / "system/controlDict").write_text("control", encoding="utf-8")
    (source / "cht_smoke_metadata.json").write_text("{}", encoding="utf-8")
    (source / "formal_sample_complete.json").write_text("{}", encoding="utf-8")
    (source / "cht_result_summary_200.json").write_text("{}", encoding="utf-8")
    (source / "processor0").mkdir()
    (source / "200").mkdir()
    (source / "training_sample_200_schema3").mkdir()
    return source


def test_builder_keeps_only_portable_case_inputs(tmp_path: Path) -> None:
    source = make_fake_source(tmp_path)
    output = tmp_path / "output"
    env = os.environ.copy()
    env.update(
        {
            "ROOT": str(ROOT),
            "SOURCE_CASE": str(source),
            "OUTPUT_ROOT": str(output),
            "CREATE_ARCHIVE": "0",
        }
    )
    subprocess.run(["bash", str(BUILDER)], env=env, check=True, capture_output=True, text=True)
    package = output / "p418_openfoam13_minimal_case"
    assert (package / "case_template/0/fluid/T").is_file()
    assert (package / "case_template/constant/fluid/polyMesh/points").is_file()
    assert (package / "case_template/cht_smoke_metadata.json").is_file()
    assert (package / "scripts/run_openfoam13_formal.sh").is_file()
    assert (package / "scripts/postprocess_openfoam13_case.sh").is_file()
    assert (package / "scripts/compare_hccb_p418_cloud_reference.py").is_file()
    assert (package / "scripts/run_with_resource_monitor.py").is_file()
    assert (package / "scripts/summarize_openfoam_cloud_resources.py").is_file()
    assert (package / "parameters/literature_parameter_manifest.csv").is_file()
    assert (package / "cloud_case_matrix.csv").is_file()
    assert (package / "cloud_case_matrix_summary.json").is_file()
    assert (package / "pending_case_ids.txt").is_file()
    assert (package / "SHA256SUMS").is_file()
    assert not list(package.rglob("processor*"))
    assert not list(package.rglob("training_sample_200_schema3"))
    assert not list(package.rglob("200"))


def test_builder_rejects_home_output(tmp_path: Path) -> None:
    source = make_fake_source(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "ROOT": str(ROOT),
            "SOURCE_CASE": str(source),
            "OUTPUT_ROOT": "/home/forbidden_p418_package",
            "CREATE_ARCHIVE": "0",
        }
    )
    result = subprocess.run(
        ["bash", str(BUILDER)], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "must not be under /home" in result.stderr


def test_cloud_scripts_have_valid_shell_syntax() -> None:
    for script in (
        BUILDER,
        ROOT / "cloud_migration/run_openfoam13_case.sh",
        ROOT / "cloud_migration/run_openfoam13_smoke.sh",
        ROOT / "cloud_migration/run_openfoam13_formal.sh",
        ROOT / "cloud_migration/run_openfoam13_batch_case.sh",
        ROOT / "cloud_migration/postprocess_openfoam13_case.sh",
        ROOT / "cloud_migration/submit_slurm_example.sh",
        ROOT / "cloud_migration/submit_slurm_array_example.sh",
    ):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_dependency_record_matches_declared_solver_route() -> None:
    text = (ROOT / "cloud_migration/VERSION_DEPENDENCIES_CN.md").read_text(
        encoding="utf-8"
    )
    assert "OpenFOAM Foundation 13" in text
    assert "foamMultiRun" in text
    assert "Open MPI 4.1.2" in text
    assert "GCC 11.4.0" in text
    assert "Python 3.10.12" in text


def test_cloud_run_checks_every_partition_at_the_target_time() -> None:
    text = (ROOT / "cloud_migration/run_openfoam13_case.sh").read_text(
        encoding="utf-8"
    )
    assert "end_time=$(foamDictionary" in text
    assert "for ((rank = 0; rank < NP; rank++))" in text
    assert "processor${rank}/${end_time}/${field}" in text
    assert "CASE_TEMPLATE" in text
    assert "SHARED_MESH_ROOT" in text
    assert "floating point exception [(]core dumped[)]" in text
    assert "|floating point exception|" not in text
    assert "postprocess_openfoam13_case.sh" in text
    assert "resource.foamMultiRun.json" in text
    assert "cloud_runtime_resources.json" in text
    assert "run_with_resource_monitor.py" in text
    assert '-entry numberOfSubdomains -set "${NP}"' in text
    assert 'mpi_command=(mpirun --bind-to none -np "${NP}")' in text
    assert 'taskset -c "${MPI_CPU_LIST}"' in text
    assert 'POSTPROCESS_SCRIPT=${POSTPROCESS_SCRIPT:-' in text


def test_formal_cloud_run_uses_workstation_reference_summary() -> None:
    text = (ROOT / "cloud_migration/run_openfoam13_formal.sh").read_text(
        encoding="utf-8"
    )
    assert "source_record/cht_result_summary_200.json" in text
    assert "REFERENCE_SUMMARY" in text
