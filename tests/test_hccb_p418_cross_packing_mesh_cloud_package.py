from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "code/build_hccb_p418_cross_packing_mesh_cloud_package.sh"
RUNNER = ROOT / "cloud_migration/run_cross_packing_mesh_preprocess.sh"
SLURM = ROOT / "cloud_migration/submit_cross_packing_mesh_array_example.sh"


def test_mesh_cloud_scripts_have_valid_shell_syntax() -> None:
    for script in (BUILDER, RUNNER, SLURM):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_runner_is_mesh_only_and_uses_seed101_settings() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "build_hccb_dense_snappy_case.py" in text
    assert "./Allmesh" in text
    assert "checkMesh -case" in text
    assert "reference_seed101_mesh_manifest.json" in text
    assert "new_physical_parameters" in text
    assert "resource.Allmesh.json" in text
    assert "resource.checkMesh.fluid.json" in text
    assert "peak_observed_aggregate_rss_gb" in text
    for forbidden in ("foamMultiRun", "mpirun", "decomposePar", "reconstructPar"):
        assert forbidden not in text


def test_slurm_example_maps_two_independent_seed_tasks() -> None:
    text = SLURM.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-1" in text
    assert "SEEDS=(202 303)" in text
    assert "--cpus-per-task=1" in text
    assert "--mem=32G" in text
    assert "--time=02:00:00" in text
    assert "RUN_PARENT" in text


def test_package_builder_keeps_inputs_and_no_generated_mesh(tmp_path: Path) -> None:
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
        ["bash", str(BUILDER)], env=env, check=True, capture_output=True, text=True
    )
    package = output / "p418_cross_packing_mesh_preprocess"
    plan = json.loads(
        (package / "parameters/hccb_p418_cross_packing_plan.json").read_text(
            encoding="utf-8"
        )
    )
    for seed in (202, 303):
        packing = package / f"packings/seed{seed}_packing.npz"
        expected = next(
            item["packing_npz_sha256"]
            for item in plan["packing_realisations"]
            if int(item["seed"]) == seed
        )
        assert hashlib.sha256(packing.read_bytes()).hexdigest() == expected
    assert (package / "reference/reference_seed101_mesh_manifest.json").is_file()
    assert (package / "scripts/run_cross_packing_mesh_preprocess.sh").is_file()
    assert (package / "code/build_clipped_hccb_solid_surface_vtk.py").is_file()
    assert (package / "code/run_with_resource_monitor.py").is_file()
    assert (package / "README_CN.md").is_file()
    assert (package / "SHA256SUMS").is_file()
    assert not list(package.rglob("polyMesh"))
    assert not list(package.rglob("processor*"))
    assert not list(package.rglob("log.Allmesh"))


def test_package_builder_rejects_home_output() -> None:
    env = os.environ.copy()
    env.update(
        {
            "ROOT": str(ROOT),
            "OUTPUT_ROOT": "/home/forbidden_cross_packing_package",
            "CREATE_ARCHIVE": "0",
        }
    )
    result = subprocess.run(
        ["bash", str(BUILDER)], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "must not be under /home" in result.stderr


def test_reference_manifest_preserves_completed_mesh_settings() -> None:
    payload = json.loads(
        (ROOT / "cloud_migration/reference_seed101_mesh_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["crop_box_dp"] == [1.234, 5.157, 3.921, 8.163, 2.906, 6.396]
    assert payload["numerical_controls"]["background_cells_per_particle_diameter"] == 10.101010101
    assert payload["numerical_controls"]["sphere_icosphere_subdivisions"] == 3
    assert payload["numerical_controls"]["surface_refinement_level"] == 2
    assert payload["numerical_controls"]["cells_between_levels"] == 2
    assert payload["new_physical_parameters"] == []
