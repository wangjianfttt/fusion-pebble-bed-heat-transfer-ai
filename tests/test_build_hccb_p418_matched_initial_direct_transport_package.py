import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_direct_transport_cloud_package_is_complete() -> None:
    source = (
        ROOT
        / "cloud_build"
        / "p418_matched_initial_direct_transport_smoke_20260809"
    )
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "package"
        output.mkdir()
        for name in (
            "compile_prepare_direct_no_solver.sbatch",
            "run_direct_matched_initial_smoke.sbatch",
            "submit_direct_matched_initial_smoke.sh",
            "smoke_plan.json",
            "README_CN.md",
        ):
            (output / name).write_bytes((source / name).read_bytes())

        subprocess.run(
            [
                "python3",
                str(ROOT / "code/build_hccb_p418_matched_initial_direct_transport_package.py"),
                "--output",
                str(output),
            ],
            check=True,
        )

        manifest = json.loads((output / "PACKAGE_MANIFEST.json").read_text())
        ready = json.loads((output / "READY").read_text())
        plan = json.loads((output / "smoke_plan.json").read_text())
        assert manifest["status"].endswith("no_solver_started")
        assert manifest["physical_correlations_changed"] is False
        assert manifest["openfoam_solver_started"] is False
        assert manifest["post_solver_observable_export_included"] is True
        assert manifest["observable_signal_count"] == 15
        assert ready["solver_approved"] is False
        assert plan["finite_pressure_table_used"] is False
        assert plan["submission_requires_exact_phrase"] == "批准短算"

        for line in (output / "PACKAGE_SHA256SUMS").read_text().splitlines():
            digest, relative = line.split("  ", 1)
            assert sha256(output / relative) == digest

        submit = (output / "submit_direct_matched_initial_smoke.sh").read_text()
        assert "APPROVAL_PHRASE" in submit
        assert "批准短算" in submit
        assert (output / "code/export_hccb_p418_transient_observables.py").is_file()
        assert (output / "code/export_hccb_p418_matched_initial_short_observables.py").is_file()
        runner = (output / "run_direct_matched_initial_smoke.sbatch").read_text()
        assert "export_hccb_p418_matched_initial_short_observables.py" in runner
        assert "observable_export_status" in runner
        preflight = (output / "compile_prepare_direct_no_solver.sbatch").read_text()
        assert 'name            "0";' in preflight
        assert 'name            0;' not in preflight
        assert "foamDictionary" in preflight
        assert "thermoType/transport" in preflight
        physical = (
            output
            / "solver_extensions"
            / "hccbHeliumTransport"
            / "physicalProperties.example"
        ).read_text()
        assert "FoamFile" in physical
        assert "class       dictionary;" in physical
        assert "object      physicalProperties;" in physical
        assert "transport       hccbHelium;" in physical
