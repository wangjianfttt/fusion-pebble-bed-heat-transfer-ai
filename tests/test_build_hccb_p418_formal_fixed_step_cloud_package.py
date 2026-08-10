import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_formal_fixed_step_cloud_package.py"
SEQUENCE_RUNNER = ROOT / "scripts/run_hccb_p418_formal_fixed_step_sequence.sh"
ARRAY_RUNNER = ROOT / "scripts/run_hccb_p418_formal_fixed_step_array_n96p.sbatch"
HIGH_RE_SEQUENCE_RUNNER = ROOT / "scripts/run_hccb_p418_high_re_fixed_step_sequence.sh"
HIGH_RE_ARRAY_RUNNER = ROOT / "scripts/run_hccb_p418_high_re_fixed_step_array_n96p.sbatch"


def load_module():
    spec = importlib.util.spec_from_file_location("formal_package", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

def test_validate_links_accepts_internal_relative_link(tmp_path: Path) -> None:
    validate_links = load_module().validate_links
    target = tmp_path / "target.txt"
    target.write_text("ok\n", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to("target.txt")
    rows = validate_links(tmp_path)
    assert rows == [{"path": "link.txt", "target": "target.txt"}]


def test_validate_links_rejects_absolute_link(tmp_path: Path) -> None:
    validate_links = load_module().validate_links
    link = tmp_path / "link.txt"
    link.symlink_to("/etc/hosts")
    try:
        validate_links(tmp_path)
    except ValueError as exc:
        assert "absolute symbolic link" in str(exc)
    else:
        raise AssertionError("absolute link should be rejected")


def test_formal_runners_export_cloud_runtime_settings() -> None:
    sequence_text = SEQUENCE_RUNNER.read_text(encoding="utf-8")
    array_text = ARRAY_RUNNER.read_text(encoding="utf-8")

    assert "export ENDPOINT_READINESS_MODE=transient_endpoint_fields" in sequence_text
    assert "export ENDPOINT_READINESS_MODE=transient_endpoint_fields" in array_text
    assert "export OPENFOAM_BASHRC=${OPENFOAM_BASHRC:?" in sequence_text
    assert "export OPENFOAM_BASHRC=${OPENFOAM_ROOT}/etc/bashrc" in array_text
    assert (
        'bash "${P418_FIXED_STEP_PATCH_ROOT}/code/run_hccb_p418_step_responses.sh"'
        in sequence_text
    )


def test_high_re_profile_and_runners_preserve_independent_test_role() -> None:
    module = load_module()
    profile = module.PACKAGE_PROFILES["high_re6"]
    assert profile["expected_sequences"] == 6
    assert profile["expected_endpoints"] == 5

    sequence_text = HIGH_RE_SEQUENCE_RUNNER.read_text(encoding="utf-8")
    array_text = HIGH_RE_ARRAY_RUNNER.read_text(encoding="utf-8")
    assert "frozen_model_independent_test_only" in sequence_text
    assert '"allowed_for_model_fitting": False' in sequence_text
    assert "hccb_p418_high_re_independent_step_plan.json" in sequence_text
    assert "#SBATCH --array=0-5%6" in array_text
    assert "hccb_p418_high_re_independent_step_plan.json" in array_text
    assert (
        'bash "${P418_FIXED_STEP_PATCH_ROOT}/code/run_hccb_p418_step_responses.sh"'
        in sequence_text
    )
