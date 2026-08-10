import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_fixed_step_runtime_patch.py"


def load_module():
    spec = importlib.util.spec_from_file_location("runtime_patch", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runtime_patch_is_small_complete_and_not_a_submission(tmp_path: Path) -> None:
    module = load_module()
    output = tmp_path / "patch"
    ready = module.build(output, overwrite=False)
    assert ready["formal_solver_submission_authorized_by_this_file"] is False
    for relative in module.FILES:
        assert (output / relative).is_file()
    manifest = (output / "PATCH_MANIFEST.json").read_text(encoding="utf-8")
    assert '"scientific_model_changed": false' in manifest
    assert "absolute OpenFOAM time indices" in manifest
    assert (output / "PATCH_SHA256SUMS").is_file()


def test_patch_wrappers_do_not_fall_back_to_original_runtime_code() -> None:
    common = (ROOT / "code/run_hccb_p418_step_responses.sh").read_text(
        encoding="utf-8"
    )
    high_re = (
        ROOT / "scripts/run_hccb_p418_high_re_fixed_step_sequence.sh"
    ).read_text(encoding="utf-8")
    formal = (
        ROOT / "scripts/run_hccb_p418_formal_fixed_step_sequence.sh"
    ).read_text(encoding="utf-8")

    assert (
        '${P418_FIXED_STEP_PATCH_ROOT}/code/build_hccb_p418_step_response_cases.py'
        in common
    )
    assert (
        '${P418_FIXED_STEP_PATCH_ROOT}/code/add_hccb_transient_temperature_outputs.py'
        in common
    )
    assert (
        '${P418_FIXED_STEP_PATCH_ROOT}/code/import_hccb_p418_parallel_history.py'
        in common
    )
    for wrapper in (high_re, formal):
        assert 'test -f "${P418_FIXED_STEP_PATCH_ROOT}/PATCH_READY.json"' in wrapper
        assert (
            'bash "${P418_FIXED_STEP_PATCH_ROOT}/code/run_hccb_p418_step_responses.sh"'
            in wrapper
        )


def test_fresh_parallel_zero_time_has_a_defined_global_index() -> None:
    common = (ROOT / "code/run_hccb_p418_step_responses.sh").read_text(
        encoding="utf-8"
    )
    assert '${missing_count} -eq ${NP_PER_CASE}' in common
    assert 'value="${time_name}"' in common
    assert "printf '0\\n'" in common
    assert "OpenFOAM time records are incomplete" in common
