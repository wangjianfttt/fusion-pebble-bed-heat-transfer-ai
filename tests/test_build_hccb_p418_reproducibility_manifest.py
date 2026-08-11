from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_p418_reproducibility_manifest import (
    OPTIONAL_FINAL_OUTPUTS,
    REQUIRED_FINAL_JSON_STATUS,
    REQUIRED_FILES,
    build_manifest,
    collect_source_paths,
    sha256,
    write_outputs,
)


FIGURE_ONE_DATA = {
    "data/apd006_hccb_source_sequence_target_packings/seed101_s80_xlo_ycentre/packing.npz",
    "runs/hccb_dense_snappy_g2_nativezone_r2/geometry/packing_crop.npz",
    "results/hccb_p418_actual_spatiotemporal_operator_37time_gpu_data_only/regional_sequence_geometry.npz",
    "results/hccb_p418_60_sourceflow_r3_model_geometry/model_geometry.npz",
}

REPOSITORY_METADATA = {
    "CITATION.cff",
    "LICENSE",
    "DATA_LICENSE.md",
    "submission/data_release_license_choice.json",
    "submission/data_release_repository_record.json",
    "reproducibility/repository_release_metadata_draft.json",
}

RESULT_TRACEABILITY = {
    "manuscript/result_source_map.csv",
    "results/hccb_p418_fixed_flow_runtime_recovery_checks/summary.json",
}


def test_figure_one_geometry_is_part_of_reproducibility_source() -> None:
    assert FIGURE_ONE_DATA.issubset(REQUIRED_FILES)


def test_repository_citation_metadata_is_required_public_source() -> None:
    assert REPOSITORY_METADATA.issubset(REQUIRED_FILES)


def test_result_traceability_files_are_required_public_source() -> None:
    assert RESULT_TRACEABILITY.issubset(REQUIRED_FILES)


def test_hash_reopens_after_a_transient_read_error(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("reproducible\n", encoding="utf-8")
    original_open = Path.open
    calls = 0

    def flaky_open(path: Path, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("temporary cloud-storage timeout")
        return original_open(path, *args, **kwargs)

    with patch.object(Path, "open", flaky_open):
        digest = sha256(source)

    assert calls == 2
    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()


def write_required_files(root: Path) -> None:
    for relative in REQUIRED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture for {relative}\n", encoding="utf-8")


def test_missing_required_files_are_reported(tmp_path: Path) -> None:
    payload = build_manifest(tmp_path)
    assert payload["source_package_ready"] is False
    assert payload["missing_required_files"]
    assert payload["formal_solver_started_by_reproduction_script"] is False


def test_complete_source_package_has_hashes_and_keeps_final_outputs_separate(
    tmp_path: Path,
) -> None:
    write_required_files(tmp_path)
    payload = build_manifest(tmp_path)
    assert payload["source_package_ready"] is True
    assert payload["final_outputs_ready"] is False
    assert payload["status"] == "p418_reproducibility_source_ready_final_outputs_pending"
    readme = next(
        row
        for row in payload["files"]
        if row["path"] == "reproducibility/README.md"
    )
    expected = hashlib.sha256(
        (tmp_path / "reproducibility/README.md").read_bytes()
    ).hexdigest()
    assert readme["sha256"] == expected
    assert len(readme["sha256"]) == 64
    gci = next(
        row
        for row in payload["files"]
        if row["path"]
        == "results/hccb_p418_three_mesh_cht_sensitivity/mesh_gci.csv"
    )
    assert gci["category"] == "small_processed_results"
    assert gci["required"] is True


def test_solver_extension_sources_are_included(tmp_path: Path) -> None:
    write_required_files(tmp_path)
    extension = (
        tmp_path
        / "solver_extensions"
        / "hccbHeliumTransport"
        / "hccbHeliumTransport.C"
    )
    extension.parent.mkdir(parents=True, exist_ok=True)
    extension.write_text("// direct helium transport model\n", encoding="utf-8")
    payload = build_manifest(tmp_path)
    row = next(
        row
        for row in payload["files"]
        if row["path"]
        == "solver_extensions/hccbHeliumTransport/hccbHeliumTransport.C"
    )
    assert row["category"] == "programs"
    assert row["sha256"] == hashlib.sha256(extension.read_bytes()).hexdigest()


def test_local_python_dependencies_are_included(tmp_path: Path) -> None:
    write_required_files(tmp_path)
    selected = tmp_path / "code" / "example_hccb_p418.py"
    selected.write_text("from shared_openfoam_helper import value\n", encoding="utf-8")
    helper = tmp_path / "code" / "shared_openfoam_helper.py"
    helper.write_text("value = 1\n", encoding="utf-8")
    paths, missing = collect_source_paths(tmp_path)
    assert not missing
    assert selected in paths
    assert helper in paths


def test_machine_local_pilot_smoke_split_is_not_public_source(tmp_path: Path) -> None:
    write_required_files(tmp_path)
    smoke = tmp_path / "parameters/hccb_p418_pilot_smoke_splits.json"
    smoke.write_text(
        '{"status":"software_smoke_split_only",'
        '"source_sample":"/private/machine/sample.npz"}\n',
        encoding="utf-8",
    )
    paths, missing = collect_source_paths(tmp_path)
    assert not missing
    assert smoke not in paths


def test_outputs_include_csv_json_and_plain_chinese_summary(tmp_path: Path) -> None:
    write_required_files(tmp_path)
    payload = build_manifest(tmp_path)
    output = tmp_path / "results" / "manifest"
    write_outputs(payload, output)
    assert (output / "manifest.json").is_file()
    assert (output / "manifest.csv").is_file()
    text = (output / "P418_复现文件说明_CN.md").read_text(encoding="utf-8")
    assert "复现脚本默认不会启动OpenFOAM或模型训练" in text


def test_thermophysical_parameter_manifest_is_required(tmp_path: Path) -> None:
    write_required_files(tmp_path)
    parameter_manifest = tmp_path / "parameters/literature_parameter_manifest.csv"
    assert parameter_manifest.is_file()
    parameter_manifest.unlink()
    _, missing = collect_source_paths(tmp_path)
    assert "parameters/literature_parameter_manifest.csv" in missing


def test_final_json_requires_completed_status(tmp_path: Path) -> None:
    write_required_files(tmp_path)
    for relative in OPTIONAL_FINAL_OUTPUTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative in REQUIRED_FINAL_JSON_STATUS:
            path.write_text('{"status": "incomplete"}\n', encoding="utf-8")
        else:
            path.write_text("result\n", encoding="utf-8")
    payload = build_manifest(tmp_path)
    assert payload["final_outputs_ready"] is False
    checked = {
        row["path"]: row
        for row in payload["optional_final_outputs"]
        if row["path"] in REQUIRED_FINAL_JSON_STATUS
    }
    assert checked
    assert all(row["present"] is False for row in checked.values())


def test_final_json_with_completed_status_is_accepted(tmp_path: Path) -> None:
    write_required_files(tmp_path)
    for relative in OPTIONAL_FINAL_OUTPUTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative in REQUIRED_FINAL_JSON_STATUS:
            path.write_text(
                '{"status": "'
                + REQUIRED_FINAL_JSON_STATUS[relative]
                + '"}\n',
                encoding="utf-8",
            )
        else:
            path.write_text("result\n", encoding="utf-8")
    payload = build_manifest(tmp_path)
    assert payload["final_outputs_ready"] is True


def test_training_coverage_progress_is_included_when_available(tmp_path: Path) -> None:
    write_required_files(tmp_path)
    coverage = (
        tmp_path
        / "results"
        / "hccb_p418_training_data_coverage_partial"
        / "summary.json"
    )
    coverage.parent.mkdir(parents=True, exist_ok=True)
    coverage.write_text(
        '{"completed_case_count": 40, "expected_case_count": 60, '
        '"missing_condition_ids": ["case_a"], "problems": [], '
        '"solver_time_semantics": "steady_iteration_index", '
        '"physical_time_s": null, '
        '"steady_iteration_column": "steady_iteration"}\n',
        encoding="utf-8",
    )
    payload = build_manifest(tmp_path)
    assert payload["training_data_progress"]["completed_case_count"] == 40
    assert (
        payload["training_data_progress"]["solver_time_semantics"]
        == "steady_iteration_index"
    )
    assert payload["training_data_progress"]["physical_time_s"] is None
    assert (
        payload["training_data_progress"]["steady_iteration_column"]
        == "steady_iteration"
    )
    output = tmp_path / "results" / "manifest"
    write_outputs(payload, output)
    text = (output / "P418_复现文件说明_CN.md").read_text(encoding="utf-8")
    assert "40/60" in text
    assert "剩余1组" in text
    assert "不是物理时间" in text
    assert "`steady_iteration`" in text
