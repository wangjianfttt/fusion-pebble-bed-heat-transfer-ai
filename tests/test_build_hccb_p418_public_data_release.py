from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_public_data_release.py"


def load_module():
    spec = importlib.util.spec_from_file_location("public_data_release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_public_data_release_preflight_is_path_free(tmp_path: Path) -> None:
    module = load_module()
    output = tmp_path / "release"
    payload = module.build(ROOT, output)
    assert payload["compact_plot_data_ready"] is True
    assert payload["repository_doi"] == "pending_assignment"
    assert payload["repository_url"] == (
        "https://github.com/wangjianfttt/fusion-pebble-bed-heat-transfer-ai"
    )
    assert payload["repository_metadata_ready"] is False
    assert payload["software_and_data_license"] == {
        "software_license": "MIT",
        "data_license": "cc-by-4.0",
    }
    assert payload["old_tritium_release_doi_reused"] is False
    assert payload["new_physical_parameters"] == []
    assert all(row["present"] for row in payload["compact_files"])
    source_layer = next(
        row for row in payload["release_layers"] if row["name"] == "small_source_archive"
    )
    assert source_layer["location"] == module.SOURCE_ARCHIVE
    assert source_layer["checksum_record"] == module.SOURCE_ARCHIVE_RECORD
    archive_record = json.loads(
        (ROOT / module.SOURCE_ARCHIVE_RECORD).read_text(encoding="utf-8")
    )
    assert archive_record["status"] == "p418_reproducibility_source_archive_ready"
    assert archive_record["archive_size_bytes"] > 0
    assert len(archive_record["archive_sha256"]) == 64
    stored = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    text = json.dumps(stored, ensure_ascii=False)
    for token in module.PRIVATE_TEXT:
        assert token not in text
    metadata = json.loads(
        (output / "zenodo_metadata_draft.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "p418_repository_metadata_draft"
    assert metadata["ready_for_deposition"] is False
    assert metadata["metadata"]["upload_type"] == "dataset"
    description = metadata["metadata"]["description"]
    assert "46,089 nodes" in description
    assert "245,848 connections" in description
    assert "56 retained field times" in description
    assert len(metadata["metadata"]["creators"]) == 5
    assert len(metadata["metadata"]["keywords"]) == 6
    assert "final_processed_files" in metadata["pending_fields"]
    assert "final_abstract" in metadata["pending_fields"]
    assert "author_selected_software_and_data_licenses" not in metadata[
        "pending_fields"
    ]
    assert metadata["license_choice"] == {
        "software_license": "MIT",
        "data_license": "cc-by-4.0",
    }
    assert metadata["metadata"]["license"] == "cc-by-4.0"
    assert metadata["metadata"]["related_identifiers"] == [
        {
            "identifier": (
                "https://github.com/wangjianfttt/"
                "fusion-pebble-bed-heat-transfer-ai"
            ),
            "relation": "isSupplementTo",
            "scheme": "url",
        }
    ]
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "## Release layers" in readme
    assert "## Reproduce the compact quantitative figures" in readme
    assert "## Scientific scope" in readme
    assert "fixed-hydrodynamics thermal-step database" in readme
    assert "not evidence of a successful fully coupled" in readme
    assert "MIT for original software; CC BY 4.0 for processed data" in readme
    assert "SHA-256 values in `summary.json`" in readme
    for token in module.PRIVATE_TEXT:
        assert token not in readme
    selected_rows = [
        row
        for row in payload["final_processed_files"]
        if row["path"] == "validation-selected test prediction"
    ]
    assert selected_rows == [
        {
            "path": "validation-selected test prediction",
            "present": False,
            "reason": "final validation-based field-model selection is pending",
        }
    ]
    assert not any(
        "bounded_data_only_pair_disjoint_stress_test" in row["path"]
        for row in payload["final_processed_files"]
    )


def test_selected_prediction_must_remain_inside_project(tmp_path: Path) -> None:
    module = load_module()
    figures = tmp_path / "figures"
    figures.mkdir()
    outside = tmp_path.parent / "outside_prediction.npz"
    outside.write_bytes(b"prediction")
    (figures / "hccb_p418_openfoam_model_field_selection.json").write_text(
        json.dumps(
            {
                "status": "selected_p418_field_figure_learned_model",
                "selection_data_role": "validation",
                "display_data_role": "test",
                "strict_split_loss_balancing_stage": "validation_selected",
                "prediction_file": str(outside),
            }
        ),
        encoding="utf-8",
    )
    try:
        module.selected_prediction_file(tmp_path)
    except ValueError as error:
        assert "outside the project root" in str(error)
    else:
        raise AssertionError("an external selected prediction was accepted")


def test_describe_rejects_invalid_npz(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "invalid.npz"
    path.write_bytes(b"not a NumPy archive")
    row = module.describe(tmp_path, path.name)
    assert row["present"] is False
    assert row["reason"] == "invalid_file_format"


def test_describe_rejects_short_cloud_placeholder_read(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "placeholder.bin"
    path.write_bytes(b"logical content")
    original = module.sha256_and_read_size
    module.sha256_and_read_size = lambda _path: ("unused", 0)
    try:
        row = module.describe(tmp_path, path.name)
    finally:
        module.sha256_and_read_size = original
    assert row["present"] is False
    assert row["reason"] == "unreadable_cloud_placeholder"
    assert row["logical_size_bytes"] == len(b"logical content")
    assert row["read_size_bytes"] == 0
