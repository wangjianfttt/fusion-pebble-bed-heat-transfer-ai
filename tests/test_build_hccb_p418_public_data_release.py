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
