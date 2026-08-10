#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_hccb_dense_cht_p418_matrix import validate_resumable_case  # noqa: E402
from hccb_p418_source_contract import CASE_PHYSICS_PARAMETER_IDS  # noqa: E402


def complete_case(root: Path, condition_id: str, packing_hash: str) -> Path:
    case = root / condition_id
    for relative in (
        "constant/fluid/polyMesh/boundary",
        "constant/solid/polyMesh/boundary",
        "system/controlDict",
        "system/fvSolution",
    ):
        path = case / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test\n", encoding="ascii")
    (case / "cht_smoke_metadata.json").write_text(
        json.dumps(
            {
                "operating_condition_id": condition_id,
                "parameter_ids": list(CASE_PHYSICS_PARAMETER_IDS),
                "mesh_source_packing_sha256": packing_hash,
                "source_channel_volume_flow_preserved": True,
                "pore_opening_boundary_velocity_m_s": 0.125,
                "inlet_open_area_fraction": 0.4,
            }
        ),
        encoding="utf-8",
    )
    return case


def test_complete_existing_case_can_be_reused(tmp_path: Path) -> None:
    case = complete_case(tmp_path, "u0p05_T300_q4p85", "packing-a")
    result = validate_resumable_case(
        case=case,
        condition_id="u0p05_T300_q4p85",
        mesh_source_packing_sha256="packing-a",
    )
    assert result["operating_condition_id"] == "u0p05_T300_q4p85"


def test_existing_case_from_another_packing_is_rejected(tmp_path: Path) -> None:
    case = complete_case(tmp_path, "u0p05_T300_q4p85", "packing-a")
    with pytest.raises(ValueError, match="different packing"):
        validate_resumable_case(
            case=case,
            condition_id="u0p05_T300_q4p85",
            mesh_source_packing_sha256="packing-b",
        )


def test_legacy_case_without_source_flow_mapping_is_rejected(tmp_path: Path) -> None:
    case = complete_case(tmp_path, "u0p05_T300_q4p85", "packing-a")
    metadata_path = case / "cht_smoke_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_channel_volume_flow_preserved"] = False
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="does not preserve"):
        validate_resumable_case(
            case=case,
            condition_id="u0p05_T300_q4p85",
            mesh_source_packing_sha256="packing-a",
        )


def test_partial_existing_case_is_rejected_without_deletion(tmp_path: Path) -> None:
    case = complete_case(tmp_path, "u0p05_T300_q4p85", "packing-a")
    missing = case / "constant/solid/polyMesh/boundary"
    missing.unlink()
    with pytest.raises(ValueError, match="incomplete"):
        validate_resumable_case(
            case=case,
            condition_id="u0p05_T300_q4p85",
            mesh_source_packing_sha256="packing-a",
        )
    assert case.is_dir()
    assert not missing.exists()
