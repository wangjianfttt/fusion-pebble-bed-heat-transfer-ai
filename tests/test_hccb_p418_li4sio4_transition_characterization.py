from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_transition_characterization_is_source_backed() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "code"))
    from hccb_p418_li4sio4_transition_characterization import (
        load_transition_characterization,
    )

    payload, regions = load_transition_characterization()
    assert payload["source"]["official_url"].startswith("https://www.jstage.jst.go.jp/")
    assert payload["source"]["evidence_page_in_pdf"] == 3
    assert [item.transition_id for item in regions] == ["Tc1", "Tc2"]
    assert [
        (item.onset_temperature_k, item.end_temperature_k)
        for item in regions
    ] == [(921.15, 956.15), (986.15, 1008.15)]
    assert [
        item.critical_temperature_reported_k for item in regions
    ] == [938.0, 996.0]
    assert [
        item.additional_enthalpy_uptake_j_mol for item in regions
    ] == [900.0, 630.0]
    use = payload["use_in_this_project"]
    assert use["temperature_history_classification"]
    assert not use["openfoam_heat_capacity_modified"]
    assert not use["neural_network_target_modified"]
    assert not use["analytic_peak_shape_assumed"]


def test_transition_characterization_rejects_an_invented_peak_shape(
    tmp_path: Path,
) -> None:
    import sys

    sys.path.insert(0, str(ROOT / "code"))
    from hccb_p418_li4sio4_transition_characterization import (
        DEFAULT_CHARACTERIZATION,
        load_transition_characterization,
    )

    payload = json.loads(DEFAULT_CHARACTERIZATION.read_text(encoding="utf-8"))
    payload["use_in_this_project"]["analytic_peak_shape_assumed"] = True
    candidate = tmp_path / "transition_characterization.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="analytic transition peak shape"):
        load_transition_characterization(candidate)
