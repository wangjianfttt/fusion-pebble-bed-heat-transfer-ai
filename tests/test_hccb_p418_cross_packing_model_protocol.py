from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_hccb_p418_cross_packing_operator",
    ROOT / "code" / "evaluate_hccb_p418_cross_packing_operator.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "code"))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def protocol():
    return json.loads(
        (ROOT / "parameters" / "hccb_p418_cross_packing_model_protocol.json").read_text(
            encoding="utf-8"
        )
    )


def test_protocol_keeps_seed303_out_of_normalization_and_selection():
    payload = protocol()
    MODULE.validate_protocol(payload)
    assert payload["normalization"]["packing_seed"] == 101
    final = [
        item
        for item in payload["evaluation_packings"]
        if item["role"] == "final_zero_shot_packing"
    ]
    assert [item["seed"] for item in final] == [303]
    assert len(final[0]["condition_ids"]) == 9
    assert final[0]["state_targets"].endswith("regional_state_targets.npz")
    assert final[0]["mass_targets"].endswith("regional_mass_flux_targets.npz")
    assert final[0]["energy_targets"].endswith("regional_energy_flux_targets.npz")
    assert payload["new_physical_parameter_values_added"] == []


def test_protocol_rejects_final_packing_normalization():
    payload = protocol()
    payload["normalization"]["packing_seed"] = 303
    with pytest.raises(ValueError, match="cannot define normalization"):
        MODULE.validate_protocol(payload)


def test_fixed_model_record_is_immutable(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    statistics = tmp_path / "statistics.json"
    protocol_path = tmp_path / "protocol.json"
    checkpoint.write_bytes(b"model-a")
    statistics.write_text("{}", encoding="utf-8")
    protocol_path.write_text("{}", encoding="utf-8")
    record_path = tmp_path / "fixed.json"
    MODULE.write_or_verify_fixed_model_record(
        path=record_path,
        checkpoint=checkpoint,
        statistics=statistics,
        protocol=protocol_path,
        architecture="transolver",
        normalization_seed=101,
    )
    checkpoint.write_bytes(b"model-b")
    with pytest.raises(ValueError, match="another model"):
        MODULE.write_or_verify_fixed_model_record(
            path=record_path,
            checkpoint=checkpoint,
            statistics=statistics,
            protocol=protocol_path,
            architecture="transolver",
            normalization_seed=101,
        )
