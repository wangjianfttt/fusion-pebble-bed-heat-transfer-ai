from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_local_transport_model_support.py"
CONTRACT = ROOT / "parameters/hccb_p418_local_transport_model_contract.json"


def test_contract_keeps_local_flow_interface_and_wall_physics() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["new_physical_parameters"] == []
    assert "local_velocity_vector_m_s" in payload["node_inputs"]
    assert "paired_interface_heat_flow_W" in payload["edge_inputs_or_targets"]
    assert "cooling_wall_temperature_K" in payload["case_inputs"]
    assert any("P417/P419" in rule and "不作为" in rule for rule in payload["rules"])
    assert set(payload["physical_source_ids"]) >= {
        "P048",
        "P070",
        "P071",
        "P092",
        "P418",
        "P419",
        "P420",
        "P421",
        "P425",
        "P426",
        "P427",
    }


def test_current_fourteen_case_support_builds(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(tmp_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "p418_local_transport_model_support_ready"
    assert payload["case_count"] == 14
    assert payload["support_is_partial"] is True
    assert all(payload["checks"].values())
    assert payload["ranges"]["local_to_throughflow_Re_ratio"][0] > 1.2
    assert payload["ranges"]["OpenFOAM_interface_flux_Nusselt_number"][0] > 4.0
    assert 0 < payload["finite_value_counts"]["P419_aggregate_Nusselt_number"] < 14
    assert (tmp_path / "local_transport_model_support.csv").is_file()
    note = tmp_path / "P418_局部三维模型数据说明_CN.md"
    assert note.is_file()
    text = note.read_text(encoding="utf-8")
    assert "只汇总早期14组稳态结果" in text
    assert "12条全耦合流动换热阶跃尚未全部完成" not in text
