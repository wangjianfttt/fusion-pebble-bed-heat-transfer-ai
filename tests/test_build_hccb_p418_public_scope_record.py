from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_public_scope_record.py"
SOURCE = ROOT / "results/hccb_p418_scope_limits_20260810/formal_failure_record_18963.json"


def load_module():
    spec = importlib.util.spec_from_file_location("public_scope_record", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_public_scope_record_is_path_free_and_preserves_physics(tmp_path: Path) -> None:
    module = load_module()
    output = tmp_path / "direct_transport_scope_limit.json"
    payload = module.build(SOURCE, output)
    assert payload["job_id"] == 18963
    assert payload["last_logged_physical_time_s"] == 0.0015220808
    assert payload["failure"]["query_temperature_K"] == 1308.759277
    assert payload["failure"]["table_upper_limit_K"] == 1300.0
    assert payload["completion_marker_present"] is False
    assert payload["observable_signal_count"] == 0
    text = output.read_text(encoding="utf-8")
    for token in module.PRIVATE_TOKENS:
        assert token not in text
    assert json.loads(text) == payload
