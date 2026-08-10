from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/run_hccb_p418_loss_balancing_protocol.py"
SPEC = importlib.util.spec_from_file_location("loss_protocol", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_candidate_arguments_cover_recorded_candidates() -> None:
    sources = json.loads(
        (ROOT / "parameters/hccb_p418_loss_balancing_sources.json").read_text(
            encoding="utf-8"
        )
    )
    candidates = sources["formal_candidates"]
    assert len(candidates) == 4
    for candidate in candidates:
        arguments = MODULE.candidate_arguments(candidate)
        assert arguments[arguments.index("--loss-balance-candidate-id") + 1] == candidate[
            "candidate_id"
        ]
        assert arguments[arguments.index("--loss-balance-method") + 1] == candidate[
            "method"
        ]
        if candidate["method"] == "relobralo":
            assert "--relobralo-temperature" in arguments
            assert "--relobralo-alpha" in arguments
            assert "--relobralo-rho" in arguments
        else:
            assert "--relobralo-temperature" not in arguments
