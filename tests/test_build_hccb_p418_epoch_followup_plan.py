from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_only_boundary_models_receive_source_epoch_followup(tmp_path: Path) -> None:
    source = tmp_path / "convergence.json"
    source.write_text(
        json.dumps(
            {
                "requested_epochs": 100,
                "result_prefix": "hccb_p418_60",
                "models": [
                    {
                        "architecture": "pinn",
                        "split": "formal",
                        "completed_epochs": 100,
                        "published_source_epochs": 3000,
                        "best_epoch_is_final_epoch": False,
                    },
                    {
                        "architecture": "graph",
                        "split": "formal",
                        "completed_epochs": 100,
                        "published_source_epochs": 2000,
                        "best_epoch_is_final_epoch": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "plan.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "code/build_hccb_p418_epoch_followup_plan.py"),
            "--convergence-summary",
            str(source),
            "--output",
            str(output),
        ],
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["followup_run_count"] == 1
    assert payload["runs"][0]["architecture"] == "graph"
    assert payload["runs"][0]["followup_epochs"] == 2000
    assert payload["new_physical_parameters"] == []
