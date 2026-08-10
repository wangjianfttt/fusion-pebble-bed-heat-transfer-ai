from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_transient_cost_table.py"
MODELS = (
    "dmdc",
    "graph_transformer_data_only",
    "graph_transformer_energy_flux",
    "graph_transformer_factorized_energy_flux",
    "low_rank_residual_correction",
    "diffusion_residual_correction",
)


def test_builds_complete_chain_cost_table(tmp_path: Path) -> None:
    speed = tmp_path / "speed.csv"
    fields = (
        "split_name",
        "model",
        "model_size_scalar_count",
        "training_wall_time_s",
        "model_inference_seconds_per_curve",
        "wall_clock_speedup",
        "training_only_break_even_curve_count",
        "full_workflow_break_even_curve_count",
    )
    with speed.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, model in enumerate(MODELS, 1):
            writer.writerow(
                {
                    "split_name": "pair_disjoint_stress_test",
                    "model": model,
                    "model_size_scalar_count": index * 100000,
                    "training_wall_time_s": index * 60,
                    "model_inference_seconds_per_curve": index * 0.1,
                    "wall_clock_speedup": 100 / index,
                    "training_only_break_even_curve_count": index,
                    "full_workflow_break_even_curve_count": index + 10,
                }
            )
    output = tmp_path / "cost.tex"
    summary = tmp_path / "summary.json"
    subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--speed-csv",
            str(speed),
            "--output",
            str(output),
            "--summary",
            str(summary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    text = output.read_text(encoding="utf-8")
    assert "Physics GT + diffusion" in text
    assert "Break-even curves (train/full)" in text
    assert text.count("\\\\") >= len(MODELS) + 1
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["status"] == "complete_p418_transient_cost_table"
    assert len(payload["records"]) == len(MODELS)
    assert payload["records"][-1]["full_workflow_break_even_curve_count"] == 16
