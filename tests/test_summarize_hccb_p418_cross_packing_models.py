from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "summarize_hccb_p418_cross_packing_models.py"


def case(index: int, scale: float):
    return {
        "condition_id": f"case_{index}",
        "state_normalized_rmse": scale * (index + 1),
        "fluid_temperature_volume_weighted_rmse_K": scale * (index + 1),
        "solid_temperature_volume_weighted_rmse_K": scale * (index + 1),
        "solid_hotspot_location_error_m": scale * (index + 1),
        "engineering_absolute_errors": {
            "pressure_drop_Pa": scale * (index + 1),
            "outlet_temperature_K": scale * (index + 1),
            "solid_maximum_temperature_K": scale * (index + 1),
            "cooling_wall_heat_into_fluid_W": scale * (index + 1),
            "solid_to_fluid_interphase_net_W": scale * (index + 1),
        },
        "local_mass_l1_over_two_inlet": scale * (index + 1),
        "global_mass_imbalance_over_inlet": scale * (index + 1),
        "local_energy_l1_over_two_generated_power": scale * (index + 1),
        "global_energy_imbalance_over_generated_power": scale * (index + 1),
    }


def write_run(path: Path, seed: int, scale: float):
    path.write_text(
        json.dumps(
            {
                "status": "cross_packing_conservative_evaluation_complete",
                "packing_seed": seed,
                "packing_role": "development_packing" if seed == 202 else "final_zero_shot_packing",
                "architecture": "transolver",
                "cases": [case(index, scale) for index in range(9)],
            }
        ),
        encoding="utf-8",
    )


def test_summary_keeps_physical_metrics_separate(tmp_path: Path):
    seed202 = tmp_path / "seed202.json"
    seed303 = tmp_path / "seed303.json"
    output = tmp_path / "summary"
    write_run(seed202, 202, 1.0)
    write_run(seed303, 303, 2.0)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(seed202),
            str(seed303),
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["composite_score_used"] is False
    assert len(summary["metric_definition"]) == 13
    ratios = summary["seed303_to_seed202"]
    assert ratios
    assert all(row["mean_ratio_seed303_to_seed202"] == 2.0 for row in ratios)
    assert (output / "cross_packing_model_metrics.csv").is_file()
    chinese = (output / "P418_跨装填模型结果_CN.md").read_text(encoding="utf-8")
    assert "不将温度、压降、热量和守恒误差合成一个总分" in chinese


def test_summary_rejects_incomplete_nine_case_run(tmp_path: Path):
    path = tmp_path / "short.json"
    write_run(path, 202, 1.0)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cases"].pop()
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(path), "--output-dir", str(tmp_path / "out")],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "nine unique conditions" in result.stderr
