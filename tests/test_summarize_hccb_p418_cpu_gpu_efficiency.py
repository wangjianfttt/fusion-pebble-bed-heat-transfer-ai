from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/summarize_hccb_p418_cpu_gpu_efficiency.py"


def result(seconds: float, peak_gpu_gb: float | None) -> dict[str, object]:
    return {
        "nodes": 100,
        "edges": 300,
        "time_points": 37,
        "spatial_temporal_mode": "factorized_static_spatial",
        "model_parameter_count": 1000,
        "physical_parameter_ids": ["P418"],
        "new_physical_parameters": [],
        "torch_num_threads": 8,
        "elapsed_seconds": seconds,
        "peak_gpu_GB": peak_gpu_gb,
        "initial_maximum_absolute_error": 0.0,
        "hydrodynamic_maximum_absolute_error": 0.0,
        "loss_finite": True,
        "all_gradients_present": True,
        "all_gradients_finite": True,
    }


def test_cpu_gpu_comparison_requires_matched_calculations(tmp_path: Path):
    cpu = tmp_path / "cpu.json"
    gpu = tmp_path / "gpu.json"
    cpu.write_text(json.dumps(result(20.0, None)), encoding="utf-8")
    gpu.write_text(json.dumps(result(2.0, 4.0)), encoding="utf-8")
    output = tmp_path / "output"
    subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--cpu-summary",
            str(cpu),
            "--gpu-summary",
            str(gpu),
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["gpu_update_speedup"] == 10.0

    changed = result(2.0, 4.0)
    changed["time_points"] = 13
    gpu.write_text(json.dumps(changed), encoding="utf-8")
    rejected = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--cpu-summary",
            str(cpu),
            "--gpu-summary",
            str(gpu),
            "--output-dir",
            str(tmp_path / "bad"),
        ],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "calculations differ" in rejected.stderr
