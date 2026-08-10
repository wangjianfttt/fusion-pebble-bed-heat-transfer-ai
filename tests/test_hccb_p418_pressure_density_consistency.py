from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch runs remotely")
def test_pressure_density_check_accepts_absolute_pressure_and_rejects_gauge(tmp_path: Path) -> None:
    import torch

    from hccb_source_backed_thermophysical import helium_density

    fields = tmp_path / "fields"
    fields.mkdir()
    pressure = np.asarray([119990.0, 120000.0, 120020.0], dtype=np.float64)
    temperature = np.asarray([350.0, 500.0, 700.0], dtype=np.float64)
    density = helium_density(
        torch.as_tensor(pressure), torch.as_tensor(temperature)
    ).numpy()
    np.savez_compressed(
        fields / "case.npz",
        fluid_pressure_Pa=pressure,
        fluid_temperature_K=temperature,
        fluid_density_kg_m3=density,
    )
    index = {
        "conditions": [
            {
                "condition_id": "case",
                "outlet_pressure_Pa": 120000.0,
                "field_file": "fields/case.npz",
            }
        ]
    }
    index_path = tmp_path / "dataset_index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    output = tmp_path / "output"
    command = [
        sys.executable,
        str(ROOT / "code/check_hccb_p418_pressure_density_consistency.py"),
        "--dataset-index",
        str(index_path),
        "--output-dir",
        str(output),
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "p418_pressure_density_consistency_ready"
    assert summary["overall_maximum_density_relative_difference"] < 1.0e-12

    np.savez_compressed(
        fields / "case.npz",
        fluid_pressure_Pa=pressure - 120000.0,
        fluid_temperature_K=temperature,
        fluid_density_kg_m3=density,
    )
    rejected = subprocess.run(command, check=False, text=True, capture_output=True)
    assert rejected.returncode != 0
    assert "gauge rather than absolute pressure" in rejected.stderr
