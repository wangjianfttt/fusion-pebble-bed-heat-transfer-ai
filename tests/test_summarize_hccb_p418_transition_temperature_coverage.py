from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/summarize_hccb_p418_transition_temperature_coverage.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state(temperatures: list[list[float]]) -> np.ndarray:
    values = np.zeros((len(temperatures), 3, 5), dtype=np.float32)
    values[:, 0, 4] = 700.0
    values[:, 1:, 4] = np.asarray(temperatures, dtype=np.float32)
    return values


def test_transition_coverage_uses_p431_and_source_reported_regions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sequences = root / "sequences"
        sequences.mkdir()
        geometry = root / "regional_sequence_geometry.npz"
        np.savez_compressed(
            geometry,
            node_type=np.asarray([0, 1, 1], dtype=np.int8),
            node_volume_m3=np.asarray([1.0, 2.0, 3.0]),
        )
        first = sequences / "lower_only.npz"
        np.savez_compressed(
            first,
            time_s=np.asarray([0.0, 1.0, 3.0]),
            state_physical=state([[900.0, 910.0], [940.0, 930.0], [950.0, 970.0]]),
        )
        second = sequences / "both.npz"
        np.savez_compressed(
            second,
            time_s=np.asarray([0.0, 1.0, 3.0]),
            state_physical=state([[900.0, 920.0], [960.0, 980.0], [1010.0, 1000.0]]),
        )
        dataset = {
            "status": "p418_regional_thermal_step_sequences_ready",
            "sequence_count": 2,
            "waiting_sequence_count": 0,
            "state_names": [
                "Ux_m_s",
                "Uy_m_s",
                "Uz_m_s",
                "pressure_Pa",
                "temperature_K",
            ],
            "regional_geometry_file": geometry.name,
            "regional_geometry_sha256": sha256(geometry),
            "sequences": [
                {
                    "sequence_id": "lower_only",
                    "family": "temperature",
                    "sequence_file": "sequences/lower_only.npz",
                    "sequence_sha256": sha256(first),
                },
                {
                    "sequence_id": "both",
                    "family": "temperature",
                    "sequence_file": "sequences/both.npz",
                    "sequence_sha256": sha256(second),
                },
            ],
        }
        dataset_path = root / "dataset_index.json"
        dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
        output = root / "coverage"
        latex = root / "coverage.tex"
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--dataset-index",
                str(dataset_path),
                "--output-dir",
                str(output),
                "--latex-output",
                str(latex),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["transition_temperatures_K"] == [938.0, 996.0]
        assert summary["sequence_count_reaching_each_transition"] == [2, 1]
        assert summary["sequence_count_with_regional_trajectory_crossing"] == [2, 1]
        assert summary["sequence_count_entering_each_transition_region"] == [2, 1]
        assert summary["transition_regions"][0]["temperature_range_K"] == [
            921.15,
            956.15,
        ]
        assert summary["transition_regions"][1]["temperature_range_K"] == [
            986.15,
            1008.15,
        ]
        assert summary["transition_regions"][0][
            "additional_enthalpy_uptake_J_mol"
        ] == 900.0
        assert summary["transition_regions"][1][
            "additional_enthalpy_uptake_J_mol"
        ] == 630.0
        assert summary["new_physical_parameters"] == []
        assert summary["new_model_physical_parameters"] == []
        assert "No fitted tolerance" in summary["interpretation"]
        assert "not imposed on the OpenFOAM solution" in summary["interpretation"]
        assert summary["transition_parameter_source"]["parameter_id"] == "P431"
        assert (output / "transition_temperature_coverage.csv").is_file()
        chinese = (output / "transition_temperature_coverage_CN.md").read_text(
            encoding="utf-8"
        )
        assert "没有人为规定相变附近多少K" in chinese
        assert "921.15--956.15 K" in chinese
        assert "986.15--1008.15 K" in chinese
        assert "900和630 J/mol" in chinese
        assert "938 K、996 K" in chinese
        latex_text = latex.read_text(encoding="utf-8")
        assert "P431" in latex_text
        assert "\\cite{kleykamp1996enthalpy}" in latex_text


def test_step_pipeline_runs_coverage_before_model_training() -> None:
    pipeline = (ROOT / "code/run_hccb_p418_step_responses.sh").read_text(
        encoding="utf-8"
    )
    export_position = pipeline.index("export_hccb_p418_step_regional_sequences.py")
    coverage_position = pipeline.index(
        "summarize_hccb_p418_transition_temperature_coverage.py"
    )
    model_position = pipeline.index("if [[ ${RUN_MODEL_TRAINING} == 1")
    assert export_position < coverage_position < model_position
    assert "generated_transition_temperature_coverage.tex" in pipeline
