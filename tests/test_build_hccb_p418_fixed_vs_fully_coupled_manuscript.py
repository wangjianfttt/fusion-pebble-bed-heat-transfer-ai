from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_fixed_vs_fully_coupled_manuscript.py"
SIGNALS = (
    ("pressure_drop_Pa", "Pa"),
    ("outlet_temperature_K", "K"),
    ("maximum_solid_temperature_K", "K"),
    ("volume_average_fluid_temperature_K", "K"),
    ("volume_average_solid_temperature_K", "K"),
    ("cooling_wall_power_W", "W"),
    ("signed_mass_residual_kg_s", "kg s^-1"),
    ("net_outward_enthalpy_flow_W", "W"),
)


def write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    rows = []
    aggregate = {}
    for signal_index, (signal, unit) in enumerate(SIGNALS, start=1):
        rmse_values = []
        absolute_values = []
        normalized_values = []
        for sequence_index in range(12):
            rmse = float(signal_index + sequence_index)
            absolute = 2.0 * rmse
            normalized = rmse / 100.0
            rmse_values.append(rmse)
            absolute_values.append(absolute)
            normalized_values.append(normalized)
            rows.append(
                {
                    "sequence_id": f"sequence_{sequence_index:02d}",
                    "signal": signal,
                    "common_time_point_count": 56,
                    "comparison_start_s": 0.0,
                    "comparison_end_s": 300.0,
                    "rmse": rmse,
                    "maximum_absolute_difference": absolute,
                    "endpoint_absolute_difference": 0.5 * absolute,
                    "maximum_difference_over_fully_coupled_response_span": normalized,
                }
            )
        ordered = sorted(rmse_values)
        aggregate[signal] = {
            "unit": unit,
            "trajectory_count": 12,
            "median_rmse": 0.5 * (ordered[5] + ordered[6]),
            "largest_absolute_difference": max(absolute_values),
            "largest_absolute_difference_sequence_id": "sequence_11",
            "largest_difference_over_fully_coupled_response_span": max(
                normalized_values
            ),
            "largest_normalized_difference_sequence_id": "sequence_11",
            "median_difference_over_fully_coupled_response_span": (
                0.5
                * (
                    sorted(normalized_values)[5]
                    + sorted(normalized_values)[6]
                )
            ),
        }
    csv_path = tmp_path / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "completed_p418_fixed_vs_fully_coupled_step_comparison",
                "sequence_count": 12,
                "signals": [signal for signal, _ in SIGNALS],
                "aggregate_by_signal": aggregate,
                "new_physical_parameters": [],
            }
        ),
        encoding="utf-8",
    )
    return summary_path, csv_path


def test_builds_table_and_text_from_complete_twelve_trajectory_comparison(
    tmp_path: Path,
) -> None:
    summary, csv_path = write_inputs(tmp_path)
    table = tmp_path / "table.tex"
    text = tmp_path / "text.tex"
    completed = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--summary",
            str(summary),
            "--csv",
            str(csv_path),
            "--table-output",
            str(table),
            "--text-output",
            str(text),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    table_text = table.read_text(encoding="utf-8")
    body_text = text.read_text(encoding="utf-8")
    assert "Differences between the fixed-hydrodynamics and fully coupled" in table_text
    assert "Maximum solid temperature" in table_text
    assert "Signed mass residual" in table_text
    assert "No acceptance" not in body_text
    assert "without introducing a fitted percentage" in body_text
    assert "same 12 source--target thermal steps" in body_text


def test_rejects_incomplete_trajectory_count(tmp_path: Path) -> None:
    summary, csv_path = write_inputs(tmp_path)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["sequence_count"] = 11
    summary.write_text(json.dumps(payload), encoding="utf-8")
    completed = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--summary",
            str(summary),
            "--csv",
            str(csv_path),
            "--table-output",
            str(tmp_path / "table.tex"),
            "--text-output",
            str(tmp_path / "text.tex"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "12 trajectories" in completed.stderr
