from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/verify_hccb_p418_three_mesh_recovery.py"
CELLS = {
    "coarse": (160989, 200162),
    "medium": (432384, 515540),
    "fine": (858419, 1011645),
}
METRICS = (
    "pressure_drop_Pa",
    "outlet_temperature_change_K",
    "solid_maximum_temperature_change_K",
    "cooling_wall_heat_fraction",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fixture(root: Path) -> None:
    levels = []
    for index, (name, (fluid, solid)) in enumerate(CELLS.items(), start=1):
        levels.append(
            {
                "mesh_level": name,
                "fluid_cells": fluid,
                "solid_cells": solid,
                "total_cells": fluid + solid,
                "fluid_basic_check_passes": True,
                "solid_basic_check_passes": True,
                "pressure_drop_Pa": 20.0 + index,
                "outlet_temperature_change_K": 70.0 + index,
                "solid_maximum_temperature_change_K": 1.0 + index,
                "cooling_wall_heat_fraction": -0.2 - 0.01 * index,
            }
        )
    convergence = []
    for index, metric in enumerate(METRICS, start=1):
        convergence.append(
            {
                "metric": metric,
                "coarse_value": 1.0 + index,
                "medium_value": 1.1 + index,
                "fine_value": 1.2 + index,
                "coarse_to_medium_refinement_ratio": 1.3,
                "medium_to_fine_refinement_ratio": 1.2,
                "observed_order": 2.0,
                "richardson_extrapolated_value": 1.3 + index,
                "fine_gci_fraction": 0.02,
                "fine_gci_absolute": 0.01,
                "convergence_status": "monotonic_gci_reported",
            }
        )
    summary = {
        "status": "completed_three_mesh_p418_cht_comparison",
        "mesh_levels": levels,
        "grid_convergence": convergence,
        "new_physical_parameters": [],
    }
    root.mkdir()
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (root / "engineering_observables.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(levels[0]))
        writer.writeheader()
        writer.writerows(levels)
    with (root / "mesh_gci.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(convergence[0]))
        writer.writeheader()
        writer.writerows(convergence)


def command(root: Path, output: Path | None = None) -> list[str]:
    values = [
        sys.executable,
        str(SCRIPT),
        "--root",
        str(root),
        "--expected-summary-sha256",
        sha256(root / "summary.json"),
        "--expected-gci-sha256",
        sha256(root / "mesh_gci.csv"),
    ]
    if output is not None:
        values.extend(["--output", str(output)])
    return values


def test_formal_three_mesh_recovery_is_verified(tmp_path: Path) -> None:
    root = tmp_path / "formal"
    output = tmp_path / "verification.json"
    write_fixture(root)
    completed = subprocess.run(
        command(root, output), check=True, capture_output=True, text=True
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "verified_formal_p418_three_mesh_recovery"
    assert all(payload["checks"].values())
    assert json.loads(output.read_text())["new_physical_parameters"] == []


def test_early_invalid_mesh_counts_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "early"
    write_fixture(root)
    payload = json.loads((root / "summary.json").read_text())
    payload["mesh_levels"][0]["fluid_cells"] = 160846
    payload["mesh_levels"][0]["total_cells"] = 360353
    (root / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    completed = subprocess.run(command(root), capture_output=True, text=True)
    assert completed.returncode != 0
    assert "coarse_formal_cell_counts" in completed.stderr


def test_failed_basic_mesh_check_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "failed-check"
    write_fixture(root)
    payload = json.loads((root / "summary.json").read_text())
    payload["mesh_levels"][1]["solid_basic_check_passes"] = False
    (root / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    completed = subprocess.run(command(root), capture_output=True, text=True)
    assert completed.returncode != 0
    assert "medium_basic_mesh_checks" in completed.stderr


def test_mismatched_gci_csv_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "bad-gci"
    write_fixture(root)
    rows = list(csv.DictReader((root / "mesh_gci.csv").open(encoding="utf-8")))
    rows[0]["fine_gci_fraction"] = "0.25"
    with (root / "mesh_gci.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    completed = subprocess.run(command(root), capture_output=True, text=True)
    assert completed.returncode != 0
    assert "pressure_drop_Pa_gci_csv_matches_summary" in completed.stderr
