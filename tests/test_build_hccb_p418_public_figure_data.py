from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/build_hccb_p418_public_figure_data.py"


def load_module():
    spec = importlib.util.spec_from_file_location("public_figure_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return len(list(csv.DictReader(handle)))


def test_public_figure_data_is_complete_and_path_free(tmp_path: Path) -> None:
    module = load_module()
    output = tmp_path / "public"
    payload = module.build(ROOT, output)
    assert payload["status"] == "completed_p418_public_figure_data"
    assert payload["row_counts"] == {
        "physical_response": 60,
        "seed202_integral_comparison": 9,
        "steady_model_comparison": 25,
    }
    assert row_count(output / "physical_response_60.csv") == 60
    assert row_count(output / "seed202_integral_comparison_9.csv") == 9
    assert row_count(output / "steady_model_comparison_5x5.csv") == 25
    text = "\n".join(path.read_text(encoding="utf-8") for path in output.iterdir())
    for token in module.PRIVATE_TEXT:
        assert token not in text
    summary = json.loads((output / "seed202_integral_summary.json").read_text())
    assert summary["complete_nine_case_comparison"] is True


def test_public_tables_regenerate_three_figures(tmp_path: Path) -> None:
    module = load_module()
    data = tmp_path / "data"
    figures = tmp_path / "figures"
    module.build(ROOT, data)
    commands = (
        (
            "plot_hccb_p418_physical_response.py",
            "--physical-csv",
            data / "physical_response_60.csv",
            "hccb_p418_physical_response.pdf",
        ),
        (
            "plot_hccb_p418_seed202_integral_partial.py",
            "--comparison-csv",
            data / "seed202_integral_comparison_9.csv",
            "hccb_p418_seed202_integral_9.pdf",
        ),
        (
            "plot_hccb_p418_steady_model_comparison.py",
            "--comparison-csv",
            data / "steady_model_comparison_5x5.csv",
            "hccb_p418_steady_model_comparison.pdf",
        ),
    )
    for script, flag, source, expected in commands:
        command = [sys.executable, str(ROOT / "code" / script), flag, str(source)]
        if "seed202" in script:
            command.extend(
                ["--summary-json", str(data / "seed202_integral_summary.json")]
            )
        command.extend(["--output-dir", str(figures)])
        subprocess.run(command, check=True, capture_output=True, text=True)
        assert (figures / expected).stat().st_size > 1000
