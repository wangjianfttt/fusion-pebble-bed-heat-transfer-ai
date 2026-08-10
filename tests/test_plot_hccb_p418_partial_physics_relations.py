import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/plot_hccb_p418_partial_physics_relations.py"


def test_actual_completed_case_figure_builds(tmp_path: Path) -> None:
    output = tmp_path / "figure"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--pressure-csv",
            str(
                ROOT
                / "results/hccb_p418_sourceflow_partial_pressure_correlation/pressure_correlation.csv"
            ),
            "--pressure-summary",
            str(
                ROOT
                / "results/hccb_p418_sourceflow_partial_pressure_correlation/summary.json"
            ),
            "--dimensionless-csv",
            str(
                ROOT
                / "results/hccb_p418_sourceflow_partial_dimensionless_heat_transfer_with_flux/dimensionless_heat_transfer.csv"
            ),
            "--dimensionless-summary",
            str(
                ROOT
                / "results/hccb_p418_sourceflow_partial_dimensionless_heat_transfer_with_flux/summary.json"
            ),
            "--boundary-summary",
            str(
                ROOT
                / "results/hccb_p418_sourceflow_partial_boundary_heat/summary.json"
            ),
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "p418_partial_physics_relations_ready"
    assert summary["case_count"] == 14
    assert summary["new_physical_parameters"] == []
    assert (output / "hccb_p418_partial_physics_relations.pdf").stat().st_size > 10_000
    assert (output / "hccb_p418_partial_physics_relations.png").stat().st_size > 50_000
    chinese = (output / "P418_14工况物理关系_CN.md").read_text(encoding="utf-8")
    assert "23%--26%" in chinese
    assert "不能直接当作当前靠壁局部区域的训练标签" in chinese
