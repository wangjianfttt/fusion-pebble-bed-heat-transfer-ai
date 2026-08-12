from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    path = ROOT / f"code/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_final_record_helpers_use_project_relative_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = project / "results/model/summary.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}\n", encoding="utf-8")
    for module_name in (
        "select_hccb_p418_field_figure_model",
        "plot_hccb_p418_field_cloud_comparison",
        "plot_hccb_p418_transient_model_comparison",
        "summarize_hccb_p418_step_model_comparison",
    ):
        module = load(module_name)
        assert module.project_relative(source, project) == "results/model/summary.json"


def test_selected_loss_integration_records_relative_selection_path() -> None:
    source = (ROOT / "code/run_hccb_p418_selected_loss_downstream.py").read_text(
        encoding="utf-8"
    )
    assert '"selection_record": str(selection_path.relative_to(result_dir))' in source
    assert '"selection_record": str(selection_path),' not in source
