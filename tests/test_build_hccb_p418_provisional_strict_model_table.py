import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "build_hccb_p418_provisional_strict_model_table.py"


def load_module():
    spec = importlib.util.spec_from_file_location("provisional_model_table", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_completed_rows_share_the_strict_split():
    module = load_module()
    rows = module.load_completed_rows(ROOT / "results/hccb_p418_physical_steps_12")
    assert len(rows) == 6
    assert {row["result_status"] for row in rows} == {
        "completed_formal_pair_disjoint_result"
    }
    assert all(row["fluid_temperature_RMSE_K"] > 0.0 for row in rows)
    assert all(row["solid_temperature_RMSE_K"] > 0.0 for row in rows)
    assert {
        (row["split_train_count"], row["split_validation_count"], row["split_test_count"])
        for row in rows
    } == {(6, 2, 4)}
    assert all(
        row["common_energy_test_volume_weighted_residual_ratio"] > 0.0
        for row in rows
    )
    assert all(row["source_energy_summary_sha256"] for row in rows)


def test_low_rank_range_limitation_is_not_hidden():
    module = load_module()
    rows = module.load_completed_rows(ROOT / "results/hccb_p418_physical_steps_12")
    low_rank = next(
        row for row in rows if row["model_id"] == "low_rank_temperature_residual"
    )
    assert low_rank["test_temperature_range_status"] == (
        "test_prediction_accepted_by_common_energy_evaluator"
    )
    assert low_rank["common_energy_rejected_roles"] == "train;validation"


def test_generated_record_remains_provisional():
    path = (
        ROOT
        / "results/hccb_p418_physical_steps_12"
        / "provisional_strict_pair_disjoint_comparison"
        / "provisional_model_comparison.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["completed_model_count"] == 6
    assert payload["final_ranking_allowed"] is False
    assert payload["hardware_normalized_training_time_available"] is False
    assert payload["split_case_counts"] == {"train": 6, "validation": 2, "test": 4}
    assert payload["pending_model_families"] == ["diffusion_temperature_correction"]
