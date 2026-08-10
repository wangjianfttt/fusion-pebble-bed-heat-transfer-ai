import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/compare_hccb_p418_preflight_formal_case.py"


def load_module():
    spec = importlib.util.spec_from_file_location("compare_preflight_formal", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_case(root: Path, pressure: float = 8.7) -> Path:
    root.mkdir()
    sample = root / "sample.npz"
    sample.write_bytes(b"same-three-dimensional-field")
    digest = hashlib.sha256(sample.read_bytes()).hexdigest()
    (root / "formal_sample_complete.json").write_text(
        json.dumps(
            {
                "time": "200",
                "training_sample": str(sample),
                "training_sample_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    (root / "cht_result_summary_200.json").write_text(
        json.dumps(
            {
                "solver_finished": True,
                "reported_iteration": 200.0,
                "flow": {"pressure_drop_Pa": pressure, "relative_mass_difference": 1e-9},
                "temperature": {"outlet_average_K": 540.0},
                "heat_balance": {"relative_energy_difference": 1e-5},
            }
        ),
        encoding="utf-8",
    )
    return root


def test_exact_repetition_is_reported(tmp_path: Path) -> None:
    module = load_module()
    preflight = make_case(tmp_path / "preflight")
    formal = make_case(tmp_path / "formal")
    result = module.compare(preflight, formal)
    assert result["status"] == "corrected_preflight_exactly_reproduced_by_first_formal_case"
    assert result["compared_numeric_quantity_count"] == 5
    assert result["maximum_absolute_difference"] == 0.0
    assert result["training_sample_hashes_identical"]
    assert result["new_physical_parameters"] == []


def test_changed_physical_result_is_reported(tmp_path: Path) -> None:
    module = load_module()
    preflight = make_case(tmp_path / "preflight")
    formal = make_case(tmp_path / "formal", pressure=9.0)
    result = module.compare(preflight, formal)
    assert result["status"] == "preflight_and_first_formal_case_differ"
    assert result["maximum_absolute_difference"] == pytest.approx(0.3)
