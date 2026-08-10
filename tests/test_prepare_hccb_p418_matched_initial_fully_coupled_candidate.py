from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from prepare_hccb_p418_matched_initial_fully_coupled_candidate import (  # noqa: E402
    INITIAL_FIELDS,
    prepare,
)


def write_case_file(path: Path, text: str = "value\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_fixed_case(root: Path) -> Path:
    case = root / "fixed"
    for index, name in enumerate(INITIAL_FIELDS):
        write_case_file(case / "0" / name, f"field-{index}\n")
    write_case_file(
        case / "system/fluid/fvSchemes",
        "ddtSchemes\n{\n    default steadyState;\n}\n",
    )
    write_case_file(
        case / "system/solid/fvSchemes",
        "ddtSchemes\n{\n    default steadyState;\n}\n",
    )
    write_case_file(
        case / "system/fluid/fvSolution",
        "solvers\n{\n}\nPIMPLE\n{\n    flow no;\n    momentumPredictor no;\n}\n",
    )
    write_case_file(case / "constant/fluid/physicalProperties")
    write_case_file(case / "constant/solid/physicalProperties")
    (case / "step_case_metadata.json").write_text(
        json.dumps(
            {
                "sequence_id": "source_up_u0p15_T700",
                "source_condition_id": "u0p15_T700_q4p85",
                "target_condition_id": "u0p15_T700_q8p85",
                "new_physical_parameters": [],
            }
        ),
        encoding="utf-8",
    )
    (case / "initial_field_map_complete.json").write_text(
        json.dumps({"status": "verified_p418_quasi_steady_flow_thermal_step_initial_field"}),
        encoding="utf-8",
    )
    return case


def test_candidate_retains_identical_initial_fields_and_enables_flow(tmp_path: Path) -> None:
    fixed = build_fixed_case(tmp_path)
    candidate = tmp_path / "candidate"
    record = prepare(fixed, candidate)

    assert record["openfoam_solver_started"] is False
    assert record["formal_twelve_curve_execution_approved"] is False
    assert record["new_physical_parameters"] == []
    assert all(
        row["byte_identical"] for row in record["time_zero_field_identity"].values()
    )
    for name in INITIAL_FIELDS:
        assert (fixed / "0" / name).read_bytes() == (candidate / "0" / name).read_bytes()

    solution = (candidate / "system/fluid/fvSolution").read_text(encoding="utf-8")
    assert "flow yes;" in solution
    assert "momentumPredictor yes;" in solution
    for region in ("fluid", "solid"):
        schemes = (candidate / f"system/{region}/fvSchemes").read_text(encoding="utf-8")
        assert "default Euler;" in schemes


def test_candidate_plan_explicitly_forbids_formal_solver() -> None:
    plan = json.loads(
        (ROOT / "parameters/hccb_p418_matched_initial_coupling_candidate.json").read_text(
            encoding="utf-8"
        )
    )
    assert plan["status"] == "candidate_preflight_only_not_approved_for_solver"
    assert plan["formal_twelve_curve_execution_approved"] is False
    assert plan["new_physical_parameters"] == []
