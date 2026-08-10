#!/usr/bin/env python3
"""Build a no-solver fully coupled candidate from one fixed-flow step case.

The fixed-flow and candidate fully coupled cases retain exactly the same time-zero
fields.  Only the transient flow switches in ``fvSchemes`` and ``fvSolution`` are
changed.  This isolates the effect of allowing hydrodynamics to evolve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

INITIAL_FIELDS = (
    "fluid/U",
    "fluid/p",
    "fluid/p_rgh",
    "fluid/phi",
    "fluid/T",
    "solid/T",
)


def replace_entry(text: str, keyword: str, value: str) -> str:
    pattern = rf"(?m)^(\s*{re.escape(keyword)}\s+)[^;]+;"
    updated, count = re.subn(pattern, rf"\g<1>{value};", text)
    if count != 1:
        raise ValueError(f"expected one {keyword} entry, found {count}")
    return updated


def replace_default_ddt(text: str) -> str:
    block = re.search(r"ddtSchemes\s*\{(?P<body>.*?)\}", text, flags=re.S)
    if block is None:
        raise ValueError("cannot find ddtSchemes block")
    body = replace_entry(block.group("body"), "default", "Euler")
    return text[: block.start("body")] + body + text[block.end("body") :]


def configure_fully_coupled_transient(case: Path) -> None:
    for region in ("fluid", "solid"):
        path = case / f"system/{region}/fvSchemes"
        path.write_text(replace_default_ddt(path.read_text(encoding="utf-8")), encoding="utf-8")
    solution_path = case / "system/fluid/fvSolution"
    solution = solution_path.read_text(encoding="utf-8")
    solution = replace_entry(solution, "flow", "yes")
    solution = replace_entry(solution, "momentumPredictor", "yes")
    solution_path.write_text(solution, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_fixed_case_files(case: Path) -> list[Path]:
    required = [
        case / "step_case_metadata.json",
        case / "initial_field_map_complete.json",
        case / "system/fluid/fvSchemes",
        case / "system/fluid/fvSolution",
        case / "system/solid/fvSchemes",
    ]
    required.extend(case / "0" / name for name in INITIAL_FIELDS)
    return required


def verify_initial_field_identity(fixed_case: Path, candidate: Path) -> dict[str, object]:
    rows: dict[str, dict[str, object]] = {}
    for name in INITIAL_FIELDS:
        fixed_path = fixed_case / "0" / name
        candidate_path = candidate / "0" / name
        fixed_sha = sha256(fixed_path)
        candidate_sha = sha256(candidate_path)
        rows[name] = {
            "fixed_flow_sha256": fixed_sha,
            "fully_coupled_candidate_sha256": candidate_sha,
            "byte_identical": fixed_sha == candidate_sha,
        }
    if not all(bool(row["byte_identical"]) for row in rows.values()):
        raise ValueError(f"candidate time-zero fields differ from fixed-flow case: {rows}")
    return rows


def prepare(fixed_case: Path, output_case: Path) -> dict[str, object]:
    fixed_case = fixed_case.resolve()
    output_case = output_case.resolve()
    missing = [str(path) for path in required_fixed_case_files(fixed_case) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"fixed-flow step case is incomplete: {missing}")
    if output_case.exists():
        raise FileExistsError(output_case)

    output_case.mkdir(parents=True)
    for name in ("0", "constant", "system"):
        source = fixed_case / name
        if not source.is_dir():
            raise FileNotFoundError(source)
        shutil.copytree(source, output_case / name)

    fixed_metadata = json.loads(
        (fixed_case / "step_case_metadata.json").read_text(encoding="utf-8")
    )
    fixed_initial_record = json.loads(
        (fixed_case / "initial_field_map_complete.json").read_text(encoding="utf-8")
    )
    configure_fully_coupled_transient(output_case)
    field_identity = verify_initial_field_identity(fixed_case, output_case)

    sequence_id = str(fixed_metadata["sequence_id"])
    record = {
        "status": "p418_matched_initial_state_fully_coupled_candidate_prepared_not_run",
        "sequence_id": sequence_id,
        "source_condition_id": fixed_metadata["source_condition_id"],
        "target_condition_id": fixed_metadata["target_condition_id"],
        "fixed_flow_reference_case": str(fixed_case),
        "candidate_case": str(output_case),
        "initial_state_rule": (
            "Both cases use byte-identical target-endpoint U, p, p_rgh and phi "
            "and byte-identical source-endpoint fluid and solid temperature fields."
        ),
        "difference_after_t0": (
            "The fixed-flow case holds U, p, p_rgh and phi fixed; the candidate "
            "fully coupled case advances continuity, momentum and both energy regions."
        ),
        "fixed_flow_initialization_status": fixed_initial_record.get("status"),
        "time_zero_field_identity": field_identity,
        "flow": "yes",
        "momentumPredictor": "yes",
        "fluid_and_solid_ddt_scheme": "Euler",
        "scientific_use": "candidate representative short calculation only",
        "formal_twelve_curve_execution_approved": False,
        "openfoam_solver_started": False,
        "new_physical_parameters": [],
    }
    (output_case / "matched_initial_state_candidate.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_case / "step_case_metadata.json").write_text(
        json.dumps(
            {
                **fixed_metadata,
                "status": "p418_matched_initial_state_fully_coupled_candidate_prepared_not_run",
                "transient_model": "fully_coupled_flow_heat_from_fixed_flow_time_zero_state",
                "flow_treatment": (
                    "time-dependent continuity, momentum, fluid energy and solid energy "
                    "from the exact fixed-flow time-zero state"
                ),
                "candidate_only": True,
                "formal_execution_approved": False,
                "new_physical_parameters": [],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-step-case", type=Path, required=True)
    parser.add_argument("--output-case", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.fixed_step_case, args.output_case),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
