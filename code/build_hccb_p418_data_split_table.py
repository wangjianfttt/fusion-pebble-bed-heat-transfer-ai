#!/usr/bin/env python3
"""Build a compact manuscript table for the registered steady and transient splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STEADY_ORDER = (
    "interleaved_all_ranges",
    "temperature_extrapolation",
    "velocity_extrapolation",
    "heat_source_interpolation",
    "heat_source_extrapolation",
)
TRANSIENT_ORDER = (
    "direction_down_test",
    "direction_up_test",
    "pair_disjoint_stress_test",
)
ROLES = ("train", "validation", "test")


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_partition(
    *, split_name: str, split: dict, expected_ids: set[str]
) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {}
    combined: list[str] = []
    for role in ROLES:
        values = list(split.get(role, []))
        if not values:
            raise ValueError(f"{split_name}/{role} is empty")
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate identifier within {split_name}/{role}")
        roles[role] = values
        combined.extend(values)
    if len(combined) != len(set(combined)):
        raise ValueError(f"roles overlap in {split_name}")
    actual = set(combined)
    if actual != expected_ids:
        missing = sorted(expected_ids - actual)
        extra = sorted(actual - expected_ids)
        raise ValueError(f"incomplete partition {split_name}: missing={missing}, extra={extra}")
    return roles


def _condition_values(
    role_ids: list[str], conditions: dict[str, dict], field: str
) -> list[float]:
    return sorted({float(conditions[item][field]) for item in role_ids})


def _number(value: float) -> str:
    return f"{value:g}"


def _list(values: list[float]) -> str:
    return ", ".join(_number(value) for value in values)


def collect_steady_records(payload: dict) -> list[dict[str, object]]:
    conditions_list = payload.get("conditions", [])
    conditions = {item["condition_id"]: item for item in conditions_list}
    if len(conditions) != int(payload.get("condition_count", -1)):
        raise ValueError("steady condition count or identifiers are inconsistent")
    if len(conditions) != 60:
        raise ValueError(f"expected 60 steady conditions, found {len(conditions)}")
    expected = set(conditions)
    splits = payload.get("splits", {})
    missing_splits = [name for name in STEADY_ORDER if name not in splits]
    if missing_splits:
        raise ValueError(f"missing steady splits: {missing_splits}")

    labels = {
        "interleaved_all_ranges": "Interleaved combinations",
        "temperature_extrapolation": "Temperature extrapolation",
        "velocity_extrapolation": "Velocity extrapolation",
        "heat_source_interpolation": "Heat-source interpolation",
        "heat_source_extrapolation": "Heat-source extrapolation",
    }
    records: list[dict[str, object]] = []
    for name in STEADY_ORDER:
        roles = ensure_partition(split_name=name, split=splits[name], expected_ids=expected)
        values = {
            role: {
                "velocity": _condition_values(
                    ids, conditions, "inlet_velocity_m_s"
                ),
                "temperature": _condition_values(
                    ids, conditions, "inlet_temperature_K"
                ),
                "source": _condition_values(
                    ids, conditions, "solid_heat_source_MW_m3"
                ),
            }
            for role, ids in roles.items()
        }
        if name == "interleaved_all_ranges":
            descriptions = {
                role: f"all sampled levels ({len(roles[role])})" for role in ROLES
            }
        elif name == "temperature_extrapolation":
            descriptions = {
                role: rf"$T_{{\rm in}}={_list(values[role]['temperature'])}$ K ({len(roles[role])})"
                for role in ROLES
            }
        elif name == "velocity_extrapolation":
            descriptions = {
                role: rf"$u_{{\rm in}}={_list(values[role]['velocity'])}$ m s$^{{-1}}$ ({len(roles[role])})"
                for role in ROLES
            }
        else:
            descriptions = {
                role: rf"$q'''={_list(values[role]['source'])}$ MW m$^{{-3}}$ ({len(roles[role])})"
                for role in ROLES
            }
            if name == "heat_source_interpolation":
                descriptions["train"] += "; disjoint combinations"
                descriptions["validation"] += "; disjoint combinations"
        records.append(
            {
                "class": "steady",
                "name": name,
                "label": labels[name],
                "train_count": len(roles["train"]),
                "validation_count": len(roles["validation"]),
                "test_count": len(roles["test"]),
                "train_description": descriptions["train"],
                "validation_description": descriptions["validation"],
                "test_description": descriptions["test"],
                "role_condition_ids": roles,
            }
        )
    return records


def _reverse_key(sequence: dict) -> tuple[str, str]:
    return tuple(sorted((sequence["source_condition_id"], sequence["target_condition_id"])))


def collect_transient_records(split_payload: dict, plan_payload: dict) -> list[dict[str, object]]:
    sequences_list = plan_payload.get("sequences", [])
    sequences = {item["sequence_id"]: item for item in sequences_list}
    if len(sequences) != 12:
        raise ValueError(f"expected 12 transient trajectories, found {len(sequences)}")
    expected = set(sequences)
    splits = split_payload.get("splits", {})
    missing_splits = [name for name in TRANSIENT_ORDER if name not in splits]
    if missing_splits:
        raise ValueError(f"missing transient splits: {missing_splits}")

    labels = {
        "direction_down_test": "Downward-step prediction",
        "direction_up_test": "Upward-step prediction",
        "pair_disjoint_stress_test": "Endpoint-pair prediction",
    }
    descriptions = {
        "direction_down_test": (
            "six paired curves",
            "three upward curves",
            "three downward curves",
        ),
        "direction_up_test": (
            "six paired curves",
            "three downward curves",
            "three upward curves",
        ),
        "pair_disjoint_stress_test": (
            "three endpoint pairs",
            "one endpoint pair",
            "two unseen endpoint pairs",
        ),
    }
    records: list[dict[str, object]] = []
    for name in TRANSIENT_ORDER:
        roles = ensure_partition(split_name=name, split=splits[name], expected_ids=expected)
        if name == "pair_disjoint_stress_test":
            role_by_id = {
                sequence_id: role for role, ids in roles.items() for sequence_id in ids
            }
            pair_roles: dict[tuple[str, str], set[str]] = {}
            for sequence_id, sequence in sequences.items():
                pair_roles.setdefault(_reverse_key(sequence), set()).add(role_by_id[sequence_id])
            split_pairs = [pair for pair, pair_role in pair_roles.items() if len(pair_role) != 1]
            if split_pairs:
                raise ValueError(
                    "forward/reverse endpoint pairs cross roles in pair_disjoint_stress_test"
                )
        train_text, validation_text, test_text = descriptions[name]
        records.append(
            {
                "class": "transient",
                "name": name,
                "label": labels[name],
                "train_count": len(roles["train"]),
                "validation_count": len(roles["validation"]),
                "test_count": len(roles["test"]),
                "train_description": f"{train_text} ({len(roles['train'])})",
                "validation_description": f"{validation_text} ({len(roles['validation'])})",
                "test_description": f"{test_text} ({len(roles['test'])})",
                "role_sequence_ids": roles,
            }
        )
    return records


def render_table(steady: list[dict[str, object]], transient: list[dict[str, object]]) -> str:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\caption{Pre-defined physical data splits. A steady condition is one complete three-dimensional field at a fixed inlet velocity, inlet temperature and pebble heat source. A transient sample is one complete staged-time trajectory; no time point from a trajectory is assigned to another role. The endpoint-pair split also keeps the forward and reverse trajectories joining the same two steady conditions together. Normalization and checkpoint selection use training and validation data only. Exact condition and trajectory identifiers are stored in the cited machine-readable split files.}",
        r"\label{tab:physical_data_splits}",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{2.6cm}>{\raggedright\arraybackslash}p{3.2cm}>{\raggedright\arraybackslash}p{3.2cm}>{\raggedright\arraybackslash}p{3.2cm}}",
        r"\toprule",
        r"Physical test & Training & Validation & Independent test \\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Steady three-dimensional fields}} \\",
    ]
    for record in steady:
        lines.append(
            "{} & {} & {} & {} \\\\".format(
                record["label"],
                record["train_description"],
                record["validation_description"],
                record["test_description"],
            )
        )
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{4}{l}{\textit{Complete thermal-step trajectories}} \\",
        ]
    )
    for record in transient:
        lines.append(
            "{} & {} & {} & {} \\\\".format(
                record["label"],
                record["train_description"],
                record["validation_description"],
                record["test_description"],
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steady-splits", type=Path, required=True)
    parser.add_argument("--transient-splits", type=Path, required=True)
    parser.add_argument("--transient-plan", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    steady_path = args.steady_splits.resolve()
    transient_path = args.transient_splits.resolve()
    plan_path = args.transient_plan.resolve()
    steady_payload = read_json(steady_path)
    transient_payload = read_json(transient_path)
    plan_payload = read_json(plan_path)
    steady = collect_steady_records(steady_payload)
    transient = collect_transient_records(transient_payload, plan_payload)

    tex_output = args.tex_output.resolve()
    tex_output.parent.mkdir(parents=True, exist_ok=True)
    tex_output.write_text(render_table(steady, transient), encoding="utf-8")
    summary = {
        "status": "completed_p418_physical_data_split_table",
        "source_parameter_id": steady_payload.get("source_parameter_id"),
        "source_doi": steady_payload.get("source_doi"),
        "steady_condition_count": len(steady_payload["conditions"]),
        "transient_trajectory_count": len(plan_payload["sequences"]),
        "steady_splits": steady,
        "transient_splits": transient,
        "input_files": [str(steady_path), str(transient_path), str(plan_path)],
        "tex_output": str(tex_output),
        "new_physical_parameters": [],
    }
    summary_output = args.summary_output.resolve()
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
