#!/usr/bin/env python3
"""Summarize legacy unbounded-temperature checkpoints as a negative control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_STATUS = "p418_checkpoint_physical_domain_diagnosis"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def minimum_record(records: list[dict], field: str) -> tuple[float, dict]:
    candidates: list[tuple[float, dict]] = []
    for record in records:
        value = float(record[field]["value"])
        if not math.isfinite(value):
            raise ValueError(f"non-finite {field} in diagnostic record")
        candidates.append((value, record))
    if not candidates:
        raise ValueError("diagnostic contains no sequence records")
    return min(candidates, key=lambda item: item[0])


def summarize(label: str, path: Path) -> dict[str, object]:
    payload = load(path)
    if payload.get("status") != EXPECTED_STATUS:
        raise ValueError(f"{label} has an unexpected diagnostic status")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label} has no diagnostic records")

    fluid_min, fluid_record = minimum_record(records, "fluid_temperature_K")
    solid_min, solid_record = minimum_record(records, "solid_temperature_K")
    pressure_min, pressure_record = minimum_record(
        records, "fluid_absolute_pressure_Pa"
    )
    nonpositive_fluid_count = sum(
        int(record["fluid_temperature_K"]["nonpositive_count"])
        for record in records
    )
    return {
        "model": label,
        "completed_epochs": int(payload["completed_epochs"]),
        "minimum_fluid_temperature_K": fluid_min,
        "minimum_solid_temperature_K": solid_min,
        "minimum_absolute_pressure_Pa": pressure_min,
        "nonpositive_fluid_temperature_count": nonpositive_fluid_count,
        "minimum_fluid_temperature_role": fluid_record["role"],
        "minimum_fluid_temperature_sequence": fluid_record["sequence_id"],
        "minimum_solid_temperature_role": solid_record["role"],
        "minimum_solid_temperature_sequence": solid_record["sequence_id"],
        "minimum_pressure_role": pressure_record["role"],
        "minimum_pressure_sequence": pressure_record["sequence_id"],
        "source": str(path.resolve()),
        "source_sha256": sha256(path),
    }


def latex_table(rows: list[dict[str, object]]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Diagnostic results from superseded graph--Transformer checkpoints with unbounded temperature outputs. These checkpoints are excluded from the final model comparison. The pressure channel remained positive, whereas the learned temperature channels left the specified thermophysical intervals; the physics-constrained checkpoint produced four non-positive fluid-temperature values.}",
        r"\label{tab:unbounded_output_diagnostic}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Checkpoint & Epochs & Min. $T_f$ (K) & Min. $T_s$ (K) & Min. $p_{\mathrm{abs}}$ (Pa) & $T_f\leq0$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['model']} & {int(row['completed_epochs'])} & "
            f"{float(row['minimum_fluid_temperature_K']):.3f} & "
            f"{float(row['minimum_solid_temperature_K']):.3f} & "
            f"{float(row['minimum_absolute_pressure_Pa']):.1f} & "
            f"{int(row['nonpositive_fluid_temperature_count'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-only", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        summarize("Data-only", args.data_only.resolve()),
        summarize("Physics-constrained", args.physics.resolve()),
    ]
    physics = rows[1]
    if int(physics["nonpositive_fluid_temperature_count"]) <= 0:
        raise ValueError(
            "physics negative control does not contain a non-positive fluid temperature"
        )
    if any(float(row["minimum_absolute_pressure_Pa"]) <= 0.0 for row in rows):
        raise ValueError("absolute pressure must remain positive in this diagnostic")

    output = {
        "status": "completed_p418_unbounded_output_diagnostic",
        "formal_model_ranking_included": False,
        "scientific_interpretation": (
            "The legacy failure was caused by an unconstrained learned-temperature "
            "output, not by non-positive absolute pressure or non-finite reference data."
        ),
        "records": rows,
        "new_physical_parameters": [],
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.tex_output.parent.mkdir(parents=True, exist_ok=True)
    args.tex_output.write_text(latex_table(rows), encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
