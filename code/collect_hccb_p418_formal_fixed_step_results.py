#!/usr/bin/env python3
"""Combine the 12 independently solved P418 fixed-flow transient histories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        return {name: payload[name] for name in payload.files}


def combine(work_root: Path, plan_path: Path, output_dir: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    sequences = plan["sequences"]
    loaded = []
    markers = []
    csv_paths = []
    for sequence in sequences:
        sequence_id = str(sequence["sequence_id"])
        task_root = work_root / "by_sequence" / sequence_id
        marker_path = task_root / "cloud_sequence_complete.json"
        artifact = task_root / "results/hccb_p418_transient_observables.npz"
        long_csv = task_root / "results/hccb_p418_transient_observables_long.csv"
        if not marker_path.is_file() or not artifact.is_file() or not long_csv.is_file():
            raise FileNotFoundError(f"incomplete formal sequence: {sequence_id}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("status") != "completed_p418_formal_fixed_hydrodynamics_sequence":
            raise ValueError(f"unexpected marker status for {sequence_id}")
        if marker["observable_artifact_sha256"] != sha256(artifact):
            raise ValueError(f"observable checksum mismatch for {sequence_id}")
        loaded.append(load_npz(artifact))
        markers.append(marker)
        csv_paths.append(long_csv)

    reference = loaded[0]
    for payload in loaded[1:]:
        for name in ("condition_names", "signal_names"):
            if not np.array_equal(payload[name], reference[name]):
                raise ValueError(f"inconsistent {name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    max_steps = max(payload["time_s"].shape[1] for payload in loaded)
    case_count = len(loaded)
    signal_count = reference["values"].shape[2]
    condition_count = reference["conditions"].shape[1]
    case_ids = np.empty(case_count, dtype=object)
    complete = np.zeros(case_count, dtype=bool)
    conditions = np.zeros((case_count, condition_count), dtype="float64")
    time_s = np.full((case_count, max_steps), np.nan, dtype="float64")
    time_mask = np.zeros((case_count, max_steps), dtype=bool)
    values = np.full(
        (case_count, max_steps, signal_count),
        np.nan,
        dtype="float64",
    )
    for index, payload in enumerate(loaded):
        count = payload["time_s"].shape[1]
        case_ids[index] = payload["case_id"][0]
        complete[index] = bool(payload["complete"][0])
        conditions[index] = payload["conditions"][0]
        time_s[index, :count] = payload["time_s"][0]
        time_mask[index, :count] = payload["time_mask"][0]
        values[index, :count] = payload["values"][0]

    artifact = output_dir / "hccb_p418_formal_fixed_step_observables.npz"
    np.savez_compressed(
        artifact,
        case_id=case_ids,
        complete=complete,
        conditions=conditions,
        condition_names=reference["condition_names"],
        time_s=time_s,
        time_mask=time_mask,
        values=values,
        signal_names=reference["signal_names"],
    )

    combined_csv = output_dir / "hccb_p418_formal_fixed_step_observables_long.csv"
    fieldnames = None
    with combined_csv.open("w", newline="", encoding="utf-8") as output:
        writer = None
        for path in csv_paths:
            with path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                if fieldnames is None:
                    fieldnames = reader.fieldnames
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                elif reader.fieldnames != fieldnames:
                    raise ValueError(f"inconsistent CSV columns: {path}")
                assert writer is not None
                writer.writerows(reader)

    summary = {
        "status": "completed_p418_formal_fixed_hydrodynamics_matrix",
        "sequence_count": case_count,
        "completed_sequence_count": int(complete.sum()),
        "duration_s": 300.0,
        "observable_artifact": str(artifact),
        "observable_artifact_bytes": artifact.stat().st_size,
        "observable_artifact_sha256": sha256(artifact),
        "long_csv": str(combined_csv),
        "long_csv_sha256": sha256(combined_csv),
        "sequence_markers": markers,
        "new_physical_parameters": [],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "FORMAL_FIXED_STEP_MATRIX_COMPLETE.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = combine(
        args.work_root.resolve(),
        args.plan.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
