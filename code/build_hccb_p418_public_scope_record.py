#!/usr/bin/env python3
"""Create a path-free public record for the representative scope-limit run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PRIVATE_TOKENS = (
    "/" + "Users/",
    "/" + "data2/",
    "/" + "n96pfs/",
    "192" + ".168.",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(source: Path, output: Path) -> dict[str, object]:
    record = json.loads(source.read_text(encoding="utf-8"))
    failure = record.get("failure", {})
    contract = record.get("common_complete_file_contract", {})
    if record.get("status") != (
        "p418_matched_initial_direct_transport_representative_smoke_failed_scope_limit"
    ):
        raise ValueError("unexpected representative scope-limit status")
    if (
        record.get("sequence_id") != "source_up_u0p15_T700"
        or record.get("slurm", {}).get("state") != "FAILED"
        or failure.get("type")
        != "solid_Cv_nonuniform_table_temperature_out_of_range"
        or float(failure["query_temperature_K"])
        <= float(failure["table_upper_limit_K"])
        or int(contract.get("mpi_ranks", 0)) != 32
        or int(contract.get("verified_file_count", 0)) != 224
        or record.get("completion_marker_present") is not False
        or record.get("observable_export_summary_present") is not False
    ):
        raise ValueError("representative scope-limit record is inconsistent")

    payload: dict[str, object] = {
        "status": (
            "p418_matched_initial_direct_transport_representative_scope_limit_public"
        ),
        "source_record_sha256": sha256(source),
        "job_id": int(record["job_id"]),
        "sequence_id": record["sequence_id"],
        "scientific_scope": record["scientific_scope"],
        "slurm": {
            "state": record["slurm"]["state"],
            "exit_code": record["slurm"]["exit_code"],
            "elapsed": record["slurm"]["elapsed"],
        },
        "last_logged_physical_time_s": record["last_logged_physical_time_s"],
        "last_logged_max_courant": record["last_logged_max_courant"],
        "last_32_rank_common_complete_time_s": record[
            "last_32_rank_common_complete_time_s"
        ],
        "common_complete_file_contract": contract,
        "failure": failure,
        "openfoam_end_present": record["openfoam_end_present"],
        "completion_marker_present": record["completion_marker_present"],
        "observable_export_summary_present": record[
            "observable_export_summary_present"
        ],
        "observable_signal_count": record["observable_signal_count"],
        "automatic_timeout_continuation_allowed": record[
            "automatic_timeout_continuation_allowed"
        ],
        "new_physical_parameters": [],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if any(token in text for token in PRIVATE_TOKENS):
        raise ValueError("private machine path leaked into public scope record")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.source, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
