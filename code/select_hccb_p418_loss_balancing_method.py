#!/usr/bin/env python3
"""Select one P418 loss-balancing candidate using validation curves only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _candidate_weights(candidate: dict[str, object]) -> dict[str, float]:
    """Return registered weights under the names written to model summaries."""
    if "temperature_data_weight" in candidate:
        source_names = {
            "temperature_data": "temperature_data_weight",
            "reference_edge_energy_flux": "reference_edge_energy_flux_weight",
            "projection_aware_transient_energy": (
                "projection_aware_transient_energy_weight"
            ),
        }
    else:
        source_names = {
            "temperature_data": "state_weight",
            "reference_edge_energy_flux": "face_flux_weight",
            "projection_aware_transient_energy": "physics_weight",
        }
    try:
        return {
            summary_name: float(candidate[source_name])
            for summary_name, source_name in source_names.items()
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("loss-balancing candidate has incomplete weights") from error


def _verified_candidate_configuration(
    candidate: dict[str, object], summary: dict[str, object]
) -> dict[str, object]:
    """Check that a completed run used the candidate declared in the source JSON."""
    balancing = summary.get("loss_balancing")
    weights = summary.get("loss_weights")
    if not isinstance(balancing, dict) or not isinstance(weights, dict):
        raise ValueError("selection summary does not record its loss configuration")

    method = candidate.get("method")
    if balancing.get("method") != method:
        raise ValueError("selection summary loss method differs from its source JSON")
    expected_weights = _candidate_weights(candidate)
    try:
        observed_weights = {
            name: float(weights[name]) for name in expected_weights
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("selection summary has incomplete loss weights") from error
    if observed_weights != expected_weights:
        raise ValueError("selection summary loss weights differ from its source JSON")

    checkpoint_state = balancing.get("selected_checkpoint_state")
    if not isinstance(checkpoint_state, dict):
        raise ValueError("selection summary has no loss-balancer checkpoint state")
    if checkpoint_state.get("method") != method:
        raise ValueError("loss-balancer checkpoint method differs from its source JSON")
    if method == "fixed":
        expected_ordered = [
            expected_weights["temperature_data"],
            expected_weights["reference_edge_energy_flux"],
            expected_weights["projection_aware_transient_energy"],
        ]
        try:
            observed_ordered = [float(value) for value in checkpoint_state["weights"]]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("fixed loss-balancer checkpoint has invalid weights") from error
        if observed_ordered != expected_ordered:
            raise ValueError("fixed loss-balancer checkpoint differs from its source JSON")
    elif method == "relobralo":
        for name in ("temperature", "alpha", "expected_rho"):
            try:
                observed = float(checkpoint_state[name])
                expected = float(candidate[name])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"ReLoBRaLo checkpoint has invalid {name}"
                ) from error
            if observed != expected:
                raise ValueError(
                    f"ReLoBRaLo checkpoint {name} differs from its source JSON"
                )
    else:
        raise ValueError(f"unsupported loss-balancing method: {method!r}")

    return {
        "method": method,
        "initial_weights": expected_weights,
        "temperature": candidate.get("temperature"),
        "alpha": candidate.get("alpha"),
        "expected_rho": candidate.get("expected_rho"),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_root = args.candidate_root.resolve()
    sources_path = args.sources.resolve()
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    candidates = sources["formal_candidates"]
    declared = [row["candidate_id"] for row in candidates]
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    if len(candidate_by_id) != len(declared):
        raise ValueError("loss-balancing source contains duplicate candidate ids")
    records: list[dict[str, object]] = []
    common: dict[str, object] | None = None
    for candidate_id in declared:
        summary_path = candidate_root / candidate_id / "selection_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("evaluation_stage") != "selection":
            raise ValueError(f"{candidate_id} was not run in validation-only selection mode")
        if summary.get("test_evaluated") or "test" in summary.get("metrics", {}):
            raise ValueError(f"{candidate_id} read the independent test curves too early")
        if summary["loss_balancing"]["candidate_id"] != candidate_id:
            raise ValueError(f"{candidate_id} summary identifies another candidate")
        verified_configuration = _verified_candidate_configuration(
            candidate_by_id[candidate_id], summary
        )
        comparable = {
            "dataset_index": summary["dataset_index"],
            "input_file_sha256": summary["input_file_sha256"],
            "split_name": summary["split_name"],
            "split_sequence_ids": summary["split_sequence_ids"],
            "seed": summary["seed"],
            "architecture": summary["architecture"],
            "physics_terms": summary["physics_terms"],
            "training_normalization_sequence_ids": summary[
                "training_normalization_sequence_ids"
            ],
        }
        if common is None:
            common = comparable
        elif comparable != common:
            raise ValueError("loss-balancing candidates do not share one data/model setting")
        records.append(
            {
                "candidate_id": candidate_id,
                "validation_selection_score": float(
                    summary["best_validation_selection_score"]
                ),
                "best_epoch": int(summary["best_epoch"]),
                "summary_path": str(summary_path),
                "summary_sha256": sha256(summary_path),
                "verified_configuration": verified_configuration,
            }
        )

    selected = min(
        records,
        key=lambda row: (
            row["validation_selection_score"],
            row["candidate_id"],
        ),
    )
    output = {
        "status": "p418_loss_balancing_selected_on_validation_only",
        "selection_metric": (
            "equal mean of dimensionless state, face-flux and physics validation groups"
        ),
        "source_file": str(sources_path),
        "source_file_sha256": sha256(sources_path),
        "candidate_records": records,
        "selected_candidate_id": selected["candidate_id"],
        "selected_validation_score": selected["validation_selection_score"],
        "selected_summary_path": selected["summary_path"],
        "selected_summary_sha256": selected["summary_sha256"],
        "common_data_and_model_setting": common,
        "independent_test_read": False,
        "next_step": (
            "Resume only the selected candidate with --evaluation-stage final and "
            "--selected-method-record pointing to this file."
        ),
        "new_physical_parameters": [],
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
