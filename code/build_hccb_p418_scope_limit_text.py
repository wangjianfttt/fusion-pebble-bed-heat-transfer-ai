#!/usr/bin/env python3
"""Build concise manuscript text from verified P418 scope-limit records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_RECORDS = {
    "MESH_PREFLIGHT_FAILED.json",
    "mesh_audit_summary.json",
    "maxCo_0p8_14721_failure.json",
    "maxCo_0p4_14722_failure.json",
    "maxCo_0p2_14723_failure.json",
    "representative_velocity_up_14724_failure.json",
}


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_transport_check(path: Path) -> None:
    record = load_json(path)
    if record.get("status") != (
        "hccb_helium_transport_openfoam13_build_and_pointwise_check_passed"
    ):
        raise ValueError("direct helium-transport check has an unexpected status")
    if record.get("solver_started") is not False:
        raise ValueError("the direct helium-transport check unexpectedly ran a solver")
    if record.get("physical_correlations_changed") is not False:
        raise ValueError("the direct helium-transport check changed the correlations")
    pointwise = record.get("pointwise_check", {})
    if (
        pointwise.get("all_values_positive_and_finite") is not True
        or int(pointwise.get("point_count", 0)) < 12
        or float(pointwise.get("maximum_mu_relative_error", 1.0)) > 1.0e-12
        or float(pointwise.get("maximum_kappa_relative_error", 1.0)) > 1.0e-12
    ):
        raise ValueError("the direct helium-transport pointwise check did not pass")


def validate_direct_coupled_failure(path: Path) -> dict:
    record = load_json(path)
    failure = record.get("failure", {})
    slurm = record.get("slurm", {})
    contract = record.get("common_complete_file_contract", {})
    required_fields = {
        "fluid/T",
        "fluid/U",
        "fluid/p",
        "fluid/p_rgh",
        "fluid/phi",
        "solid/T",
        "uniform/time",
    }
    if record.get("status") not in {
        "p418_matched_initial_direct_transport_representative_smoke_failed_scope_limit",
        "p418_matched_initial_direct_transport_representative_scope_limit_public",
    }:
        raise ValueError("matched-initial direct-transport record has an unexpected status")
    if record.get("sequence_id") != "source_up_u0p15_T700":
        raise ValueError("matched-initial direct-transport sequence is unexpected")
    if slurm.get("state") != "FAILED" or slurm.get("exit_code") != "1:0":
        raise ValueError("matched-initial direct-transport job did not record its failure")
    if (
        failure.get("type") != "solid_Cv_nonuniform_table_temperature_out_of_range"
        or failure.get("foam_fatal") is not True
        or failure.get("mpi_abort") is not True
        or failure.get("nan") is not False
        or failure.get("nonpositive_transport_input") is not False
        or failure.get("segmentation_fault") is not False
    ):
        raise ValueError("matched-initial direct-transport failure type is unexpected")
    query_temperature = float(failure["query_temperature_K"])
    upper_limit = float(failure["table_upper_limit_K"])
    if query_temperature <= upper_limit:
        raise ValueError("solid heat-capacity lookup did not exceed its upper limit")
    if (
        int(contract.get("mpi_ranks", 0)) != 32
        or set(contract.get("files_per_rank", [])) != required_fields
        or int(contract.get("verified_file_count", 0)) != 32 * len(required_fields)
    ):
        raise ValueError("matched-initial direct-transport common field set is incomplete")
    if (
        float(record["last_32_rank_common_complete_time_s"])
        >= float(record["last_logged_physical_time_s"])
        or float(record["last_logged_physical_time_s"]) >= 0.01
        or record.get("openfoam_end_present") is not False
        or record.get("completion_marker_present") is not False
        or record.get("observable_export_summary_present") is not False
        or int(record.get("observable_signal_count", -1)) != 0
        or record.get("automatic_timeout_continuation_allowed") is not False
    ):
        raise ValueError("matched-initial direct-transport completion status is inconsistent")
    return record


def build(
    summary_path: Path,
    transport_check_path: Path | None = None,
    direct_coupled_failure_path: Path | None = None,
) -> str:
    summary = load_json(summary_path)
    if summary.get("status") != "P418_SCOPE_LIMITS_EVIDENCE_SYNCED":
        raise ValueError("scope-limit summary has an unexpected status")

    records = summary.get("records")
    if not isinstance(records, list):
        raise ValueError("scope-limit summary does not contain a record list")
    by_name = {str(row.get("filename")): row for row in records}
    if set(by_name) != EXPECTED_RECORDS:
        raise ValueError(f"scope-limit record set is incomplete: {set(by_name)}")

    root = summary_path.parent
    for name, row in by_name.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != row.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {path}")
        if row.get("slurm_state") != "FAILED":
            raise ValueError(f"{name} is not a recorded failed calculation")

    mesh_failure = load_json(root / "MESH_PREFLIGHT_FAILED.json")
    mesh_audit = load_json(root / "mesh_audit_summary.json")
    if mesh_failure.get("steady_solver_started") is not False:
        raise ValueError("the full-domain steady solver-start flag is not false")
    if int(mesh_audit.get("fluid_checkMesh_exit_code", 0)) == 0:
        raise ValueError("the recorded full-domain fluid checkMesh did not fail")

    cfl_times: list[float] = []
    for name in (
        "maxCo_0p8_14721_failure.json",
        "maxCo_0p4_14722_failure.json",
        "maxCo_0p2_14723_failure.json",
    ):
        record = load_json(root / name)
        if (
            record.get("status") != "failed_solver_exit_propagated"
            or record.get("has_property_range_error") is not True
            or record.get("has_nan") is not False
        ):
            raise ValueError(f"{name} is not the expected property-range failure")
        cfl_times.append(float(record["last_logged_physical_time_s"]))

    representative = load_json(
        root / "representative_velocity_up_14724_failure.json"
    )
    pressure_min = float(representative["logged_pressure_min_pa"])
    pressure_limit = float(representative["helium_pressure_range_pa"][0])
    if (
        representative.get("error_scan", {}).get("property_range_error") is not True
        or pressure_min >= pressure_limit
    ):
        raise ValueError("the representative startup does not prove a low-pressure exit")

    text = (
        "A separate full-domain fluid mesh failed the ordinary \\texttt{checkMesh} "
        "test, so no solver was run on that mesh and no full-domain result is "
        "claimed. This limitation is distinct from the accepted local three-grid "
        "sensitivity study reported above. On the accepted local mesh, fixed-flow "
        "medium-to-fine thermal-curve differences remained below 0.043\\%. Three "
        "fully coupled Courant-number tests stopped between "
        f"\\SI{{{min(cfl_times):.6f}}}{{s}} and "
        f"\\SI{{{max(cfl_times):.6f}}}{{s}} after pressure left the helium-property "
        f"range; a run with consistent inlet velocity and mass flux similarly reached "
        f"\\SI{{{pressure_min:.1f}}}{{Pa}}, below its "
        f"\\SI{{{pressure_limit:.0f}}}{{Pa}} limit. Model claims therefore concern "
        "thermal evolution with a prescribed hydrodynamic field."
    )
    if transport_check_path is not None:
        validate_transport_check(transport_check_path)
        text += (
            " A direct OpenFOAM implementation of the same helium transport "
            "correlations was verified pointwise without a flow solve."
        )
    if direct_coupled_failure_path is not None:
        direct_failure = validate_direct_coupled_failure(direct_coupled_failure_path)
        failure = direct_failure["failure"]
        text += (
            " A matched-initial representative using that implementation advanced "
            f"to \\SI{{{float(direct_failure['last_logged_physical_time_s']):.6f}}}{{s}} "
            "before a solid heat-capacity query reached "
            f"\\SI{{{float(failure['query_temperature_K']):.1f}}}{{K}}, above the "
            f"registered \\SI{{{float(failure['table_upper_limit_K']):.0f}}}{{K}} "
            "limit; this startup is used only to delimit property validity, not as "
            "fully coupled accuracy evidence."
        )
    elif transport_check_path is not None:
        text += " No matched fully coupled result is claimed."
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--transport-check", type=Path)
    parser.add_argument("--direct-coupled-failure", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    text = build(
        args.summary,
        args.transport_check,
        args.direct_coupled_failure,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
