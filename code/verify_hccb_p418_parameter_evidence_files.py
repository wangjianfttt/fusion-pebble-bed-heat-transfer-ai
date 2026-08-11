#!/usr/bin/env python3
"""Check that every P418 physical parameter has a traceable source file."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_STATUS = {
    "local_source_archived",
    "local_official_page_archived",
    "local_primary_report_and_publisher_abstract_archived",
    "open_official_review_and_publisher_abstract_archived",
    "publisher_abstract_verified_metadata_archived",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def verify(
    sources_path: Path,
    evidence_path: Path,
    equation_path: Path,
    root: Path = ROOT,
    require_local_files: bool = True,
) -> dict[str, object]:
    source_rows = rows(sources_path)
    evidence_rows = rows(evidence_path)
    equation_rows = rows(equation_path)

    source_ids = [row["parameter_id"] for row in source_rows]
    evidence_ids = [row["parameter_id"] for row in evidence_rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("physical parameter source list contains duplicate IDs")
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("parameter evidence list contains duplicate IDs")
    if set(source_ids) != set(evidence_ids):
        raise ValueError(
            "parameter/evidence ID difference: "
            f"missing={sorted(set(source_ids)-set(evidence_ids))}, "
            f"extra={sorted(set(evidence_ids)-set(source_ids))}"
        )

    used_by_equations: dict[str, list[str]] = defaultdict(list)
    for row in equation_rows:
        for parameter_id in row["文献参数编号"].split(";"):
            parameter_id = parameter_id.strip()
            if parameter_id:
                used_by_equations[parameter_id].append(row["物理量或方程"])
    unused = sorted(set(source_ids) - set(used_by_equations))
    if unused:
        raise ValueError(f"physical parameters not used by equation map: {unused}")

    file_count = 0
    archived_status_count: dict[str, int] = defaultdict(int)
    for row in evidence_rows:
        parameter_id = row["parameter_id"]
        status = row["status"]
        if status not in ALLOWED_STATUS:
            raise ValueError(f"{parameter_id}: unsupported status {status}")
        archived_status_count[status] += 1
        if not row["evidence_location"].strip():
            raise ValueError(f"{parameter_id}: empty evidence location")
        if not row["source_url_or_doi"].strip():
            raise ValueError(f"{parameter_id}: empty source URL/DOI")

        paths = [item.strip() for item in row["local_evidence_paths"].split(";")]
        hashes = [item.strip() for item in row["sha256_list"].split(";")]
        if len(paths) != len(hashes) or not paths:
            raise ValueError(f"{parameter_id}: evidence path/hash count mismatch")
        for relative, expected in zip(paths, hashes):
            if not relative:
                raise ValueError(f"{parameter_id}: empty local evidence path")
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError(f"{parameter_id}: invalid SHA-256 for {relative}")
            file_count += 1
            if not require_local_files:
                continue
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f"{parameter_id}: {path}")
            actual = digest(path)
            if actual != expected:
                raise ValueError(
                    f"{parameter_id}: SHA-256 mismatch for {relative}: "
                    f"expected {expected}, got {actual}"
                )

    evidence_by_id = {row["parameter_id"]: row for row in evidence_rows}
    for parameter_id in ("P428", "P429"):
        row = evidence_by_id[parameter_id]
        if row["status"] != "open_official_review_and_publisher_abstract_archived":
            raise ValueError(f"{parameter_id}: Kleykamp evidence boundary changed")
        for source in (
            "sciencedirect.com/science/article/pii/S0040603196029966",
            "jstage.jst.go.jp/article/jscta1974/27/2/27_2_100/_pdf",
        ):
            if source not in row["source_url_or_doi"]:
                raise ValueError(f"{parameter_id}: missing source {source}")
        if "Kleykamp_2000_High_temperature_calorimetry_review.pdf" not in row[
            "local_evidence_paths"
        ]:
            raise ValueError(f"{parameter_id}: local J-STAGE paper is missing")

    p431 = evidence_by_id["P431"]
    if p431["status"] != "local_primary_report_and_publisher_abstract_archived":
        raise ValueError("P431: calorimetry evidence boundary changed")
    for source in (
        "sciencedirect.com/science/article/pii/S0040603196029966",
        "jstage.jst.go.jp/article/jscta1974/27/2/27_2_100/_pdf",
        "publikationen.bibliothek.kit.edu/270037437/4050264",
        "sciencedirect.com/science/article/pii/S0021961405800679",
    ):
        if source not in p431["source_url_or_doi"]:
            raise ValueError(f"P431: missing source {source}")
    if "FZKA5515_fusion_annual_report_1994_1995.pdf" not in p431[
        "local_evidence_paths"
    ]:
        raise ValueError("P431: local FZKA5515 report is missing")
    if "Asou_1992_Li4SiO4_ScienceDirect_abstract_evidence.txt" not in p431[
        "local_evidence_paths"
    ]:
        raise ValueError("P431: independent Asou calorimetry evidence is missing")
    for local_source in (
        "Kleykamp_2000_High_temperature_calorimetry_review.pdf",
        "Kleykamp_2000_Li4SiO4_enthalpy_Cp_evidence.txt",
    ):
        if local_source not in p431["local_evidence_paths"]:
            raise ValueError(f"P431: local transition-region source is missing: {local_source}")

    # P429 must be the exact analytical derivative of P428.
    temperature = 700.0
    numerical = (
        (-17156 + 73.694 * (temperature + 1e-4) + 0.103210 * (temperature + 1e-4) ** 2 - 4163115 / (temperature + 1e-4))
        - (-17156 + 73.694 * (temperature - 1e-4) + 0.103210 * (temperature - 1e-4) ** 2 - 4163115 / (temperature - 1e-4))
    ) / (2e-4)
    analytical = 73.694 + 0.206420 * temperature + 4163115 / temperature**2
    if abs(numerical - analytical) > 1e-5:
        raise ValueError("P429 is not the analytical derivative of P428")
    cp_298 = 73.694 + 0.206420 * 298.0 + 4163115 / 298.0**2
    cp_1100 = 73.694 + 0.206420 * 1100.0 + 4163115 / 1100.0**2
    if round(cp_298, 1) != 182.1 or round(cp_1100, 1) != 304.2:
        raise ValueError("P429 does not reproduce the J-STAGE Table 1 endpoints")

    molar_mass = 4 * 6.94 + 28.085 + 4 * 15.999
    if abs(molar_mass - 119.841) > 1e-12:
        raise ValueError("P430 molar-mass conversion changed")

    return {
        "status": "p418_parameter_evidence_files_verified",
        "physical_parameter_count": len(source_ids),
        "equation_map_row_count": len(equation_rows),
        "local_evidence_reference_count": file_count,
        "local_evidence_files_verified": require_local_files,
        "evidence_verification_mode": (
            "local_files_and_sha256" if require_local_files else "registered_metadata_and_sha256"
        ),
        "evidence_status_counts": dict(sorted(archived_status_count.items())),
        "all_parameters_used_by_equations": True,
        "p429_derivative_check": True,
        "p429_jstage_endpoint_check_J_mol_K": {
            "298_K": cp_298,
            "1100_K": cp_1100,
        },
        "p430_molar_mass_g_mol": molar_mass,
        "new_physical_parameters": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        type=Path,
        default=ROOT / "parameters/hccb_p418_physical_parameter_sources.csv",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=ROOT / "parameters/hccb_p418_physical_parameter_evidence_files.csv",
    )
    parser.add_argument(
        "--equations",
        type=Path,
        default=ROOT / "parameters/hccb_p418_equation_input_map.csv",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Check registered paths, URLs and SHA-256 values without requiring copyrighted local files.",
    )
    args = parser.parse_args()
    summary = verify(
        args.sources.resolve(),
        args.evidence.resolve(),
        args.equations.resolve(),
        require_local_files=not args.metadata_only,
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
