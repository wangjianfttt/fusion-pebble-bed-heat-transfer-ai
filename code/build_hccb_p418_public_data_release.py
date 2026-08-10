#!/usr/bin/env python3
"""Describe the public P418 data release without inventing pending results."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from check_hccb_p418_ijhmt_submission import (
    extract_environment,
    extract_title,
    final_abstract,
    strip_latex,
)


COMPACT_FILES = (
    "results/hccb_p418_public_figure_data/README.md",
    "results/hccb_p418_public_figure_data/physical_response_60.csv",
    "results/hccb_p418_public_figure_data/seed202_integral_comparison_9.csv",
    "results/hccb_p418_public_figure_data/seed202_integral_summary.json",
    "results/hccb_p418_public_figure_data/steady_model_comparison_5x5.csv",
    "results/hccb_p418_public_figure_data/summary.json",
    "results/hccb_p418_public_figure_data/direct_transport_scope_limit.json",
    "results/hccb_p418_three_mesh_cht_sensitivity/summary.json",
    "results/hccb_p418_three_mesh_cht_sensitivity/engineering_observables.csv",
    "results/hccb_p418_three_mesh_cht_sensitivity/mesh_gci.csv",
    "results/hccb_p418_public_data_release_preflight/formal_training_manifest_public.json",
)

FINAL_PROCESSED_FILES = (
    "results/hccb_p418_physical_steps_12/regional_sequences/regional_sequence_geometry.npz",
    "data/apd006_hccb_source_sequence_target_packings/seed101_s80_xlo_ycentre/packing.npz",
    "results/hccb_p418_physical_steps_12/model_comparison/summary.json",
    "results/hccb_p418_physical_steps_12/model_comparison/physical_step_model_metrics.csv",
    "results/hccb_p418_physical_steps_12/model_comparison/physical_step_model_speedup.csv",
    "results/hccb_p418_physical_steps_12/fixed_flow_loss_balancing_pair_disjoint_stress_test/selected_downstream_integration.json",
    "results/hccb_p418_physical_steps_12/regional_persistence_pair_disjoint_stress_test/test_temperature_predictions.npz",
    "results/hccb_p418_physical_steps_12/regional_graph_transformer_bounded_data_only_pair_disjoint_stress_test/test_temporal_temperature_predictions.npz",
)

PRIVATE_TEXT = (
    "/" + "Users/",
    "/" + "data2/",
    "/" + "n96pfs/",
    "192" + ".168.",
)

CREATORS = (
    {
        "name": "Wang, Jian",
        "affiliation": (
            "Anhui University of Science and Technology; Institute of Plasma "
            "Physics, Hefei Institutes of Physical Science, Chinese Academy of "
            "Sciences"
        ),
    },
    {
        "name": "Wen, Wei",
        "affiliation": (
            "Anhui University of Science and Technology; Institute of Plasma "
            "Physics, Hefei Institutes of Physical Science, Chinese Academy of "
            "Sciences"
        ),
    },
    {"name": "Shen, Gang", "affiliation": "Anhui University of Science and Technology"},
    {
        "name": "Lei, Mingzhun",
        "affiliation": (
            "Institute of Plasma Physics, Hefei Institutes of Physical Science, "
            "Chinese Academy of Sciences"
        ),
    },
    {
        "name": "Song, Yuntao",
        "affiliation": (
            "Institute of Plasma Physics, Hefei Institutes of Physical Science, "
            "Chinese Academy of Sciences"
        ),
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(project_root: Path, relative: str) -> dict[str, object]:
    path = project_root / relative
    present = path.is_file() and path.stat().st_size > 0
    row: dict[str, object] = {"path": relative, "present": present}
    if present:
        row.update({"size_bytes": path.stat().st_size, "sha256": sha256(path)})
    return row


def license_choice(project_root: Path) -> dict[str, str] | None:
    path = project_root / "submission/data_release_license_choice.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    software = str(payload.get("software_license") or "").strip()
    data = str(payload.get("data_license") or "").strip()
    if not software or not data or "pending" in f"{software} {data}".lower():
        return None
    return {"software_license": software, "data_license": data}


def repository_record(project_root: Path) -> dict[str, object]:
    path = project_root / "submission/data_release_repository_record.json"
    if not path.is_file():
        return {"repository_url": None, "repository_doi": None}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "repository_url": str(payload.get("repository_url") or "").strip() or None,
        "repository_doi": str(payload.get("repository_doi") or "").strip() or None,
    }


def expand_generated_manuscript_values(project_root: Path, text: str) -> str:
    """Expand scalar LaTeX macros before converting the abstract to plain text."""
    values = project_root / "manuscript/generated_results.tex"
    if not values.is_file():
        return text
    macros = dict(
        re.findall(
            r"\\newcommand\{\\([A-Za-z]+)\}\{([^{}]*)\}",
            values.read_text(encoding="utf-8"),
        )
    )
    for name, value in macros.items():
        if value.isdigit() and int(value) >= 1000:
            value = f"{int(value):,}"
        text = text.replace(f"\\{name}{{}}", value)
        text = re.sub(rf"\\{re.escape(name)}(?![A-Za-z])", value, text)
    return text


def repository_metadata(
    project_root: Path,
    *,
    final_processed_ready: bool,
) -> dict[str, object]:
    main = project_root / "manuscript/main.tex"
    main_text = main.read_text(encoding="utf-8")
    title = extract_title(main_text).replace("--", "–")
    abstract_latex, abstract_source = final_abstract(project_root, main_text)
    description = strip_latex(
        expand_generated_manuscript_values(project_root, abstract_latex)
    )
    keyword_text = extract_environment(main_text, "keyword")
    keywords = [
        strip_latex(item)
        for item in re.split(r"\\sep|;", keyword_text)
        if strip_latex(item)
    ]
    licenses = license_choice(project_root)
    repository = repository_record(project_root)
    final_abstract_ready = abstract_source == "generated_final_abstract.tex"
    pending = []
    if not final_processed_ready:
        pending.append("final_processed_files")
    if not final_abstract_ready:
        pending.append("final_abstract")
    if licenses is None:
        pending.append("author_selected_software_and_data_licenses")
    metadata: dict[str, object] = {
        "title": f"Data and code for: {title}",
        "upload_type": "dataset",
        "description": description,
        "creators": list(CREATORS),
        "keywords": keywords,
        "access_right": "open",
    }
    if licenses is not None:
        metadata["license"] = licenses["data_license"]
    if repository["repository_url"]:
        metadata["related_identifiers"] = [
            {
                "identifier": repository["repository_url"],
                "relation": "isSupplementTo",
                "scheme": "url",
            }
        ]
    return {
        "status": (
            "p418_repository_metadata_ready"
            if not pending
            else "p418_repository_metadata_draft"
        ),
        "metadata": metadata,
        "description_source": abstract_source,
        "license_choice": licenses,
        "pending_fields": pending,
        "ready_for_deposition": not pending,
        "new_physical_parameters": [],
    }


def build(project_root: Path, output_dir: Path) -> dict[str, object]:
    compact = [describe(project_root, path) for path in COMPACT_FILES]
    final_processed = [describe(project_root, path) for path in FINAL_PROCESSED_FILES]
    compact_ready = all(row["present"] for row in compact)
    final_ready = all(row["present"] for row in final_processed)
    metadata = repository_metadata(
        project_root,
        final_processed_ready=final_ready,
    )
    licenses = license_choice(project_root)
    repository = repository_record(project_root)
    payload: dict[str, object] = {
        "status": (
            "p418_public_data_release_ready"
            if compact_ready and final_ready
            else "p418_public_data_release_preflight"
        ),
        "repository_doi": repository["repository_doi"] or "pending_assignment",
        "repository_url": repository["repository_url"],
        "software_and_data_license": licenses or "pending_author_choice",
        "repository_metadata_ready": metadata["ready_for_deposition"],
        "compact_plot_data_ready": compact_ready,
        "final_processed_archive_ready": final_ready,
        "compact_files": compact,
        "final_processed_files": final_processed,
        "release_layers": [
            {
                "name": "small_source_archive",
                "content": "code, inputs, tests and compact plot-ready tables",
                "location": "results/hccb_p418_reproducibility_manifest/p418_reproduction_source.tar.gz",
            },
            {
                "name": "citable_processed_data_archive",
                "content": "geometry, selected prediction arrays, model metrics and figure records",
                "location": "to be deposited after the final model comparison",
            },
            {
                "name": "large_openfoam_fields",
                "content": "decomposed and reconstructed three-dimensional OpenFOAM fields",
                "location": "institutional/NAS archive; available on reasonable request",
            },
        ],
        "raw_openfoam_fields_in_public_small_archive": False,
        "old_tritium_release_doi_reused": False,
        "private_machine_paths_in_manifest": False,
        "new_physical_parameters": [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = output_dir / "summary.json"
    summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata_path = output_dir / "zenodo_metadata_draft.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    readme = output_dir / "README.md"
    ready_count = sum(bool(row["present"]) for row in final_processed)
    license_line = (
        "- Licences: MIT for original software; CC BY 4.0 for processed data\n"
        if licenses is not None
        else "- Licences: pending author choice\n"
    )
    readme.write_text(
        "# P418 public data release preflight\n\n"
        "This release accompanies the study of pore-resolved conjugate heat transfer "
        "and reduced-order prediction in an internally heated ceramic pebble bed. "
        "It is a preflight record: completed files are listed with size and SHA-256 "
        "in `summary.json`, whereas missing final model files remain explicitly absent.\n\n"
        "## Release layers\n\n"
        "1. **Compact source archive.** Code, registered inputs, tests, plot-ready "
        "tables and the three-mesh engineering/GCI results.\n"
        "2. **Citable processed-data archive.** Regional geometry, selected model "
        "predictions, transient comparison tables and final figure records. This "
        "layer is not declared complete before the formal model comparison finishes.\n"
        "3. **Large OpenFOAM archive.** Reconstructed and decomposed three-dimensional "
        "fields retained in the institutional archive; these are not duplicated in "
        "the compact public package.\n\n"
        "## Reproduce the compact quantitative figures\n\n"
        "From the project root, follow "
        "`results/hccb_p418_public_figure_data/README.md`. It provides the exact "
        "commands for the 60-condition response map, the seed101--seed202 integral "
        "comparison and the steady-model comparison. The three-mesh engineering "
        "observables and GCI values are supplied as CSV and JSON files rather than "
        "reconstructed from a PDF.\n\n"
        "## Training and data splits\n\n"
        "The public training manifest is provided as "
        "`formal_training_manifest_public.json`. It preserves the 75-job model, "
        "random-seed, data-split and dependency plan while replacing workstation "
        "paths with `${PROJECT_ROOT}`. It therefore documents the complete comparison "
        "plan without exposing machine-specific directories. Training, validation "
        "and independent-test trajectory identifiers are retained.\n\n"
        "## Scientific scope\n\n"
        "The successful transient database advances the energy equations on frozen, "
        "target-condition hydrodynamic fields. It is therefore a fixed-hydrodynamics "
        "thermal-step database, not evidence of a successful fully coupled flow-startup "
        "calculation. The independent seed202 results use a second intact spherical-"
        "pebble arrangement. Failed full-domain and fully coupled screening runs are "
        "reported only as applicability limits and are not included as successful "
        "training samples. The path-free direct-transport scope record is supplied "
        "with the compact files.\n\n"
        "## Current release state\n\n"
        f"- Compact files ready: {sum(bool(row['present']) for row in compact)}/"
        f"{len(compact)}\n"
        f"- Final processed files currently ready: {ready_count}/"
        f"{len(final_processed)}\n"
        "- DOI: pending assignment for this P418 study\n"
        f"{license_line}"
        "- Zenodo metadata draft: `zenodo_metadata_draft.json`\n"
        "- Raw decomposed OpenFOAM fields: retained in the large institutional/NAS "
        "archive and not duplicated in the small source package\n\n"
        "Use the SHA-256 values in `summary.json` to verify every listed file after "
        "download or transfer. No machine-local absolute path is required to reproduce "
        "the compact figures.\n",
        encoding="utf-8",
    )
    for path in (summary, readme, metadata_path):
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in PRIVATE_TEXT):
            raise ValueError(f"private machine path leaked into {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.project_root.resolve(), args.output_dir.resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
