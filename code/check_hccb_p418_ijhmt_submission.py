#!/usr/bin/env python3
"""Check the final P418 paper against the declared IJHMT submission limits."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import unicodedata
from pathlib import Path

from check_hccb_p418_final_figure_outputs import parse_pdffonts_output

REQUIRED_SECTIONS = (
    "Data and code availability",
    "CRediT authorship contribution statement",
    "Declaration of competing interest",
    "Acknowledgements",
    "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process",
)
REQUIRED_AUTHORS = (
    "Jian Wang",
    "Wei Wen",
    "Gang Shen",
    "Mingzhun Lei",
    "Yuntao Song",
)
EXPECTED_AUTHOR_LINES = (
    r"\\author\[a,b\]\{Jian Wang\\corref\{cor1\}\}",
    r"\\author\[a,b\]\{Wei Wen\}",
    r"\\author\[a\]\{Gang Shen\}",
    r"\\author\[b\]\{Mingzhun Lei\}",
    r"\\author\[b\]\{Yuntao Song\}",
)
EXPECTED_AFFILIATIONS = (
    "Anhui University of Science and Technology, Huainan 232001, China",
    "Institute of Plasma Physics, Hefei Institutes of Physical Science, Chinese Academy of Sciences, Hefei 230031, China",
)
FORBIDDEN_PDF_TEXT = (
    "[Draft:",
    "Formal accuracy, cost and cross-packing results will be inserted",
    "The final conclusions will be written after",
    "Insert the completed",
    "Insert the response-surface",
    "Insert the frozen fixed-flow",
    "The final model ranking will be generated",
)
FORBIDDEN_SOURCE_TEXT = (
    "\\draftnote",
    "\\PendingFormalResult",
    "pending formal calculation",
)
FORBIDDEN_PREVIOUS_REPOSITORY_IDENTIFIERS = (
    "pinn-tritium-release-observability",
    "tritium-release-observability",
    "10.5281/zenodo.21207574",
)
MAIN_FIGURE_SOURCE_FILES = (
    "main.tex",
    "methods_condensed.tex",
    "results_condensed.tex",
)
INTERNAL_MAIN_PAGE_TARGET = 25
INTERNAL_MAIN_WORD_TARGET = 7500
INTERNAL_MAIN_FIGURE_TARGET = 7
INTERNAL_INTRODUCTION_WORD_TARGET = 800
IJHMT_REFERENCE_LIMIT = 50
IJHMT_GUIDE_URL = (
    "https://www.sciencedirect.com/journal/"
    "international-journal-of-heat-and-mass-transfer/publish/guide-for-authors"
)
EDITABLE_MANUSCRIPT_SOURCES = (
    "manuscript/main.tex",
    "manuscript/methods_condensed.tex",
    "manuscript/results_condensed.tex",
    "manuscript/references.bib",
    "manuscript/elsarticle.cls",
    "manuscript/elsarticle-num.bst",
)
FORMAL_FIGURE_EVIDENCE = {
    "hccb_p418_steady_model_comparison.pdf": {
        "paths": ("manuscript/generated_steady_model_comparison_validated.tex",),
        "status_path": None,
        "status": None,
    },
    "hccb_p418_transient_model_comparison.pdf": {
        "paths": (
            "manuscript/generated_transient_model_comparison_validated.tex",
            "figures/hccb_p418_transient_model_comparison.json",
        ),
        "status_path": "figures/hccb_p418_transient_model_comparison.json",
        "status": "complete_formal_p418_transient_model_comparison_figure",
    },
    "hccb_p418_openfoam_model_field_comparison.pdf": {
        "paths": (
            "manuscript/generated_openfoam_model_field_comparison_validated.tex",
            "figures/hccb_p418_openfoam_model_field_comparison.json",
        ),
        "status_path": "figures/hccb_p418_openfoam_model_field_comparison.json",
        "status": "complete_same_scale_openfoam_model_field_comparison",
    },
}


def strip_latex(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", " ", text)
    text = text.replace(r"\%", "%")
    text = re.sub(r"\\(?:cite|ref|label|url|href)\{[^{}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_environment(text: str, name: str) -> str:
    match = re.search(
        rf"\\begin\{{{re.escape(name)}\}}(.*?)\\end\{{{re.escape(name)}\}}",
        text,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError(f"Missing LaTeX environment: {name}")
    return match.group(1)


def extract_title(text: str) -> str:
    match = re.search(r"\\title\{([^{}]+)\}", text)
    if match is None:
        raise ValueError("Missing LaTeX title")
    return re.sub(r"\s+", " ", match.group(1)).strip()


def extract_unnumbered_section(text: str, title: str, next_title: str) -> str:
    match = re.search(
        rf"\\section\*\{{{re.escape(title)}\}}(.*?)"
        rf"\\section\*\{{{re.escape(next_title)}\}}",
        text,
        flags=re.DOTALL,
    )
    return match.group(1) if match else ""


def normalized_statement(text: str) -> str:
    """Normalize Markdown/LaTeX wrappers while preserving statement wording."""
    text = re.sub(r"^\s*#+\s+.*$", " ", text, flags=re.MULTILINE)
    text = text.replace("\\textbf", "")
    text = text.replace("--", "-").replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("{", "").replace("}", "").replace("*", "")
    return re.sub(r"\s+", " ", text).strip()


def final_abstract(root: Path, main_text: str) -> tuple[str, str]:
    generated = root / "manuscript/generated_final_abstract.tex"
    if generated.is_file() and generated.stat().st_size:
        return generated.read_text(encoding="utf-8"), "generated_final_abstract.tex"
    environment = extract_environment(main_text, "abstract")
    fallback_match = re.fullmatch(
        r"\s*\\IfFileExists\{[^}]+\}"
        r"\{\\input\{[^}]+\}\}\{%\s*(.*?)\s*\}\s*",
        environment,
        flags=re.DOTALL,
    )
    return (fallback_match.group(1) if fallback_match else environment), "main.tex fallback"


def pdf_text_and_pages(
    path: Path, *, prefer_pypdf: bool = True
) -> tuple[str, int]:
    if prefer_pypdf:
        try:
            from pypdf import PdfReader
        except ModuleNotFoundError:
            pass
        else:
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text, len(reader.pages)

    text_process = subprocess.run(
        ["pdftotext", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    info_process = subprocess.run(
        ["pdfinfo", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    page_match = re.search(r"^Pages:\s+(\d+)\s*$", info_process.stdout, re.MULTILINE)
    if page_match is None:
        raise ValueError(f"Cannot read page count from pdfinfo: {path}")
    return text_process.stdout, int(page_match.group(1))


def pdf_font_checks(path: Path) -> dict[str, object]:
    completed = subprocess.run(
        ["pdffonts", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_pdffonts_output(completed.stdout, path)


def pdf_first_page_size_points(path: Path) -> tuple[float, float]:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError:
        completed = subprocess.run(
            ["pdfinfo", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        match = re.search(
            r"^Page size:\s+([0-9.]+)\s+x\s+([0-9.]+)\s+pts",
            completed.stdout,
            re.MULTILINE,
        )
        if match is None:
            raise ValueError(f"Cannot read page size from pdfinfo: {path}")
        return float(match.group(1)), float(match.group(2))
    page = PdfReader(path).pages[0]
    return float(page.mediabox.width), float(page.mediabox.height)


def formal_figure_evidence(
    root: Path, figure: Path
) -> tuple[bool, list[str]]:
    requirement = FORMAL_FIGURE_EVIDENCE.get(figure.name)
    if requirement is None:
        return True, []
    evidence_paths = [root / str(item) for item in requirement["paths"]]
    resolved_paths = [str(path) for path in evidence_paths]
    if any(not path.is_file() or path.stat().st_size == 0 for path in evidence_paths):
        return False, resolved_paths
    expected_status = requirement["status"]
    if expected_status is None:
        return True, resolved_paths
    evidence = root / str(requirement["status_path"])
    try:
        payload = json.loads(evidence.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False, resolved_paths
    return payload.get("status") == expected_status, resolved_paths


def planned_main_figures(root: Path, manuscript: Path) -> list[dict[str, object]]:
    pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
    figures: dict[str, dict[str, object]] = {}
    for source_name in MAIN_FIGURE_SOURCE_FILES:
        source = manuscript / source_name
        if not source.is_file():
            continue
        text = source.read_text(encoding="utf-8")
        for raw_path in pattern.findall(text):
            resolved = (source.parent / raw_path).resolve()
            file_present = resolved.is_file() and resolved.stat().st_size > 0
            canvas_size_points = (
                list(pdf_first_page_size_points(resolved)) if file_present else None
            )
            evidence_present, evidence_paths = formal_figure_evidence(root, resolved)
            key = str(resolved)
            figures.setdefault(
                key,
                {
                    "source": source_name,
                    "path": raw_path,
                    "resolved_path": key,
                    "file_present": file_present,
                    "canvas_size_points": canvas_size_points,
                    "formal_evidence_path": (
                        evidence_paths[0] if evidence_paths else None
                    ),
                    "formal_evidence_paths": evidence_paths,
                    "formal_evidence_present": evidence_present,
                    "present": file_present and evidence_present,
                },
            )
    return sorted(figures.values(), key=lambda item: str(item["path"]))


def citable_data_record(root: Path) -> dict[str, object]:
    """Read the formal data-release record without accepting placeholder IDs."""
    record = root / "results/hccb_p418_public_data_release_preflight/summary.json"
    if not record.is_file():
        return {
            "ready": False,
            "path": str(record.resolve()),
            "repository_doi": None,
            "repository_url": None,
        }
    try:
        payload = json.loads(record.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        payload = {}
    doi = str(payload.get("repository_doi") or "").strip()
    url = str(payload.get("repository_url") or "").strip()
    doi_ready = bool(doi) and "pending" not in doi.lower()
    url_ready = bool(url) and "pending" not in url.lower()
    return {
        "ready": doi_ready and url_ready,
        "path": str(record.resolve()),
        "repository_doi": doi or None,
        "repository_url": url or None,
    }


def reference_metadata_record(root: Path) -> dict[str, object]:
    candidates = sorted(
        (root / "results").glob("hccb_p418_reference_metadata_check_*/summary.json")
    )
    if not candidates:
        return {
            "ready": False,
            "path": None,
            "entry_count": 0,
            "doi_entry_count": 0,
            "unresolved_review_count": 0,
            "fetch_failure_count": 0,
        }
    path = candidates[-1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "ready": (
            payload.get("status") == "p418_reference_metadata_check_complete"
            and int(payload.get("entry_count", 0)) > 0
            and int(payload.get("doi_unresolved_review_count", -1)) == 0
            and int(payload.get("doi_fetch_failure_count", -1)) == 0
        ),
        "path": str(path.resolve()),
        "entry_count": int(payload.get("entry_count", 0)),
        "doi_entry_count": int(payload.get("doi_entry_count", 0)),
        "unresolved_review_count": int(
            payload.get("doi_unresolved_review_count", 0)
        ),
        "fetch_failure_count": int(payload.get("doi_fetch_failure_count", 0)),
    }


def repository_doi_text_checks(
    data_record: dict[str, object],
    manuscript_text: str,
    cover_letter_text: str,
) -> dict[str, bool]:
    """Switch from pending wording to the assigned DOI in the final files."""
    manuscript_lower = manuscript_text.lower()
    cover_lower = cover_letter_text.lower()
    doi = str(data_record.get("repository_doi") or "").strip().lower()
    if bool(data_record.get("ready")):
        manuscript_ok = bool(doi) and doi in manuscript_lower
        cover_ok = bool(doi) and doi in cover_lower
        no_stale_future_wording = all(
            phrase not in manuscript_lower and phrase not in cover_lower
            for phrase in (
                "will be added",
                "will be included before submission",
            )
        )
    else:
        manuscript_ok = all(
            phrase in manuscript_lower
            for phrase in ("versioned doi", "before submission")
        )
        cover_ok = all(
            phrase in cover_lower
            for phrase in ("versioned zenodo doi", "before submission")
        )
        no_stale_future_wording = True
    return {
        "manuscript": manuscript_ok,
        "cover_letter": cover_ok,
        "no_stale_future_wording": no_stale_future_wording,
    }


def cover_letter_scientific_claims(
    root: Path, cover_letter_text: str
) -> dict[str, object]:
    """Recompute the quantitative cover-letter claims from formal result files."""
    steady = json.loads(
        (root / "results/hccb_p418_sourceflow_complete_physics_60/summary.json")
        .read_text(encoding="utf-8")
    )
    transient = json.loads(
        (root / "results/hccb_p418_physical_steps_12/regional_sequences/dataset_index.json")
        .read_text(encoding="utf-8")
    )
    packing = json.loads(
        (root / "results/hccb_p418_cross_packing_seed202_integral_9/summary.json")
        .read_text(encoding="utf-8")
    )
    external = json.loads(
        (root / "results/hccb_heat_ai_external_evidence/summary.json")
        .read_text(encoding="utf-8")
    )
    metrics = packing["metric_summary"]
    values = {
        "steady_case_count": int(steady["completed_case_count"]),
        "transient_sequence_count": int(transient["sequence_count"]),
        "packing_case_count": int(packing["accepted_common_case_count"]),
        "maximum_integral_temperature_change_percent": max(
            float(metrics["outlet_temperature_K"]["maximum_absolute_relative_change_percent"]),
            float(metrics["maximum_solid_temperature_K"]["maximum_absolute_relative_change_percent"]),
        ),
        "pressure_drop_change_min_percent": float(
            metrics["pressure_drop_Pa"]["relative_change_percent_range"][0]
        ),
        "pressure_drop_change_max_percent": float(
            metrics["pressure_drop_Pa"]["relative_change_percent_range"][1]
        ),
        "nusselt_mean_absolute_relative_error_percent": float(
            external["hcpb_annulus"]["mean_absolute_relative_error_percent"]
        ),
        "pressure_gradient_median_absolute_relative_error_percent": float(
            external["fixed_bed_pressure"]["median_absolute_relative_error_percent"]
        ),
    }
    required_phrases = (
        f"{values['steady_case_count']} three-dimensional steady OpenFOAM states",
        f"{values['transient_sequence_count']} fixed-hydrodynamic thermal-step trajectories",
        "at nine matched conditions",
        "change by less than 0.67%",
        (
            "pressure drop changes by "
            f"{values['pressure_drop_change_min_percent']:.1f}–"
            f"{values['pressure_drop_change_max_percent']:.1f}%"
        ),
        (
            "mean and median absolute relative errors of "
            f"{values['nusselt_mean_absolute_relative_error_percent']:.2f}% and "
            f"{values['pressure_gradient_median_absolute_relative_error_percent']:.2f}%"
        ),
        "thermal evolution with a prescribed hydrodynamic field",
        "no full-domain or fully coupled startup accuracy is claimed",
    )
    checks = {
        phrase: phrase in cover_letter_text for phrase in required_phrases
    }
    checks["temperature_bound_supported_by_results"] = (
        values["maximum_integral_temperature_change_percent"] < 0.67
    )
    return {
        "matches": all(checks.values()),
        "values": values,
        "required_phrases": list(required_phrases),
        "checks": checks,
        "source_files": [
            "results/hccb_p418_sourceflow_complete_physics_60/summary.json",
            "results/hccb_p418_physical_steps_12/regional_sequences/dataset_index.json",
            "results/hccb_p418_cross_packing_seed202_integral_9/summary.json",
            "results/hccb_heat_ai_external_evidence/summary.json",
        ],
    }


def build(
    root: Path, *, require_supplement: bool = False
) -> dict[str, object]:
    manuscript = root / "manuscript"
    main_tex = manuscript / "main.tex"
    main_pdf = manuscript / "main.pdf"
    supplement_tex = manuscript / "supplement.tex"
    supplement_pdf = manuscript / "supplement.pdf"
    submission = root / "submission"
    title_page_file = submission / "title_page.txt"
    cover_letter = submission / "cover_letter_IJHMT.md"
    cover_letter_pdf = submission / "cover_letter_IJHMT.pdf"
    highlights_file = submission / "highlights.txt"
    credit_file = submission / "CRediT_author_statement.md"
    competing_interest_file = submission / "declaration_of_competing_interest.md"
    acknowledgements_file = submission / "acknowledgements.md"
    ai_declaration_file = submission / "declaration_of_generative_ai_use.md"
    if not main_tex.is_file() or not main_pdf.is_file():
        raise FileNotFoundError("main.tex or main.pdf is missing")
    main_text = main_tex.read_text(encoding="utf-8")
    manuscript_title = extract_title(main_text)
    abstract_latex, abstract_source = final_abstract(root, main_text)
    abstract_words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", strip_latex(abstract_latex))
    introduction_match = re.search(
        r"\\section\{Introduction\}(.*?)\\section\{Methods\}",
        main_text,
        flags=re.DOTALL,
    )
    introduction_latex = introduction_match.group(1) if introduction_match else ""
    introduction_latex = re.sub(
        r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}",
        " ",
        introduction_latex,
        flags=re.DOTALL,
    )
    introduction_words = re.findall(
        r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*",
        strip_latex(introduction_latex),
    )
    keyword_environment = extract_environment(main_text, "keyword")
    keywords = [
        strip_latex(item)
        for item in re.split(r"\\sep|;", keyword_environment)
        if strip_latex(item)
    ]
    pdf_text, page_count = pdf_text_and_pages(main_pdf)
    pdf_words = re.findall(
        r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", pdf_text
    )
    main_pdf_fonts = pdf_font_checks(main_pdf)
    main_figures = planned_main_figures(root, manuscript)
    missing_main_figures = [
        str(item["path"]) for item in main_figures if not bool(item["present"])
    ]
    present_main_figure_widths_points = [
        float(item["canvas_size_points"][0])
        for item in main_figures
        if item["file_present"] and item["canvas_size_points"] is not None
    ]
    forbidden_found = [item for item in FORBIDDEN_PDF_TEXT if item in pdf_text]
    source_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            main_tex,
            manuscript / "methods_condensed.tex",
            manuscript / "results_condensed.tex",
            manuscript / "generated_results.tex",
        )
        if path.is_file()
    )
    citation_keys = sorted(
        {
            key.strip()
            for group in re.findall(r"\\cite\w*\{([^}]*)\}", source_text)
            for key in group.split(",")
            if key.strip()
        }
    )
    bibliography_text = (manuscript / "references.bib").read_text(
        encoding="utf-8"
    )
    bibliography_entry_matches = list(
        re.finditer(r"^@\w+\{([^,]+),", bibliography_text, flags=re.MULTILINE)
    )
    bibliography_keys = [match.group(1) for match in bibliography_entry_matches]
    bibliography_entries_missing_persistent_identifier = []
    for index, match in enumerate(bibliography_entry_matches):
        end = (
            bibliography_entry_matches[index + 1].start()
            if index + 1 < len(bibliography_entry_matches)
            else len(bibliography_text)
        )
        entry_text = bibliography_text[match.start() : end]
        if not re.search(
            r"^\s*(?:doi|url)\s*=", entry_text, flags=re.MULTILINE | re.IGNORECASE
        ):
            bibliography_entries_missing_persistent_identifier.append(
                match.group(1)
            )
    editable_source_paths = [root / item for item in EDITABLE_MANUSCRIPT_SOURCES]
    missing_editable_sources = [
        str(path.resolve())
        for path in editable_source_paths
        if not path.is_file() or path.stat().st_size == 0
    ]
    data_record = citable_data_record(root)
    reference_record = reference_metadata_record(root)
    missing_bibliography_entries = sorted(
        set(citation_keys) - set(bibliography_keys)
    )
    uncited_bibliography_entries = sorted(
        set(bibliography_keys) - set(citation_keys)
    )
    forbidden_source_found = [
        item for item in FORBIDDEN_SOURCE_TEXT if item in source_text
    ]
    previous_repository_identifiers_found = [
        item
        for item in FORBIDDEN_PREVIOUS_REPOSITORY_IDENTIFIERS
        if item.lower() in main_text.lower()
    ]
    availability_text = ""
    availability_match = re.search(
        r"\\section\*\{Data and code availability\}(.*?)"
        r"\\section\*\{CRediT authorship contribution statement\}",
        main_text,
        flags=re.DOTALL,
    )
    if availability_match:
        availability_text = strip_latex(availability_match.group(1)).lower()
    missing_sections = [item for item in REQUIRED_SECTIONS if item not in main_text]
    credit_text = extract_unnumbered_section(
        main_text,
        "CRediT authorship contribution statement",
        "Declaration of competing interest",
    )
    competing_interest_main = extract_unnumbered_section(
        main_text,
        "Declaration of competing interest",
        "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process",
    )
    acknowledgements_match = re.search(
        r"\\section\*\{Acknowledgements\}(.*?)\\begingroup",
        main_text,
        flags=re.DOTALL,
    )
    acknowledgements_main = (
        acknowledgements_match.group(1) if acknowledgements_match else ""
    )
    ai_declaration_match = re.search(
        r"\\section\*\{Declaration of generative AI and AI-assisted "
        r"technologies in the manuscript preparation process\}(.*?)"
        r"\\section\*\{Acknowledgements\}",
        main_text,
        flags=re.DOTALL,
    )
    ai_declaration_main = (
        ai_declaration_match.group(1) if ai_declaration_match else ""
    )
    missing_credit_authors = [item for item in REQUIRED_AUTHORS if item not in credit_text]

    supplement_exists = (
        supplement_pdf.is_file() and supplement_pdf.stat().st_size > 0
    )
    supplement_title = (
        extract_title(supplement_tex.read_text(encoding="utf-8"))
        if supplement_tex.is_file()
        else ""
    )
    supplement_title_matches = (
        not supplement_exists
        or (
            bool(supplement_title)
            and manuscript_title in supplement_title
        )
    )
    highlights = (
        [
            line.strip()
            for line in highlights_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if highlights_file.is_file()
        else []
    )
    highlight_lengths = [len(item) for item in highlights]
    cover_letter_text = (
        cover_letter.read_text(encoding="utf-8")
        if cover_letter.is_file()
        else ""
    )
    cover_letter_pdf_text, cover_letter_pdf_pages = (
        pdf_text_and_pages(cover_letter_pdf)
        if cover_letter_pdf.is_file() and cover_letter_pdf.stat().st_size > 0
        else ("", 0)
    )
    normalized_cover_pdf = " ".join(
        unicodedata.normalize("NFKC", cover_letter_pdf_text).split()
    )
    normalized_manuscript_title = " ".join(
        unicodedata.normalize("NFKC", manuscript_title.replace("--", "–")).split()
    )
    canonical_cover_pdf = re.sub(r"[^a-z0-9]", "", normalized_cover_pdf.lower())
    canonical_manuscript_title = re.sub(
        r"[^a-z0-9]", "", normalized_manuscript_title.lower()
    )
    doi_text_checks = repository_doi_text_checks(
        data_record, main_text, cover_letter_text
    )
    cover_pdf_doi_checks = repository_doi_text_checks(
        data_record, main_text, normalized_cover_pdf
    )
    cover_letter_claims = cover_letter_scientific_claims(root, cover_letter_text)
    expected_external_highlight = (
        "External Nusselt and pressure-gradient errors are "
        f"{cover_letter_claims['values']['nusselt_mean_absolute_relative_error_percent']:.2f}% and "
        f"{cover_letter_claims['values']['pressure_gradient_median_absolute_relative_error_percent']:.2f}%."
    )
    title_page_text = (
        title_page_file.read_text(encoding="utf-8")
        if title_page_file.is_file()
        else ""
    )
    title_page_complete = (
        (
            manuscript_title in title_page_text
            or manuscript_title.replace("--", "–") in title_page_text
        )
        and all(author in title_page_text for author in REQUIRED_AUTHORS)
        and all(item in title_page_text for item in EXPECTED_AFFILIATIONS)
        and "wjfttt@mail.ustc.edu.cn" in title_page_text
    )
    corresponding_author_phone_complete = bool(
        re.search(
            r"(?im)^Phone:\s*\+\d[\d\s()-]{6,}$",
            title_page_text,
        )
    )
    cover_letter_title_present = (
        manuscript_title in cover_letter_text
        or manuscript_title.replace("--", "–") in cover_letter_text
    )
    cover_letter_complete = cover_letter_title_present and all(
        item in cover_letter_text
        for item in (
            "International Journal of Heat and Mass Transfer",
            "not under consideration by another journal",
            "no competing interests",
            "wjfttt@mail.ustc.edu.cn",
        )
    )
    cover_letter_pdf_complete = (
        cover_letter_pdf_pages == 1
        and canonical_manuscript_title in canonical_cover_pdf
        and "International Journal of Heat and Mass Transfer" in normalized_cover_pdf
        and "wjfttt@mail.ustc.edu.cn" in normalized_cover_pdf
    )
    cover_letter_availability_complete = all(
        phrase in cover_letter_text.lower()
        for phrase in (
            "https://github.com/wangjianfttt/fusion-pebble-bed-heat-transfer-ai",
            "software and data licences",
            "institutional archive",
            "reasonable request",
        )
    ) and doi_text_checks["cover_letter"] and doi_text_checks[
        "no_stale_future_wording"
    ]
    credit_file_text = (
        credit_file.read_text(encoding="utf-8") if credit_file.is_file() else ""
    )
    competing_interest_text = (
        competing_interest_file.read_text(encoding="utf-8")
        if competing_interest_file.is_file()
        else ""
    )
    acknowledgements_text = (
        acknowledgements_file.read_text(encoding="utf-8")
        if acknowledgements_file.is_file()
        else ""
    )
    ai_declaration_text = (
        ai_declaration_file.read_text(encoding="utf-8")
        if ai_declaration_file.is_file()
        else ""
    )
    checks = {
        "abstract_at_most_250_words": len(abstract_words) <= 250,
        "introduction_at_most_800_words": len(introduction_words)
        <= INTERNAL_INTRODUCTION_WORD_TARGET,
        "keyword_count_1_to_7": 1 <= len(keywords) <= 7,
        "main_pdf_at_most_40_pages": page_count <= 40,
        "main_pdf_fonts_embedded_without_type3": (
            bool(main_pdf_fonts["all_fonts_embedded"])
            and int(main_pdf_fonts["type3_font_count"]) == 0
        ),
        "compact_main_pdf_at_most_25_pages": page_count
        <= INTERNAL_MAIN_PAGE_TARGET,
        "compact_main_pdf_at_most_7500_words": len(pdf_words)
        <= INTERNAL_MAIN_WORD_TARGET,
        "planned_main_figure_count_at_most_7": len(main_figures)
        <= INTERNAL_MAIN_FIGURE_TARGET,
        "present_main_figure_canvas_widths_consistent": (
            not present_main_figure_widths_points
            or max(present_main_figure_widths_points)
            - min(present_main_figure_widths_points)
            <= 0.1
        ),
        "main_manuscript_has_no_appendix": (
            "\\appendix" not in source_text
            and not re.search(r"\\section\*?\{Appendix", source_text)
        ),
        "all_planned_main_figures_present": not missing_main_figures,
        "supplement_requirement_satisfied": (
            supplement_exists or not require_supplement
        ),
        "supplement_title_matches_manuscript": supplement_title_matches,
        "required_sections_present": not missing_sections,
        "all_citations_have_bibtex_entries": not missing_bibliography_entries,
        "cited_reference_count_at_most_50": len(citation_keys)
        <= IJHMT_REFERENCE_LIMIT,
        "bibtex_keys_are_unique": len(bibliography_keys)
        == len(set(bibliography_keys)),
        "all_bibliography_entries_have_doi_or_source_url": (
            not bibliography_entries_missing_persistent_identifier
        ),
        "reference_metadata_checked_without_unresolved_differences": bool(
            reference_record["ready"]
        )
        and int(reference_record["entry_count"]) == len(bibliography_keys),
        "editable_manuscript_sources_present": not missing_editable_sources,
        "data_and_code_availability_is_truthful": (
            "https://github.com/wangjianfttt/"
            "fusion-pebble-bed-heat-transfer-ai" in main_text
            and all(
                phrase in availability_text
                for phrase in (
                    "mit license",
                    "cc by 4.0",
                    "institutional archive",
                    "reasonable request",
                )
            )
            and doi_text_checks["manuscript"]
            and doi_text_checks["no_stale_future_wording"]
        ),
        "citable_data_repository_record_ready": bool(data_record["ready"]),
        "no_previous_project_repository_identifier": (
            not previous_repository_identifiers_found
        ),
        "all_authors_in_credit_statement": not missing_credit_authors,
        "author_names_affiliations_and_email_complete": (
            all(re.search(pattern, main_text) for pattern in EXPECTED_AUTHOR_LINES)
            and all(item in main_text for item in EXPECTED_AFFILIATIONS)
            and "\\ead{wjfttt@mail.ustc.edu.cn}" in main_text
        ),
        "no_draft_or_future_result_text_in_pdf": not forbidden_found,
        "no_draft_markers_in_submission_sources": not forbidden_source_found,
        "highlights_3_to_5": 3 <= len(highlights) <= 5,
        "highlights_each_at_most_85_characters": bool(highlights)
        and max(highlight_lengths) <= 85,
        "highlights_external_errors_match_results": (
            expected_external_highlight in highlights
        ),
        "cover_letter_present": cover_letter_complete,
        "cover_letter_pdf_ready": cover_letter_pdf_complete,
        "cover_letter_pdf_repository_record_matches": (
            cover_pdf_doi_checks["cover_letter"]
            and cover_pdf_doi_checks["no_stale_future_wording"]
        ),
        "cover_letter_data_availability_matches_manuscript": (
            cover_letter_availability_complete
        ),
        "cover_letter_title_matches_manuscript": cover_letter_title_present,
        "cover_letter_scientific_claims_match_results": bool(
            cover_letter_claims["matches"]
        ),
        "separate_title_page_matches_manuscript": title_page_complete,
        "corresponding_author_phone_complete": (
            corresponding_author_phone_complete
        ),
        "separate_credit_statement_complete": bool(credit_file_text)
        and all(author in credit_file_text for author in REQUIRED_AUTHORS),
        "separate_credit_statement_matches_manuscript": (
            normalized_statement(credit_file_text)
            == normalized_statement(credit_text)
        ),
        "separate_competing_interest_statement_complete": (
            "no known competing financial interests" in competing_interest_text
        ),
        "separate_competing_interest_matches_manuscript": (
            normalized_statement(competing_interest_text)
            == normalized_statement(competing_interest_main)
        ),
        "separate_acknowledgements_complete": all(
            item in acknowledgements_text
            for item in (
                "2408085QA030",
                "2504000000-04-05-42671",
                "AIMTEERC202307",
                "DSJJ-2025-08",
                "2024M753266",
            )
        ),
        "separate_acknowledgements_matches_manuscript": (
            normalized_statement(acknowledgements_text)
            == normalized_statement(acknowledgements_main)
        ),
        "separate_ai_declaration_complete": all(
            phrase in ai_declaration_text
            for phrase in (
                "OpenAI Codex",
                "independently verified the calculations",
                "No generative AI system was used to create or alter the scientific images or numerical results",
            )
        ),
        "separate_ai_declaration_matches_manuscript": (
            normalized_statement(ai_declaration_text)
            == normalized_statement(ai_declaration_main)
        ),
        "final_generated_abstract_present": abstract_source
        == "generated_final_abstract.tex",
    }
    complete = all(checks.values())
    return {
        "status": (
            "completed_p418_ijhmt_submission_check"
            if complete
            else "p418_ijhmt_submission_check_incomplete"
        ),
        "checks": checks,
        "abstract_source": abstract_source,
        "manuscript_title": manuscript_title,
        "abstract_word_count": len(abstract_words),
        "introduction_word_count": len(introduction_words),
        "internal_introduction_word_target": INTERNAL_INTRODUCTION_WORD_TARGET,
        "keyword_count": len(keywords),
        "keywords": keywords,
        "main_pdf_page_count": page_count,
        "main_pdf_word_count": len(pdf_words),
        "main_pdf_fonts": main_pdf_fonts,
        "internal_main_page_target": INTERNAL_MAIN_PAGE_TARGET,
        "internal_main_word_target": INTERNAL_MAIN_WORD_TARGET,
        "planned_main_figure_count": len(main_figures),
        "internal_main_figure_target": INTERNAL_MAIN_FIGURE_TARGET,
        "planned_main_figures": main_figures,
        "present_main_figure_widths_points": present_main_figure_widths_points,
        "missing_main_figures": missing_main_figures,
        "supplement_required": require_supplement,
        "supplement_pdf_exists": supplement_exists,
        "supplement_title": supplement_title,
        "cover_letter_path": str(cover_letter.resolve()),
        "cover_letter_pdf_path": str(cover_letter_pdf.resolve()),
        "cover_letter_pdf_page_count": cover_letter_pdf_pages,
        "cover_letter_scientific_claims": cover_letter_claims,
        "title_page_path": str(title_page_file.resolve()),
        "highlights_path": str(highlights_file.resolve()),
        "credit_statement_path": str(credit_file.resolve()),
        "competing_interest_path": str(competing_interest_file.resolve()),
        "acknowledgements_path": str(acknowledgements_file.resolve()),
        "ai_declaration_path": str(ai_declaration_file.resolve()),
        "highlights": highlights,
        "highlight_character_counts": highlight_lengths,
        "expected_external_highlight": expected_external_highlight,
        "missing_sections": missing_sections,
        "citation_count": len(citation_keys),
        "ijhmt_reference_limit": IJHMT_REFERENCE_LIMIT,
        "ijhmt_guide_url": IJHMT_GUIDE_URL,
        "bibtex_entry_count": len(bibliography_keys),
        "missing_bibliography_entries": missing_bibliography_entries,
        "uncited_bibliography_entries": uncited_bibliography_entries,
        "bibliography_entries_missing_persistent_identifier": (
            bibliography_entries_missing_persistent_identifier
        ),
        "reference_metadata_record": reference_record,
        "editable_manuscript_sources": [
            str(path.resolve()) for path in editable_source_paths
        ],
        "missing_editable_manuscript_sources": missing_editable_sources,
        "citable_data_record": data_record,
        "repository_doi_text_checks": doi_text_checks,
        "cover_letter_pdf_repository_doi_text_checks": cover_pdf_doi_checks,
        "missing_credit_authors": missing_credit_authors,
        "forbidden_pdf_text_found": forbidden_found,
        "forbidden_source_text_found": forbidden_source_found,
        "previous_repository_identifiers_found": (
            previous_repository_identifiers_found
        ),
        "new_physical_parameters": [],
    }


def write_chinese(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# IJHMT投稿前文字和格式检查",
        "",
        f"- 当前状态：{'全部通过' if payload['status'].startswith('completed') else '尚未全部通过'}",
        f"- 摘要词数：{payload['abstract_word_count']}（上限250）",
        f"- 引言词数：{payload['introduction_word_count']}"
        f"（内部精简目标不超过{payload['internal_introduction_word_target']}）",
        f"- 关键词数量：{payload['keyword_count']}（要求1--7）",
        f"- 主文PDF页数：{payload['main_pdf_page_count']}（当前检查上限40）",
        f"- 内部精简目标：不超过{payload['internal_main_page_target']}页",
        f"- 主文PDF英文词数：{payload['main_pdf_word_count']}"
        f"（内部目标不超过{payload['internal_main_word_target']}）",
        f"- 计划主图：{payload['planned_main_figure_count']}张"
        f"（目标不超过{payload['internal_main_figure_target']}张）",
        "- 已生成主图画布宽度："
        + ", ".join(
            f"{float(value):.1f} pt"
            for value in payload["present_main_figure_widths_points"]
        ),
        f"- 正文引用：{payload['citation_count']}篇"
        f"（IJHMT上限{payload['ijhmt_reference_limit']}篇）",
        "",
        "## 逐项结果",
        "",
    ]
    labels = {
        "abstract_at_most_250_words": "摘要长度",
        "introduction_at_most_800_words": "引言精简长度",
        "keyword_count_1_to_7": "关键词数量",
        "main_pdf_at_most_40_pages": "主文页数",
        "main_pdf_fonts_embedded_without_type3": (
            "主文PDF字体全部嵌入且无Type 3字体"
        ),
        "compact_main_pdf_at_most_25_pages": "内部精简页数目标",
        "compact_main_pdf_at_most_7500_words": "内部精简词数目标",
        "planned_main_figure_count_at_most_7": "主图数量",
        "present_main_figure_canvas_widths_consistent": "主图画布宽度一致",
        "main_manuscript_has_no_appendix": "主文不设附录",
        "all_planned_main_figures_present": "全部计划主图已经生成",
        "supplement_requirement_satisfied": (
            "英文补充材料（本次投稿方案需要时）"
        ),
        "supplement_title_matches_manuscript": "补充材料题目与主文一致",
        "required_sections_present": "数据说明、作者贡献、利益冲突和基金",
        "all_citations_have_bibtex_entries": "正文引用都有BibTeX条目",
        "cited_reference_count_at_most_50": "正文引用不超过50篇",
        "bibtex_keys_are_unique": "BibTeX键没有重复",
        "all_bibliography_entries_have_doi_or_source_url": (
            "每条文献都有DOI或原始来源网址"
        ),
        "reference_metadata_checked_without_unresolved_differences": (
            "文献题录已逐条复核且无未解决差异"
        ),
        "editable_manuscript_sources_present": "LaTeX和BibTeX可编辑源文件齐全",
        "data_and_code_availability_is_truthful": "数据和代码公开说明符合当前实际状态",
        "citable_data_repository_record_ready": "本项目数据仓库DOI和链接已确定",
        "no_previous_project_repository_identifier": "没有误用上一项研究的仓库或DOI",
        "all_authors_in_credit_statement": "五位作者贡献",
        "author_names_affiliations_and_email_complete": "作者、单位和通讯邮箱",
        "separate_credit_statement_complete": "单独上传的作者贡献文件",
        "separate_credit_statement_matches_manuscript": "作者贡献附件与正文一致",
        "separate_competing_interest_statement_complete": "单独上传的利益冲突声明",
        "separate_competing_interest_matches_manuscript": "利益冲突附件与正文一致",
        "separate_acknowledgements_complete": "单独上传的基金致谢文件",
        "separate_acknowledgements_matches_manuscript": "基金致谢附件与正文一致",
        "separate_ai_declaration_complete": "单独上传的AI辅助使用声明",
        "separate_ai_declaration_matches_manuscript": "AI辅助使用声明附件与正文一致",
        "no_draft_or_future_result_text_in_pdf": "PDF中没有待填结果或草稿文字",
        "no_draft_markers_in_submission_sources": "投稿源文件中没有旧草稿标记",
        "highlights_3_to_5": "Highlights数量为3--5条",
        "highlights_each_at_most_85_characters": "每条Highlight不超过85个字符",
        "highlights_external_errors_match_results": (
            "Highlights中的Nusselt数和压力梯度误差与正式结果一致"
        ),
        "cover_letter_present": "本论文专用Cover letter",
        "cover_letter_pdf_ready": "Cover letter PDF为单页且题目、期刊和邮箱正确",
        "cover_letter_pdf_repository_record_matches": "Cover letter PDF中的数据DOI状态与正文一致",
        "cover_letter_data_availability_matches_manuscript": "投稿信的数据公开说明与正文一致",
        "cover_letter_title_matches_manuscript": "Cover letter题目与论文一致",
        "cover_letter_scientific_claims_match_results": (
            "Cover letter关键数字与正式结果一致"
        ),
        "separate_title_page_matches_manuscript": "单独Title page与正文作者信息一致",
        "corresponding_author_phone_complete": "通讯作者国际格式电话号码",
        "final_generated_abstract_present": "摘要已经由正式结果生成",
    }
    for key, value in payload["checks"].items():
        lines.append(f"- {'通过' if value else '未通过'}：{labels[key]}")
    if payload["forbidden_pdf_text_found"]:
        lines.extend(["", "当前PDF中仍出现："])
        lines.extend(f"- `{item}`" for item in payload["forbidden_pdf_text_found"])
    if payload["forbidden_source_text_found"]:
        lines.extend(["", "投稿源文件中仍出现："])
        lines.extend(
            f"- `{item}`" for item in payload["forbidden_source_text_found"]
        )
    if payload["missing_main_figures"]:
        lines.extend(["", "尚未生成的正式主图："])
        lines.extend(f"- `{item}`" for item in payload["missing_main_figures"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-supplement", action="store_true")
    args = parser.parse_args()
    payload = build(
        args.project_root.resolve(),
        require_supplement=args.require_supplement,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_chinese(args.output_dir / "IJHMT_投稿前检查_CN.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.require_complete and not payload["status"].startswith("completed"):
        raise SystemExit("IJHMT submission check is incomplete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
