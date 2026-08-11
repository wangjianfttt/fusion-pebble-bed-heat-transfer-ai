from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from check_hccb_p418_ijhmt_submission import (  # noqa: E402
    build,
    citable_data_record,
    final_abstract,
    formal_figure_evidence,
    pdf_text_and_pages,
    repository_doi_text_checks,
    strip_latex,
)


def test_working_paper_has_clean_pdf_but_waits_for_final_transient_result() -> None:
    if not (ROOT / "manuscript/main.pdf").is_file():
        pytest.skip(
            "the public source repository intentionally excludes the compiled manuscript PDF"
        )
    payload = build(ROOT)
    checks = payload["checks"]
    assert checks["abstract_at_most_250_words"]
    assert checks["keyword_count_1_to_7"]
    assert checks["main_pdf_at_most_40_pages"]
    assert checks["compact_main_pdf_at_most_25_pages"]
    assert checks["compact_main_pdf_at_most_7500_words"]
    assert checks["present_main_figure_canvas_widths_consistent"]
    assert checks["main_manuscript_has_no_appendix"]
    assert checks["main_pdf_fonts_embedded_without_type3"]
    assert payload["main_pdf_fonts"]["all_fonts_embedded"]
    assert payload["main_pdf_fonts"]["type3_font_count"] == 0
    assert checks["supplement_requirement_satisfied"]
    assert checks["supplement_title_matches_manuscript"]
    assert not payload["supplement_required"]
    assert payload["manuscript_title"] in payload["supplement_title"]
    assert checks["required_sections_present"]
    assert checks["all_citations_have_bibtex_entries"]
    assert checks["cited_reference_count_at_most_50"]
    assert checks["bibtex_keys_are_unique"]
    assert checks["all_bibliography_entries_have_doi_or_source_url"]
    assert checks["editable_manuscript_sources_present"]
    assert checks["data_and_code_availability_is_truthful"]
    assert not checks["citable_data_repository_record_ready"]
    assert payload["citable_data_record"]["repository_doi"] == (
        "pending_assignment"
    )
    assert checks["no_previous_project_repository_identifier"]
    assert payload["previous_repository_identifiers_found"] == []
    assert payload["missing_bibliography_entries"] == []
    assert payload["bibliography_entries_missing_persistent_identifier"] == []
    assert checks["all_authors_in_credit_statement"]
    assert checks["author_names_affiliations_and_email_complete"]
    assert not checks["final_generated_abstract_present"]
    assert not checks["all_planned_main_figures_present"]
    assert checks["no_draft_or_future_result_text_in_pdf"]
    assert checks["no_draft_markers_in_submission_sources"]
    assert checks["highlights_3_to_5"]
    assert checks["highlights_each_at_most_85_characters"]
    assert checks["highlights_external_errors_match_results"]
    assert payload["expected_external_highlight"] == (
        "External Nusselt and pressure-gradient errors are 3.87% and 3.71%."
    )
    assert checks["cover_letter_present"]
    assert checks["cover_letter_pdf_ready"]
    assert checks["cover_letter_pdf_repository_record_matches"]
    assert payload["cover_letter_pdf_page_count"] == 1
    assert checks["cover_letter_data_availability_matches_manuscript"]
    assert checks["cover_letter_title_matches_manuscript"]
    assert checks["cover_letter_scientific_claims_match_results"]
    assert checks["separate_title_page_matches_manuscript"]
    assert not checks["corresponding_author_phone_complete"]
    assert checks["separate_credit_statement_complete"]
    assert checks["separate_credit_statement_matches_manuscript"]
    assert checks["separate_competing_interest_statement_complete"]
    assert checks["separate_competing_interest_matches_manuscript"]
    assert checks["separate_acknowledgements_complete"]
    assert checks["separate_acknowledgements_matches_manuscript"]
    assert checks["separate_ai_declaration_complete"]
    assert checks["separate_ai_declaration_matches_manuscript"]
    assert payload["forbidden_source_text_found"] == []
    assert payload["highlight_character_counts"] == [76, 74, 76, 66, 74]
    claims = payload["cover_letter_scientific_claims"]
    assert claims["matches"]
    assert claims["values"]["steady_case_count"] == 60
    assert claims["values"]["transient_sequence_count"] == 12
    assert claims["values"]["packing_case_count"] == 9
    assert claims["values"]["maximum_integral_temperature_change_percent"] < 0.67
    assert round(claims["values"]["pressure_drop_change_min_percent"], 1) == 14.7
    assert round(claims["values"]["pressure_drop_change_max_percent"], 1) == 18.0
    assert round(
        claims["values"]["nusselt_mean_absolute_relative_error_percent"], 2
    ) == 3.87
    assert round(
        claims["values"]["pressure_gradient_median_absolute_relative_error_percent"],
        2,
    ) == 3.71
    assert "thermal evolution with a prescribed hydrodynamic field" in claims[
        "required_phrases"
    ]
    assert "no full-domain or fully coupled startup accuracy is claimed" in claims[
        "required_phrases"
    ]
    cover_letter_text = (
        ROOT / "submission/cover_letter_IJHMT.md"
    ).read_text(encoding="utf-8")
    assert (
        payload["manuscript_title"] in cover_letter_text
        or payload["manuscript_title"].replace("--", "–") in cover_letter_text
    )
    assert payload["abstract_word_count"] <= 250
    assert payload["main_pdf_word_count"] <= 7500
    assert payload["keyword_count"] == 6
    assert payload["citation_count"] <= payload["ijhmt_reference_limit"] == 50
    assert payload["missing_editable_manuscript_sources"] == []
    figure_by_name = {
        Path(str(item["resolved_path"])).name: item
        for item in payload["planned_main_figures"]
    }
    assert {
        round(float(width), 1)
        for width in payload["present_main_figure_widths_points"]
    } == {388.8}
    steady = figure_by_name["hccb_p418_steady_model_comparison.pdf"]
    assert steady["file_present"]
    assert steady["formal_evidence_present"]
    assert steady["present"]
    assert (
        "../figures/hccb_p418_steady_model_comparison.pdf"
        not in payload["missing_main_figures"]
    )
    assert (
        "../figures/hccb_p418_transient_model_comparison.pdf"
        in payload["missing_main_figures"]
    )


def test_transient_figure_requires_result_json_and_validation_marker(
    tmp_path: Path,
) -> None:
    figure = tmp_path / "figures/hccb_p418_transient_model_comparison.pdf"
    figure.parent.mkdir(parents=True)
    figure.write_bytes(b"%PDF-formal")
    result = figure.with_suffix(".json")
    result.write_text(
        '{"status":"complete_formal_p418_transient_model_comparison_figure"}\n',
        encoding="utf-8",
    )
    complete, paths = formal_figure_evidence(tmp_path, figure)
    assert not complete
    assert len(paths) == 2

    marker = (
        tmp_path
        / "manuscript/generated_transient_model_comparison_validated.tex"
    )
    marker.parent.mkdir()
    marker.write_text("% validated\n", encoding="utf-8")
    complete, _ = formal_figure_evidence(tmp_path, figure)
    assert complete


def test_poppler_fallback_reads_the_current_pdf() -> None:
    if not (ROOT / "manuscript/main.pdf").is_file():
        pytest.skip(
            "the public source repository intentionally excludes the compiled manuscript PDF"
        )
    text, pages = pdf_text_and_pages(
        ROOT / "manuscript/main.pdf", prefer_pypdf=False
    )
    assert pages > 0
    assert "Transient conjugate heat transfer" in text
    assert "graph\u2013Transformer prediction" in text


def test_citable_data_record_rejects_placeholder_and_accepts_doi(
    tmp_path: Path,
) -> None:
    record = (
        tmp_path
        / "results/hccb_p418_public_data_release_preflight/summary.json"
    )
    record.parent.mkdir(parents=True)
    record.write_text(
        '{"repository_doi":"pending_assignment","repository_url":null}\n',
        encoding="utf-8",
    )
    assert not citable_data_record(tmp_path)["ready"]

    record.write_text(
        '{"repository_doi":"10.5281/zenodo.1234567",'
        '"repository_url":"https://doi.org/10.5281/zenodo.1234567"}\n',
        encoding="utf-8",
    )
    payload = citable_data_record(tmp_path)
    assert payload["ready"]
    assert payload["repository_doi"] == "10.5281/zenodo.1234567"


def test_repository_doi_text_switches_from_pending_to_assigned_record() -> None:
    pending = {"ready": False, "repository_doi": "pending_assignment"}
    pending_checks = repository_doi_text_checks(
        pending,
        "A versioned DOI will be added before submission.",
        "A versioned Zenodo DOI will be included before submission.",
    )
    assert all(pending_checks.values())

    final = {"ready": True, "repository_doi": "10.5281/zenodo.1234567"}
    final_checks = repository_doi_text_checks(
        final,
        "Data: https://doi.org/10.5281/zenodo.1234567",
        "Archived at DOI 10.5281/zenodo.1234567.",
    )
    assert all(final_checks.values())

    stale_checks = repository_doi_text_checks(
        final,
        "DOI 10.5281/zenodo.1234567 will be added before submission.",
        "DOI 10.5281/zenodo.1234567 will be included before submission.",
    )
    assert stale_checks["manuscript"]
    assert stale_checks["cover_letter"]
    assert not stale_checks["no_stale_future_wording"]


def test_fallback_abstract_is_unwrapped_and_keeps_escaped_percent() -> None:
    main_text = (ROOT / "manuscript/main.tex").read_text(encoding="utf-8")
    abstract, source = final_abstract(ROOT, main_text)
    plain = strip_latex(abstract)
    assert source == "main.tex fallback"
    assert not abstract.lstrip().startswith(r"\IfFileExists")
    assert plain.startswith("Pore-scale heat transfer")
    assert "0.67% and 0.31%" in plain
    assert plain.endswith("finite-volume energy consistency.")
