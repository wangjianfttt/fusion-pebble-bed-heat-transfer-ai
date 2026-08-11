from __future__ import annotations

import csv
import glob
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MAP = ROOT / "manuscript/result_source_map.csv"


def test_every_registered_program_exists() -> None:
    with SOURCE_MAP.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    missing = [
        program
        for row in rows
        for program in row["program"].split(";")
        if not (ROOT / program).is_file()
    ]
    assert missing == []


def test_completed_result_sources_are_local_and_present() -> None:
    with SOURCE_MAP.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    pending_tokens = ("pending", "partial", "optional")
    absolute = []
    missing = []
    for row in rows:
        paths = [path for path in row["source_data"].split(";") if path]
        absolute.extend(
            (row["result_or_section"], path)
            for path in paths
            if Path(path).is_absolute()
        )
        if any(token in row["status"] for token in pending_tokens):
            continue
        for path in paths:
            present = bool(glob.glob(str(ROOT / path))) if "*" in path else (ROOT / path).exists()
            if not present:
                missing.append((row["result_or_section"], path))

    assert absolute == []
    assert missing == []


def test_pending_results_are_not_claimed_ready() -> None:
    with SOURCE_MAP.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_section = {row["result_or_section"]: row for row in rows}
    assert by_section["60-condition physical response"]["status"] == "ready"
    assert by_section["steady common inputs"]["status"] == "ready"
    assert by_section["steady model comparison"]["status"] == "complete_5_models_5_splits"
    assert by_section["steady performance table"]["status"] == "complete_5_models_5_splits"
    assert by_section["steady result text"]["status"] == "complete_5_models_5_splits"
    assert (
        by_section["steady initialization repeat"]["status"]
        == "complete_4_models_3_seeds"
    )
    assert (
        by_section["steady training-condition learning curve"]["status"]
        == "p418_steady_learning_curve_complete"
    )
    assert by_section["regional representation fidelity"]["status"] == "complete_60_conditions"
    assert by_section["native temperature reconstruction"]["status"] == "complete_60_conditions"
    assert by_section["regional fidelity manuscript text"]["status"] == "complete_60_conditions"
    assert by_section["native-cell model prediction"]["status"] == "complete_5_models_main_split"
    assert by_section["native-cell performance table"]["status"] == "complete_5_models_main_split"
    assert by_section["graph-Transformer comparison"]["status"] == "pending_formal_models"
    assert by_section["diffusion refinement"]["status"] == "pending_diffusion_model"
    assert (
        by_section["transient trajectory-count learning curve"]["status"]
        == "pending_trajectory_learning_curve"
    )
    assert by_section["steady-transient split alignment"]["status"] == "ready"
    assert by_section["fused end-to-end prediction"]["status"] == "pending_formal_models"
    assert (
        by_section["parallel transient restart fields"]["status"]
        == "complete_fixed_flow_12_recovered"
    )
    requirements = json.loads(
        (
            ROOT
            / "results/hccb_p418_final_scientific_requirements_current/summary.json"
        ).read_text(encoding="utf-8")
    )
    expected_progress = (
        f"partial_{requirements['completed_count']}_of_"
        f"{requirements['required_count']}"
    )
    progress = by_section["current final-paper requirements"]
    assert progress["status"] == expected_progress
    assert (
        f"{requirements['completed_count']}/{requirements['required_count']}"
        in progress["plain_explanation"]
    )
    assert by_section["seed202 OpenFOAM terminal status"]["status"] == "complete_9_of_9"
    assert (
        by_section["seed101-seed202 integral response comparison"]["status"]
        == "complete_9_common_cases"
    )
    assert (
        by_section["high-Re independent fixed-flow prediction"]["status"]
        == "complete_fixed_flow_6_bounded_models_3"
    )
    assert (
        by_section["fixed-flow and fully coupled thermal-step comparison"]["status"]
        == "fixed_flow_12_complete_fully_coupled_unavailable_property_range"
    )
    figure_quality = by_section["final figure output quality"]
    assert figure_quality["status"] == "partial_5_of_7"
    assert "hccb_p418_seed202_integral_9.pdf" in figure_quality["source_data"]
    assert "hccb_p418_openfoam_model_field_comparison.pdf" in figure_quality["source_data"]
    assert "hccb_p418_cross_packing_results.pdf" not in figure_quality["source_data"]


def test_high_re_comparison_is_connected_to_manuscript_outputs() -> None:
    with SOURCE_MAP.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_section = {row["result_or_section"]: row for row in rows}
    comparison = by_section["high-Re independent fixed-flow prediction"]
    assert "generated_high_re_comparison.tex" in comparison["source_data"]
    assert "summarize_hccb_p418_high_re_model_comparison.py" in comparison["program"]
    final_sections = by_section["final abstract discussion conclusions"]
    assert "hccb_p418_high_re_three_bounded_model_evaluation/comparison/summary.json" in final_sections["source_data"]
    final_pdf = by_section["formal manuscript PDF"]
    assert "generated_high_re_comparison.tex" in final_pdf["source_data"]


def test_failed_fully_coupled_comparison_is_not_presented_as_a_result() -> None:
    with SOURCE_MAP.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_section = {row["result_or_section"]: row for row in rows}
    comparison = by_section["fixed-flow and fully coupled thermal-step comparison"]
    assert "scope_limits_20260730/scope_limits_summary.json" in comparison["source_data"]
    assert "hccb_p418_fully_coupled_steps_12" not in comparison["source_data"]
    assert "generated_fixed_vs_fully_coupled" not in comparison["source_data"]
    assert comparison["status"].endswith("unavailable_property_range")
    final_sections = by_section["final abstract discussion conclusions"]
    assert "fixed_vs_fully_coupled/summary.json" not in final_sections["source_data"]
    final_pdf = by_section["formal manuscript PDF"]
    assert "generated_fixed_vs_fully_coupled_steps.tex" not in final_pdf["source_data"]
    assert "generated_scope_limits.tex" in final_pdf["source_data"]
