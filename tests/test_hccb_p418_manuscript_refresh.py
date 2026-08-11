from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/run_hccb_p418_manuscript_refresh.sh"
FORMAL = ROOT / "code/run_hccb_p418_formal_calculations.sh"
MANUSCRIPT = ROOT / "manuscript/main.tex"
METHODS = ROOT / "manuscript/methods_condensed.tex"
RESULTS = ROOT / "manuscript/results_condensed.tex"


def test_refresh_requires_completed_results_before_compilation() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    required_check = text.index('for path in "${required[@]}"')
    value_generation = text.index("build_hccb_p418_manuscript_values.py")
    dimensionless = text.index("build_hccb_p418_inlet_dimensionless_envelope.py")
    compile_step = text.index("latexmk -pdf")
    record_step = text.index('"status": "completed_p418_formal_manuscript_refresh"')
    assert required_check < value_generation < dimensionless < compile_step < record_step
    assert "hccb_p418_source_summary.json" in text
    assert "HCCB_P418_PARAMETER_AND_MODEL_SOURCES_CN.md" in text
    assert "generated_dimensionless_envelope.tex" in text
    assert "generated_data_splits.tex" in text
    assert "generated_transient_cost.tex" in text
    assert "generated_high_re_comparison.tex" in text
    assert (
        "HIGH_RE_ROOT=${RESULT_ROOT}/hccb_p418_high_re_three_bounded_model_evaluation"
        in text
    )
    assert '"${HIGH_RE_COMPARISON}/summary.json"' in text
    assert '"${HIGH_RE_COMPARISON}/aggregate_fixed_model_comparison.csv"' in text
    assert "--high-re-comparison" in text
    assert "--high-re-aggregate" in text
    assert "generated_cross_packing_integral_9.tex" in text
    assert "--cross-packing-summary" in text
    assert "--scope-limits" in text
    assert "check_hccb_p418_final_figure_outputs.py" in text
    assert "hccb_p418_final_figure_quality" in text
    assert "generated_steady_result_text.tex" in text
    assert (
        "STEADY_COMPARISON=${STEADY_COMPARISON:-${RESULT_ROOT}/"
        "hccb_p418_60_corrected_20260731_model_comparison_100epoch}"
        in text
    )
    assert '"${STEADY_COMPARISON}/corrected_result_assembly.json"' in text
    assert '"${STEADY_COMPARISON}/steady_result_text.json"' in text
    assert "generated_steady_model_comparison_validated.tex" in text
    assert "generated_transient_model_comparison_validated.tex" in text
    assert "generated_fused_chain_text.tex" in text
    assert "chained_initial_state/manuscript_table_summary.json" in text
    assert "generated_openfoam_model_field_comparison_validated.tex" in text
    assert "hccb_p418_60_model_comparison_100epoch/steady_result_text.json" not in text
    assert "build_hccb_p418_data_split_table.py" in text
    assert "hccb_p418_seed202_integral_9.pdf" in text
    assert "hccb_p418_openfoam_model_field_comparison.pdf" in text
    assert "hccb_p418_three_mesh_cht_sensitivity" in text
    assert 'MESH_SENSITIVITY_ENGINEERING=${RESULT_ROOT}/hccb_p418_three_mesh_cht_sensitivity/engineering_observables.csv' in text
    assert 'MESH_SENSITIVITY_GCI=${RESULT_ROOT}/hccb_p418_three_mesh_cht_sensitivity/mesh_gci.csv' in text
    assert "verify_hccb_p418_three_mesh_recovery.py" in text
    assert "formal_recovery_verification.json" in text
    assert '"${MESH_SENSITIVITY_ENGINEERING}"' in text
    assert '"${MESH_SENSITIVITY_GCI}"' in text
    assert "build_hccb_p418_chinese_reader.py" in text
    assert "P418_论文中文便读版.md" in text
    assert "package_hccb_p418_reproducibility_source.py" in text
    assert "package_hccb_p418_processed_data_release.py" in text
    assert "p418_processed_data_release.zip" in text
    assert "processed_data_archive_record.json" in text
    assert "package_hccb_p418_ijhmt_submission.py" in text
    processed = text.index("package_hccb_p418_processed_data_release.py")
    journal_check = text.index("check_hccb_p418_ijhmt_submission.py")
    upload_bundle = text.index("package_hccb_p418_ijhmt_submission.py")
    assert processed < journal_check < upload_bundle
    assert "p418_ijhmt_upload_bundle.zip" in text
    assert "fixed_vs_fully_coupled" not in text
    assert "seed303" not in text
    assert '"new_physical_parameters": []' in text


def test_formal_calculation_refreshes_manuscript_last() -> None:
    text = FORMAL.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(FORMAL)], check=True)
    final_record = text.index("hccb_p418_formal_calculations_complete.json")
    refresh = text.index("run_hccb_p418_manuscript_refresh.sh")
    final_echo = text.index("P418 formal heat-transfer calculations are complete")
    assert final_record < refresh < final_echo


def test_condensed_main_contains_the_decisive_evidence() -> None:
    main = MANUSCRIPT.read_text(encoding="utf-8")
    methods = METHODS.read_text(encoding="utf-8")
    methods_normalized = " ".join(methods.split())
    results = RESULTS.read_text(encoding="utf-8")
    assert r"\input{methods_condensed}" in main
    assert r"\input{results_condensed}" in main
    assert r"\iffalse" not in main
    assert "seed303" not in main
    assert "three packing realizations" not in main
    assert "A second independently generated spherical-pebble arrangement" in main
    assert "The final discussion will" not in main
    assert "The final model ranking will" not in main
    assert "does not by itself establish a final model ranking" in main
    assert "coarse, medium and fine fluid--solid meshes" in methods_normalized
    assert "$Re_{p,\\mathrm{in}}=0.078$--$2.40$" in methods
    assert "$Pe_{p,\\mathrm{in}}=0.051$--$1.62$" in methods
    assert "$Re_{p,\\mathrm{AVE}}<1.8$" in methods
    assert "passed the basic mesh checks" in methods_normalized
    assert "three-grid GCI" in methods
    assert "Supplementary Material" not in methods
    assert r"\IfFileExists{generated_scope_limits.tex}" in results
    assert "fine-grid GCI is 1.96" in results
    assert "medium-to-fine thermal-curve" in results
    assert "below 0.043" in results
    assert r"\IfFileExists{generated_mesh_sensitivity.tex}" in results
    assert r"\IfFileExists{generated_timestep_sensitivity.tex}" not in results
    assert r"\IfFileExists{generated_steady_performance.tex}" in results
    assert r"\IfFileExists{generated_steady_model_comparison_validated.tex}" in results
    assert r"\IfFileExists{generated_transient_performance.tex}" in results
    assert (
        r"\IfFileExists{generated_transient_model_comparison_validated.tex}"
        in results
    )
    assert (
        r"\IfFileExists{generated_openfoam_model_field_comparison_validated.tex}"
        in results
    )
    assert r"\IfFileExists{generated_high_re_comparison.tex}" in results
    assert "hccb_p418_physical_response.pdf" in results
    assert "hccb_heat_ai_external_evidence.pdf" in results
    assert "hccb_p418_steady_model_comparison.pdf" in results
    assert "hccb_p418_transient_model_comparison.pdf" in results
    assert "hccb_p418_openfoam_model_field_comparison.pdf" in results
    assert r"\PFieldModelLabel" in results
    assert r"\PFieldFluidRMSE" in results
    assert r"\providecommand{\PFieldModelLabel}{}" in results
    assert r"\providecommand{\PFieldFluidRMSE}{}" in results
    assert r"\providecommand{\PFieldSolidRMSE}{}" in results
    assert r"\providecommand{\PFieldFluidMaxError}{}" in results
    assert r"\providecommand{\PFieldSolidMaxError}{}" in results
    assert "12.80" not in results
    assert "401.96" not in results
    assert r"\input{generated_openfoam_model_field_comparison_validated}" in results
    assert "selected using validation trajectories only" in results
    assert "Most of the section" not in results
    assert "hccb_p418_seed202_integral_9.pdf" in results
    assert "symmetric-logarithmic" in results
    assert "Fully coupled" in results
    assert "specified helium-property pressure range" in results
    assert "270,441 trainable parameters" in methods
    assert "two four-head Physics-Attention" in methods
    assert "initial learning rate of $10^{-3}$" in methods
    assert "$5\\mathcal L_T+\\mathcal L_Q+\\mathcal L_E$" in methods
    assert "Checkpoints are chosen only" in methods


def test_supplement_is_opt_in_only() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "BUILD_SUPPLEMENT=${BUILD_SUPPLEMENT:-0}" in text
    assert 'required+=("${MANUSCRIPT_DIR}/supplement.tex")' in text
    assert "latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex" in text
    assert "ijhmt_args+=(--require-supplement)" in text
    assert "build_hccb_p418_supplement_inputs.py" not in text
    assert "supplement_built" in text
