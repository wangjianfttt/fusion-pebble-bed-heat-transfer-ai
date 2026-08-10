#!/usr/bin/env bash
# Regenerate and compile the condensed IJHMT manuscript from accepted evidence.

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RESULT_ROOT=${RESULT_ROOT:-${ROOT}/results}
MANUSCRIPT_DIR=${MANUSCRIPT_DIR:-${ROOT}/manuscript}
OUTPUT_RECORD=${OUTPUT_RECORD:-${RESULT_ROOT}/hccb_p418_manuscript_refresh_complete.json}
BUILD_SUPPLEMENT=${BUILD_SUPPLEMENT:-0}

TRANSIENT_ROOT=${RESULT_ROOT}/hccb_p418_physical_steps_12
TRANSIENT_COMPARISON=${TRANSIENT_ROOT}/model_comparison
STEADY_COMPARISON=${STEADY_COMPARISON:-${RESULT_ROOT}/hccb_p418_60_corrected_20260731_model_comparison_100epoch}
HIGH_RE_ROOT=${RESULT_ROOT}/hccb_p418_high_re_three_bounded_model_evaluation
HIGH_RE_COMPARISON=${HIGH_RE_ROOT}/comparison
PACKING_SUMMARY=${RESULT_ROOT}/hccb_p418_cross_packing_seed202_integral_9/summary.json
SCOPE_SUMMARY=${RESULT_ROOT}/hccb_p418_scope_limits_20260730/scope_limits_summary.json
TRANSPORT_CHECK=${RESULT_ROOT}/hccb_p418_helium_transport_lookup_20260802/openfoam13_direct_transport_build.json
DIRECT_COUPLED_FAILURE=${RESULT_ROOT}/hccb_p418_public_figure_data/direct_transport_scope_limit.json
MESH_SENSITIVITY_SUMMARY=${RESULT_ROOT}/hccb_p418_three_mesh_cht_sensitivity/summary.json
MESH_SENSITIVITY_ENGINEERING=${RESULT_ROOT}/hccb_p418_three_mesh_cht_sensitivity/engineering_observables.csv
MESH_SENSITIVITY_GCI=${RESULT_ROOT}/hccb_p418_three_mesh_cht_sensitivity/mesh_gci.csv
MESH_SENSITIVITY_VERIFICATION=${RESULT_ROOT}/hccb_p418_three_mesh_cht_sensitivity/formal_recovery_verification.json
UNCERTAINTY_ROOT=${RESULT_ROOT}/hccb_p418_final_uncertainty_summary
UNCERTAINTY_TEXT=${MANUSCRIPT_DIR}/generated_final_uncertainty_text.tex

mkdir -p "${RESULT_ROOT}/hccb_heat_ai_external_evidence"
python3 "${ROOT}/code/build_hccb_heat_ai_external_evidence.py" \
  > "${RESULT_ROOT}/hccb_heat_ai_external_evidence/build_stdout.json"

python3 "${ROOT}/code/build_hccb_p418_partial_cross_packing_text.py" \
  --summary "${PACKING_SUMMARY}" \
  --output "${MANUSCRIPT_DIR}/generated_cross_packing_integral_9.tex"

python3 "${ROOT}/code/build_hccb_p418_final_uncertainty_summary.py" \
  --mesh-summary "${MESH_SENSITIVITY_SUMMARY}" \
  --fixed-timestep-summary "${RESULT_ROOT}/hccb_p418_thermal_timestep_sensitivity/thermal_timestep_sensitivity.json" \
  --scope-limit-summary "${SCOPE_SUMMARY}" \
  --steady-seed-summary "${RESULT_ROOT}/hccb_p418_60_steady_seed_robustness_100epoch/summary.json" \
  --transient-seed-summary "${TRANSIENT_ROOT}/seed_robustness_pair_disjoint_stress_test/summary.json" \
  --transient-metrics "${TRANSIENT_COMPARISON}/physical_step_model_metrics.csv" \
  --cross-packing-summary "${PACKING_SUMMARY}" \
  --external-summary "${RESULT_ROOT}/hccb_heat_ai_external_evidence/summary.json" \
  --external-metrics "${RESULT_ROOT}/hccb_heat_ai_external_evidence/metrics.csv" \
  --output-dir "${UNCERTAINTY_ROOT}" \
  --tex-output "${UNCERTAINTY_TEXT}"

required=(
  "${MANUSCRIPT_DIR}/main.tex"
  "${MANUSCRIPT_DIR}/methods_condensed.tex"
  "${MANUSCRIPT_DIR}/results_condensed.tex"
  "${MANUSCRIPT_DIR}/elsarticle.cls"
  "${MANUSCRIPT_DIR}/elsarticle-num.bst"
  "${MANUSCRIPT_DIR}/references.bib"
  "${ROOT}/parameters/HCCB_P418_PARAMETER_AND_MODEL_SOURCES_CN.md"
  "${RESULT_ROOT}/hccb_p418_source_summary.json"
  "${RESULT_ROOT}/hccb_p418_60_sourceflow_r3_completed_physics/summary.json"
  "${RESULT_ROOT}/hccb_p418_60_sourceflow_r3_completed_physics/completed_case_physics.csv"
  "${RESULT_ROOT}/hccb_p418_60_sourceflow_r3_steady_hotspots/summary.json"
  "${STEADY_COMPARISON}/corrected_result_assembly.json"
  "${STEADY_COMPARISON}/steady_result_text.json"
  "${MANUSCRIPT_DIR}/generated_steady_model_comparison_validated.tex"
  "${MANUSCRIPT_DIR}/generated_steady_performance.tex"
  "${MANUSCRIPT_DIR}/generated_steady_result_text.tex"
  "${RESULT_ROOT}/hccb_p418_60_steady_seed_robustness_100epoch/summary.json"
  "${RESULT_ROOT}/hccb_p418_learning_curve_model_comparison_100epoch/learning_curve_summary.json"
  "${TRANSIENT_COMPARISON}/summary.json"
  "${TRANSIENT_COMPARISON}/physical_step_model_metrics.csv"
  "${TRANSIENT_COMPARISON}/transient_performance_table.json"
  "${TRANSIENT_COMPARISON}/transient_cost_table.json"
  "${TRANSIENT_ROOT}/seed_robustness_pair_disjoint_stress_test/summary.json"
  "${RESULT_ROOT}/hccb_p418_transient_learning_curve/summary.json"
  "${MANUSCRIPT_DIR}/generated_transient_performance.tex"
  "${MANUSCRIPT_DIR}/generated_transient_cost.tex"
  "${MANUSCRIPT_DIR}/generated_transient_result_text.tex"
  "${MANUSCRIPT_DIR}/generated_fused_chain_results.tex"
  "${MANUSCRIPT_DIR}/generated_fused_chain_text.tex"
  "${TRANSIENT_ROOT}/chained_initial_state/manuscript_table_summary.json"
  "${MANUSCRIPT_DIR}/generated_transient_model_comparison_validated.tex"
  "${MANUSCRIPT_DIR}/generated_openfoam_model_field_comparison_validated.tex"
  "${HIGH_RE_COMPARISON}/summary.json"
  "${HIGH_RE_COMPARISON}/aggregate_fixed_model_comparison.csv"
  "${HIGH_RE_COMPARISON}/per_curve_fixed_model_comparison.csv"
  "${MANUSCRIPT_DIR}/generated_high_re_comparison.tex"
  "${PACKING_SUMMARY}"
  "${MANUSCRIPT_DIR}/generated_cross_packing_integral_9.tex"
  "${SCOPE_SUMMARY}"
  "${TRANSPORT_CHECK}"
  "${DIRECT_COUPLED_FAILURE}"
  "${MESH_SENSITIVITY_SUMMARY}"
  "${MESH_SENSITIVITY_ENGINEERING}"
  "${MESH_SENSITIVITY_GCI}"
  "${MANUSCRIPT_DIR}/generated_scope_limits.tex"
  "${UNCERTAINTY_ROOT}/summary.json"
  "${UNCERTAINTY_ROOT}/uncertainty_components.csv"
  "${UNCERTAINTY_TEXT}"
  "${RESULT_ROOT}/hccb_heat_ai_external_evidence/summary.json"
  "${RESULT_ROOT}/hccb_heat_ai_external_evidence/metrics.csv"
  "${ROOT}/figures/hccb_p418_physical_model_domain.pdf"
  "${ROOT}/figures/hccb_p418_physical_response.pdf"
  "${ROOT}/figures/hccb_heat_ai_external_evidence.pdf"
  "${ROOT}/figures/hccb_p418_steady_model_comparison.pdf"
  "${ROOT}/figures/hccb_p418_transient_model_comparison.pdf"
  "${ROOT}/figures/hccb_p418_openfoam_model_field_comparison.pdf"
  "${ROOT}/figures/hccb_p418_seed202_integral_9.pdf"
)

if [[ ${BUILD_SUPPLEMENT} == 1 ]]; then
  required+=("${MANUSCRIPT_DIR}/supplement.tex")
fi

for path in "${required[@]}"; do
  if [[ ! -s ${path} ]]; then
    echo "formal manuscript input is missing or empty: ${path}" >&2
    exit 1
  fi
done

python3 "${ROOT}/code/verify_hccb_p418_three_mesh_recovery.py" \
  --root "${RESULT_ROOT}/hccb_p418_three_mesh_cht_sensitivity" \
  --output "${MESH_SENSITIVITY_VERIFICATION}"

python3 "${ROOT}/code/build_hccb_p418_scope_limit_text.py" \
  --summary "${SCOPE_SUMMARY}" \
  --transport-check "${TRANSPORT_CHECK}" \
  --direct-coupled-failure "${DIRECT_COUPLED_FAILURE}" \
  --output "${MANUSCRIPT_DIR}/generated_scope_limits.tex"

python3 "${ROOT}/code/build_hccb_p418_manuscript_values.py" \
  --project-root "${ROOT}" \
  --output "${MANUSCRIPT_DIR}/generated_results.tex"
python3 "${ROOT}/code/build_hccb_p418_inlet_dimensionless_envelope.py" \
  --manifest "${ROOT}/parameters/literature_parameter_manifest.csv" \
  --input-summary "${RESULT_ROOT}/hccb_p418_60_actual_case_input_check/summary.json" \
  --output-dir "${RESULT_ROOT}/hccb_p418_inlet_dimensionless_envelope" \
  --tex-output "${MANUSCRIPT_DIR}/generated_dimensionless_envelope.tex"
python3 "${ROOT}/code/build_hccb_p418_manuscript_model_table.py" \
  --registry "${ROOT}/parameters/hccb_p418_ai_architecture_sources.json" \
  --settings "${ROOT}/parameters/hccb_p418_model_numerical_settings.csv" \
  --output "${MANUSCRIPT_DIR}/generated_model_settings.tex"
python3 "${ROOT}/code/build_hccb_p418_data_split_table.py" \
  --steady-splits "${ROOT}/parameters/hccb_p418_model_splits.json" \
  --transient-splits "${ROOT}/parameters/hccb_p418_step_response_splits.json" \
  --transient-plan "${ROOT}/parameters/hccb_p418_transient_step_plan.json" \
  --tex-output "${MANUSCRIPT_DIR}/generated_data_splits.tex" \
  --summary-output "${RESULT_ROOT}/hccb_p418_data_split_table/summary.json"

FINAL_NARRATIVE_SUMMARY=${RESULT_ROOT}/hccb_p418_final_manuscript_narrative.json
python3 "${ROOT}/code/build_hccb_p418_final_narrative.py" \
  --steady-summary "${STEADY_COMPARISON}/steady_result_text.json" \
  --completed-physics-summary "${RESULT_ROOT}/hccb_p418_60_sourceflow_r3_completed_physics/summary.json" \
  --steady-hotspot-summary "${RESULT_ROOT}/hccb_p418_60_sourceflow_r3_steady_hotspots/summary.json" \
  --steady-seed-robustness "${RESULT_ROOT}/hccb_p418_60_steady_seed_robustness_100epoch/summary.json" \
  --transient-seed-robustness "${TRANSIENT_ROOT}/seed_robustness_pair_disjoint_stress_test/summary.json" \
  --steady-learning-curve "${RESULT_ROOT}/hccb_p418_learning_curve_model_comparison_100epoch/learning_curve_summary.json" \
  --transient-learning-curve "${RESULT_ROOT}/hccb_p418_transient_learning_curve/summary.json" \
  --loss-balancing-selection "${TRANSIENT_ROOT}/fixed_flow_loss_balancing_pair_disjoint_stress_test/selected_loss_balancing_method.json" \
  --transient-summary "${TRANSIENT_COMPARISON}/summary.json" \
  --transient-metrics "${TRANSIENT_COMPARISON}/physical_step_model_metrics.csv" \
  --transient-cost "${TRANSIENT_COMPARISON}/transient_cost_table.json" \
  --high-re-comparison "${HIGH_RE_COMPARISON}/summary.json" \
  --high-re-aggregate "${HIGH_RE_COMPARISON}/aggregate_fixed_model_comparison.csv" \
  --cross-packing-summary "${PACKING_SUMMARY}" \
  --external-evidence "${RESULT_ROOT}/hccb_heat_ai_external_evidence/summary.json" \
  --scope-limits "${SCOPE_SUMMARY}" \
  --abstract-output "${MANUSCRIPT_DIR}/generated_final_abstract.tex" \
  --discussion-output "${MANUSCRIPT_DIR}/generated_final_discussion.tex" \
  --conclusion-output "${MANUSCRIPT_DIR}/generated_final_conclusions.tex" \
  --summary-output "${FINAL_NARRATIVE_SUMMARY}"

CHINESE_READER_SUMMARY=${RESULT_ROOT}/hccb_p418_chinese_reader/summary.json
python3 "${ROOT}/code/build_hccb_p418_chinese_reader.py" \
  --final-narrative-summary "${FINAL_NARRATIVE_SUMMARY}" \
  --mesh-sensitivity-summary "${MESH_SENSITIVITY_SUMMARY}" \
  --output "${MANUSCRIPT_DIR}/P418_论文中文便读版.md" \
  --summary-output "${CHINESE_READER_SUMMARY}"

FIGURE_QUALITY_DIR=${RESULT_ROOT}/hccb_p418_final_figure_quality
python3 "${ROOT}/code/check_hccb_p418_final_figure_outputs.py" \
  --figure-dir "${ROOT}/figures" \
  --output "${FIGURE_QUALITY_DIR}/summary.json"

compile_log=${RESULT_ROOT}/hccb_p418_manuscript_compile.log
if command -v latexmk >/dev/null 2>&1; then
  (
    cd "${MANUSCRIPT_DIR}"
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
  ) > "${compile_log}" 2>&1
else
  (
    cd "${MANUSCRIPT_DIR}"
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    bibtex main
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
  ) > "${compile_log}" 2>&1
fi

if [[ ${BUILD_SUPPLEMENT} == 1 ]]; then
  supplement_log=${RESULT_ROOT}/hccb_p418_supplement_compile.log
  (
    cd "${MANUSCRIPT_DIR}"
    latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
  ) > "${supplement_log}" 2>&1
fi

IJHMT_CHECK_DIR=${RESULT_ROOT}/hccb_p418_ijhmt_submission_check
ijhmt_args=(
  --project-root "${ROOT}"
  --output-dir "${IJHMT_CHECK_DIR}"
  --require-complete
)
if [[ ${BUILD_SUPPLEMENT} == 1 ]]; then
  ijhmt_args+=(--require-supplement)
fi
python3 "${ROOT}/code/check_hccb_p418_ijhmt_submission.py" \
  "${ijhmt_args[@]}"

REPRODUCIBILITY_DIR=${RESULT_ROOT}/hccb_p418_reproducibility_manifest
PUBLIC_DATA_DIR=${RESULT_ROOT}/hccb_p418_public_data_release_preflight
python3 "${ROOT}/code/build_hccb_p418_public_training_manifest.py" \
  --input "${RESULT_ROOT}/hccb_p418_physical_steps_12/formal_training_jobs_workstation.json" \
  --output "${PUBLIC_DATA_DIR}/formal_training_manifest_public.json" \
  --source-root "${ROOT}"
python3 "${ROOT}/code/build_hccb_p418_public_data_release.py" \
  --project-root "${ROOT}" \
  --output-dir "${PUBLIC_DATA_DIR}"
python3 "${ROOT}/code/build_hccb_p418_reproducibility_manifest.py" \
  --project-root "${ROOT}" \
  --output-dir "${REPRODUCIBILITY_DIR}" \
  --require-source-complete

REPRODUCIBILITY_ARCHIVE=${REPRODUCIBILITY_DIR}/p418_reproduction_source.tar.gz
REPRODUCIBILITY_ARCHIVE_RECORD=${REPRODUCIBILITY_DIR}/source_archive_record.json
python3 "${ROOT}/code/package_hccb_p418_reproducibility_source.py" \
  --project-root "${ROOT}" \
  --manifest "${REPRODUCIBILITY_DIR}/manifest.json" \
  --output "${REPRODUCIBILITY_ARCHIVE}" \
  --record "${REPRODUCIBILITY_ARCHIVE_RECORD}"

FINAL_REQUIREMENTS_DIR=${RESULT_ROOT}/hccb_p418_final_scientific_requirements
python3 "${ROOT}/code/check_hccb_p418_final_scientific_requirements.py" \
  --project-root "${ROOT}" \
  --output-dir "${FINAL_REQUIREMENTS_DIR}" \
  --require-complete

SUBMISSION_BUNDLE_DIR=${RESULT_ROOT}/hccb_p418_ijhmt_submission_bundle
SUBMISSION_BUNDLE_RECORD=${SUBMISSION_BUNDLE_DIR}/record.json
python3 "${ROOT}/code/package_hccb_p418_ijhmt_submission.py" \
  --project-root "${ROOT}" \
  --output-dir "${SUBMISSION_BUNDLE_DIR}" \
  --record "${SUBMISSION_BUNDLE_RECORD}" \
  --require-complete

SUBMISSION_BUNDLE_VERIFY_RECORD=${SUBMISSION_BUNDLE_DIR}/submission_bundle_verification.json
submission_verify_args=(
  --bundle-dir "${SUBMISSION_BUNDLE_DIR}"
  --output "${SUBMISSION_BUNDLE_VERIFY_RECORD}"
  --require-complete
)
if [[ -d ${ROOT}/runtime/p418_texmf ]]; then
  submission_verify_args+=(--texinputs "${ROOT}/runtime/p418_texmf")
fi
python3 "${ROOT}/code/verify_hccb_p418_ijhmt_submission_bundle.py" \
  "${submission_verify_args[@]}"

record_inputs=(
  "${required[@]}"
  "${MANUSCRIPT_DIR}/generated_results.tex"
  "${MANUSCRIPT_DIR}/generated_dimensionless_envelope.tex"
  "${MANUSCRIPT_DIR}/generated_model_settings.tex"
  "${MANUSCRIPT_DIR}/generated_data_splits.tex"
  "${UNCERTAINTY_ROOT}/summary.json"
  "${UNCERTAINTY_ROOT}/uncertainty_components.csv"
  "${UNCERTAINTY_TEXT}"
  "${MANUSCRIPT_DIR}/generated_final_abstract.tex"
  "${MANUSCRIPT_DIR}/generated_final_discussion.tex"
  "${MANUSCRIPT_DIR}/generated_final_conclusions.tex"
  "${FINAL_NARRATIVE_SUMMARY}"
  "${MANUSCRIPT_DIR}/P418_论文中文便读版.md"
  "${CHINESE_READER_SUMMARY}"
  "${FIGURE_QUALITY_DIR}/summary.json"
  "${IJHMT_CHECK_DIR}/summary.json"
  "${REPRODUCIBILITY_DIR}/manifest.json"
  "${REPRODUCIBILITY_ARCHIVE}"
  "${REPRODUCIBILITY_ARCHIVE_RECORD}"
  "${FINAL_REQUIREMENTS_DIR}/summary.json"
  "${SUBMISSION_BUNDLE_RECORD}"
  "${SUBMISSION_BUNDLE_VERIFY_RECORD}"
  "${SUBMISSION_BUNDLE_DIR}/p418_ijhmt_upload_bundle.zip"
)

python3 - "${OUTPUT_RECORD}" "${MANUSCRIPT_DIR}/main.pdf" \
  "${BUILD_SUPPLEMENT}" "${record_inputs[@]}" <<'PY'
import hashlib
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1]).resolve()
pdf = pathlib.Path(sys.argv[2]).resolve()
supplement_built = sys.argv[3] == "1"
inputs = [pathlib.Path(value).resolve() for value in sys.argv[4:]]
if not pdf.is_file() or pdf.stat().st_size == 0:
    raise SystemExit("compiled manuscript PDF is missing or empty")
records = []
seen = set()
for path in inputs:
    if path in seen:
        continue
    seen.add(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"recorded manuscript input is missing: {path}")
    records.append(
        {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
payload = {
    "status": "completed_p418_formal_manuscript_refresh",
    "scientific_scope": (
        "steady and fixed-hydrodynamic thermal modelling with explicit "
        "full-domain and fully-coupled limitations"
    ),
    "manuscript_inputs": records,
    "compiled_pdf": {
        "path": str(pdf),
        "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
    },
    "supplement_built": supplement_built,
    "new_physical_parameters": [],
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(output)
PY

echo "P418 manuscript refreshed: ${MANUSCRIPT_DIR}/main.pdf"
