#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RESULT_PREFIX=${RESULT_PREFIX:-${ROOT}/results/hccb_p418_60_sourceflow_r3}
RESULT_NAMESPACE=${RESULT_NAMESPACE:-hccb_p418_60}
EPOCHS=${EPOCHS:-100}
COMPARISON_OUTPUT_DIR=${COMPARISON_OUTPUT_DIR:-${ROOT}/results/${RESULT_NAMESPACE}_model_comparison_${EPOCHS}epoch}
REGIONAL_TOPOLOGY=${REGIONAL_TOPOLOGY:-${ROOT}/results/hccb_p418_regional_topology_r2/regional_topology.npz}
STATE_TARGETS=${STATE_TARGETS:-${RESULT_PREFIX}_regional_state_targets/regional_state_targets.npz}
MASS_TARGETS=${MASS_TARGETS:-${RESULT_PREFIX}_regional_mass_flux_targets/regional_mass_flux_targets.npz}
TRAINING_STATISTICS=${TRAINING_STATISTICS:-${RESULT_PREFIX}_training_statistics.json}
ARCHITECTURES=${ARCHITECTURES:-pinn_data_only pinn graph transolver}
SPLIT_NAMES=${SPLIT_NAMES:-interleaved_all_ranges temperature_extrapolation velocity_extrapolation heat_source_interpolation heat_source_extrapolation}

required=(
  "${COMPARISON_OUTPUT_DIR}/model_comparison.csv"
  "${COMPARISON_OUTPUT_DIR}/native_cell_model_comparison.json"
  "${COMPARISON_OUTPUT_DIR}/thermal_regime_split_coverage.json"
)
for path in "${required[@]}"; do
  if [[ ! -f ${path} ]]; then
    echo "missing completed comparison input: ${path}" >&2
    exit 1
  fi
done

python3 "${ROOT}/code/build_hccb_p418_native_cell_model_table.py" \
  --comparison-summary "${COMPARISON_OUTPUT_DIR}/native_cell_model_comparison.json" \
  --output "${ROOT}/manuscript/generated_native_cell_performance.tex" \
  --summary "${COMPARISON_OUTPUT_DIR}/native_cell_model_table.json"

python3 "${ROOT}/code/plot_hccb_p418_steady_model_comparison.py" \
  --comparison-csv "${COMPARISON_OUTPUT_DIR}/model_comparison.csv" \
  --output-dir "${ROOT}/figures"

python3 "${ROOT}/code/build_hccb_p418_steady_performance_table.py" \
  --comparison-csv "${COMPARISON_OUTPUT_DIR}/model_comparison.csv" \
  --output "${ROOT}/manuscript/generated_steady_performance.tex" \
  --summary "${COMPARISON_OUTPUT_DIR}/steady_performance_table.json"

python3 "${ROOT}/code/build_hccb_p418_steady_result_text.py" \
  --comparison-csv "${COMPARISON_OUTPUT_DIR}/model_comparison.csv" \
  --thermal-regime-coverage "${COMPARISON_OUTPUT_DIR}/thermal_regime_split_coverage.json" \
  --output "${ROOT}/manuscript/generated_steady_result_text.tex" \
  --summary "${COMPARISON_OUTPUT_DIR}/steady_result_text.json"

printf '%% Generated only after all five corrected steady-model results were assembled.\n' \
  > "${ROOT}/manuscript/generated_steady_model_comparison_validated.tex"

python3 "${ROOT}/code/summarize_hccb_p418_thermal_regime_model_errors.py" \
  --physical-csv "${RESULT_PREFIX}_completed_physics/completed_case_physics.csv" \
  --results-root "${ROOT}/results" \
  --result-prefix "${RESULT_NAMESPACE}" \
  --epochs "${EPOCHS}" \
  --architectures response_surface ${ARCHITECTURES} \
  --splits ${SPLIT_NAMES} \
  --output-dir "${COMPARISON_OUTPUT_DIR}"

for architecture in response_surface ${ARCHITECTURES}; do
  model_output=${ROOT}/results/${RESULT_NAMESPACE}_${architecture}_interleaved_all_ranges_${EPOCHS}epoch
  MODEL_OUTPUT="${model_output}" \
  MODEL_NAME="${architecture} P418 interleaved-all-ranges" \
  SPLIT_NAME=interleaved_all_ranges \
  RESULT_PREFIX="${RESULT_PREFIX}" \
  REGIONAL_TOPOLOGY="${REGIONAL_TOPOLOGY}" \
  REFERENCE_STATE_TARGETS="${STATE_TARGETS}" \
  MASS_TARGETS="${MASS_TARGETS}" \
  TRAINING_STATISTICS="${TRAINING_STATISTICS}" \
  OUTPUT_DIR="${COMPARISON_OUTPUT_DIR}/experimental_${architecture}" \
    bash "${ROOT}/code/run_hccb_p418_learned_model_experimental_comparison.sh"
done

echo "P418 steady-model paper postprocessing completed without model training."
