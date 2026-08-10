#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RESULT_PREFIX=${RESULT_PREFIX:-${ROOT}/results/hccb_p418_60_sourceflow_r3}
SOURCE_NAMESPACE=${SOURCE_NAMESPACE:-hccb_p418_60}
CORRECTED_NAMESPACE=${CORRECTED_NAMESPACE:-hccb_p418_60_normfix_20260731}
OUTPUT_NAMESPACE=${OUTPUT_NAMESPACE:-hccb_p418_60_corrected_20260731}
EPOCHS=${EPOCHS:-100}
SPLITS=${SPLITS:-${ROOT}/parameters/hccb_p418_model_splits.json}
TRAINING_STATISTICS=${TRAINING_STATISTICS:-${RESULT_PREFIX}_training_statistics.json}
STATE_TARGETS=${STATE_TARGETS:-${RESULT_PREFIX}_regional_state_targets/regional_state_targets.npz}
MASS_TARGETS=${MASS_TARGETS:-${RESULT_PREFIX}_regional_mass_flux_targets/regional_mass_flux_targets.npz}
ENERGY_TARGETS=${ENERGY_TARGETS:-${RESULT_PREFIX}_regional_energy_flux_targets/regional_energy_flux_targets.npz}
DATASET_INDEX=${DATASET_INDEX:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3_dataset/dataset_index.json}
SUBFACE_GEOMETRY=${SUBFACE_GEOMETRY:-${ROOT}/results/hccb_p418_subface_residual_geometry_r2/subface_residual_geometry.npz}
COMPARISON_OUTPUT_DIR=${COMPARISON_OUTPUT_DIR:-${ROOT}/results/${OUTPUT_NAMESPACE}_model_comparison_${EPOCHS}epoch}
METHODS=(response_surface pinn_data_only pinn graph transolver)
SPLIT_NAMES=(
  interleaved_all_ranges
  temperature_extrapolation
  velocity_extrapolation
  heat_source_interpolation
  heat_source_extrapolation
)

python3 "${ROOT}/code/assemble_hccb_p418_corrected_steady_results.py" \
  --results-root "${ROOT}/results" \
  --source-namespace "${SOURCE_NAMESPACE}" \
  --corrected-namespace "${CORRECTED_NAMESPACE}" \
  --output-namespace "${OUTPUT_NAMESPACE}" \
  --corrected-split heat_source_extrapolation \
  --epochs "${EPOCHS}" \
  --manifest "${COMPARISON_OUTPUT_DIR}/corrected_result_assembly.json"

python3 "${ROOT}/code/validate_hccb_p418_steady_comparison_inputs.py" \
  --state-targets "${STATE_TARGETS}" \
  --mass-targets "${MASS_TARGETS}" \
  --energy-targets "${ENERGY_TARGETS}" \
  --split-file "${SPLITS}" \
  --training-statistics "${TRAINING_STATISTICS}" \
  --expected-cases 60 \
  --output "${COMPARISON_OUTPUT_DIR}/common_input_check.json"

python3 "${ROOT}/code/summarize_hccb_p418_thermal_regime_split_coverage.py" \
  --physical-csv "${RESULT_PREFIX}_completed_physics/completed_case_physics.csv" \
  --split-file "${SPLITS}" \
  --output-dir "${COMPARISON_OUTPUT_DIR}"

python3 "${ROOT}/code/summarize_hccb_p418_60_model_comparison.py" \
  --results-root "${ROOT}/results" \
  --result-prefix "${OUTPUT_NAMESPACE}" \
  --epochs "${EPOCHS}" \
  --architectures "${METHODS[@]}" \
  --splits "${SPLIT_NAMES[@]}" \
  --split-file "${SPLITS}" \
  --output-dir "${COMPARISON_OUTPUT_DIR}"

python3 "${ROOT}/code/assess_hccb_p418_training_convergence.py" \
  --results-root "${ROOT}/results" \
  --result-prefix "${OUTPUT_NAMESPACE}" \
  --epochs "${EPOCHS}" \
  --architectures pinn_data_only pinn graph transolver \
  --splits "${SPLIT_NAMES[@]}" \
  --architecture-registry "${ROOT}/parameters/hccb_p418_ai_architecture_sources.json" \
  --output-dir "${COMPARISON_OUTPUT_DIR}"

native_result_args=()
for method in "${METHODS[@]}"; do
  model_output=${ROOT}/results/${OUTPUT_NAMESPACE}_${method}_interleaved_all_ranges_${EPOCHS}epoch
  native_output=${COMPARISON_OUTPUT_DIR}/native_cell_${method}_interleaved_all_ranges
  python3 "${ROOT}/code/evaluate_hccb_p418_native_cell_prediction.py" \
    --dataset-index "${DATASET_INDEX}" \
    --subface-geometry "${SUBFACE_GEOMETRY}" \
    --regional-state-targets "${STATE_TARGETS}" \
    --regional-predictions "${model_output}/test_regional_predictions.npz" \
    --training-statistics "${TRAINING_STATISTICS}" \
    --split-name interleaved_all_ranges \
    --output-dir "${native_output}"
  native_result_args+=(--result "${method}=${native_output}/summary.json")
done

python3 "${ROOT}/code/summarize_hccb_p418_native_cell_predictions.py" \
  "${native_result_args[@]}" \
  --output-dir "${COMPARISON_OUTPUT_DIR}"

RESULT_NAMESPACE="${OUTPUT_NAMESPACE}" \
COMPARISON_OUTPUT_DIR="${COMPARISON_OUTPUT_DIR}" \
EPOCHS="${EPOCHS}" \
  bash "${ROOT}/code/run_hccb_p418_60_model_postprocess_only.sh"

echo "Corrected P418 steady comparison assembled without model retraining."
