#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
EPOCHS=${EPOCHS:-100}
DEVICE=${DEVICE:-cuda}
STEADY_SPLIT=${STEADY_SPLIT:-interleaved_all_ranges}
STEP_SPLITS=${STEP_SPLITS:-"direction_down_test direction_up_test pair_disjoint_stress_test"}
STEADY_SUMMARY=${STEADY_SUMMARY:-${ROOT}/results/hccb_p418_60_pinn_${STEADY_SPLIT}_${EPOCHS}epoch/summary.json}
STATE_TARGETS=${STATE_TARGETS:-${ROOT}/results/hccb_p418_60_sourceflow_r3_regional_state_targets/regional_state_targets.npz}
STEADY_STATISTICS=${STEADY_STATISTICS:-${ROOT}/results/hccb_p418_60_sourceflow_r3_training_statistics.json}
TRANSIENT_RESULT_ROOT=${TRANSIENT_RESULT_ROOT:-${ROOT}/results/hccb_p418_physical_steps_12}
TRANSIENT_DATASET=${TRANSIENT_DATASET:-${TRANSIENT_RESULT_ROOT}/regional_sequences/dataset_index.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-${TRANSIENT_RESULT_ROOT}/chained_initial_state}
RESIDUAL_GEOMETRY=${RESIDUAL_GEOMETRY:-${ROOT}/results/hccb_p418_subface_residual_geometry_r2/subface_residual_geometry.npz}
ENERGY_DEVICE=${ENERGY_DEVICE:-cpu}
DIFFUSION_DEVICE=${DIFFUSION_DEVICE:-cuda}

python3 "${ROOT}/code/build_hccb_p418_chain_split_manifest.py" \
  --steady-splits "${ROOT}/parameters/hccb_p418_model_splits.json" \
  --transient-plan "${ROOT}/parameters/hccb_p418_transient_step_plan.json" \
  --transient-splits "${ROOT}/parameters/hccb_p418_step_response_splits.json" \
  --steady-split-name "${STEADY_SPLIT}" \
  --output-csv "${OUTPUT_ROOT}/steady_transient_split_manifest.csv" \
  --output-json "${OUTPUT_ROOT}/steady_transient_split_summary.json"

for required in "${STEADY_SUMMARY}" "${STATE_TARGETS}" "${STEADY_STATISTICS}" "${TRANSIENT_DATASET}" "${RESIDUAL_GEOMETRY}"; do
  if [[ ! -f ${required} ]]; then
    echo "missing completed input: ${required}" >&2
    exit 1
  fi
done

for split_name in ${STEP_SPLITS}; do
  transient_summary=${TRANSIENT_RESULT_ROOT}/regional_graph_transformer_bounded_physics_${split_name}/summary.json
  diffusion_summary=${TRANSIENT_RESULT_ROOT}/temporal_diffusion_${split_name}/summary.json
  if [[ ! -f ${transient_summary} ]]; then
    echo "missing completed transient graph-Transformer: ${transient_summary}" >&2
    exit 1
  fi
  if [[ ! -f ${diffusion_summary} ]]; then
    echo "missing completed diffusion refiner: ${diffusion_summary}" >&2
    exit 1
  fi
  python3 "${ROOT}/code/evaluate_hccb_p418_chained_initial_state.py" \
    --transient-summary "${transient_summary}" \
    --transient-dataset-index "${TRANSIENT_DATASET}" \
    --steady-summary "${STEADY_SUMMARY}" \
    --steady-state-targets "${STATE_TARGETS}" \
    --steady-training-statistics "${STEADY_STATISTICS}" \
    --steady-split-name "${STEADY_SPLIT}" \
    --role test \
    --device "${DEVICE}" \
    --output-dir "${OUTPUT_ROOT}/${split_name}"
  python3 "${ROOT}/code/evaluate_hccb_p418_temporal_energy_balance.py" \
    --model-summary "${OUTPUT_ROOT}/${split_name}/summary.json" \
    --dataset-index "${TRANSIENT_DATASET}" \
    --residual-geometry "${RESIDUAL_GEOMETRY}" \
    --roles test \
    --device "${ENERGY_DEVICE}" \
    --output "${OUTPUT_ROOT}/${split_name}/energy_balance_summary.json"
  python3 "${ROOT}/code/evaluate_hccb_p418_chained_diffusion.py" \
    --chained-summary "${OUTPUT_ROOT}/${split_name}/summary.json" \
    --diffusion-summary "${diffusion_summary}" \
    --role test \
    --device "${DIFFUSION_DEVICE}" \
    --output-dir "${OUTPUT_ROOT}/${split_name}/diffusion"
  python3 "${ROOT}/code/evaluate_hccb_p418_temporal_energy_balance.py" \
    --model-summary "${OUTPUT_ROOT}/${split_name}/diffusion/summary.json" \
    --dataset-index "${TRANSIENT_DATASET}" \
    --residual-geometry "${RESIDUAL_GEOMETRY}" \
    --roles test \
    --device "${ENERGY_DEVICE}" \
    --output "${OUTPUT_ROOT}/${split_name}/diffusion/energy_balance_summary.json"
  python3 "${ROOT}/code/summarize_hccb_p418_fused_chain.py" \
    --chained-summary "${OUTPUT_ROOT}/${split_name}/summary.json" \
    --chained-energy "${OUTPUT_ROOT}/${split_name}/energy_balance_summary.json" \
    --diffusion-summary "${OUTPUT_ROOT}/${split_name}/diffusion/summary.json" \
    --diffusion-energy "${OUTPUT_ROOT}/${split_name}/diffusion/energy_balance_summary.json" \
    --output "${OUTPUT_ROOT}/${split_name}/fused_chain_summary.json"
done

python3 "${ROOT}/code/build_hccb_p418_fused_chain_table.py" \
  --result-root "${OUTPUT_ROOT}" \
  --splits ${STEP_SPLITS} \
  --output "${ROOT}/manuscript/generated_fused_chain_results.tex" \
  --text-output "${ROOT}/manuscript/generated_fused_chain_text.tex" \
  --summary "${OUTPUT_ROOT}/manuscript_table_summary.json"
