#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXECUTE=${EXECUTE:-0}
DATASET_INDEX=${DATASET_INDEX:-${ROOT}/results/hccb_p418_step_responses/regional_sequences/dataset_index.json}
RESIDUAL_GEOMETRY=${RESIDUAL_GEOMETRY:-${ROOT}/results/hccb_p418_step_responses/regional_sequences/subface_residual_geometry.npz}
SPLITS=${SPLITS:-${ROOT}/parameters/hccb_p418_transient_learning_curve_splits.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/results/hccb_p418_transient_learning_curve}
SEED=${SEED:-20260717}
SPLIT_NAMES=${SPLIT_NAMES:-"transient_learning_n03_up transient_learning_n03_down transient_learning_n06_both"}

printf 'P418 transient training-curve sensitivity\n'
printf 'dataset=%s\n' "${DATASET_INDEX}"
printf 'splits=%s\n' "${SPLITS}"
printf 'model=physics-constrained factorized graph-Transformer\n'
printf 'seed=%s\n' "${SEED}"
printf 'split_names=%s\n' "${SPLIT_NAMES}"
printf 'execute=%s\n' "${EXECUTE}"

if [[ ${EXECUTE} != 1 ]]; then
  printf 'No training started. Set EXECUTE=1 only after all 12 physical curves are complete.\n'
  exit 0
fi

if [[ -e "${ROOT}/control/PAUSE_NEW_P418_CASES_FOR_CLOUD_MIGRATION" ]]; then
  printf 'P418 calculations remain paused for cloud migration.\n' >&2
  exit 2
fi
if [[ ! -f "${DATASET_INDEX}" || ! -f "${RESIDUAL_GEOMETRY}" ]]; then
  printf 'Complete 12-curve regional data are unavailable.\n' >&2
  exit 2
fi

for split_name in ${SPLIT_NAMES}; do
  output=${OUTPUT_ROOT}/${split_name}
  python3 "${ROOT}/code/train_hccb_p418_spatiotemporal_regional_operator.py" \
    --dataset-index "${DATASET_INDEX}" \
    --splits "${SPLITS}" \
    --split-name "${split_name}" \
    --residual-geometry "${RESIDUAL_GEOMETRY}" \
    --output-dir "${output}" \
    --run-role formal_factorized \
    --physics-mode energy_and_flux \
    --spatial-temporal-mode factorized_static_spatial \
    --seed "${SEED}" \
    --resume
done
