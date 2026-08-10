#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RESULT_PREFIX=${RESULT_PREFIX:-${ROOT}/results/hccb_p418_60_sourceflow_r3}
DATASET_INDEX=${DATASET_INDEX:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3_dataset/dataset_index.json}
BASE_SPLITS=${BASE_SPLITS:-${ROOT}/parameters/hccb_p418_model_splits.json}
SPLITS=${SPLITS:-${ROOT}/parameters/hccb_p418_learning_curve_splits.json}
TRAINING_STATISTICS=${TRAINING_STATISTICS:-${RESULT_PREFIX}_learning_curve_training_statistics.json}
RESULT_NAMESPACE=${RESULT_NAMESPACE:-hccb_p418_learning_curve}
COMPARISON_OUTPUT_DIR=${COMPARISON_OUTPUT_DIR:-${ROOT}/results/${RESULT_NAMESPACE}_model_comparison_${EPOCHS:-100}epoch}
SPLIT_NAMES=${SPLIT_NAMES:-"learning_curve_n09 learning_curve_n18 learning_curve_n27 learning_curve_n36"}
ARCHITECTURES=${ARCHITECTURES:-"pinn_data_only pinn graph transolver"}
EPOCHS=${EPOCHS:-100}
THREADS=${THREADS:-16}
DEVICE=${DEVICE:-cpu}

python3 "${ROOT}/code/build_hccb_p418_learning_curve_splits.py" \
  --base-splits "${BASE_SPLITS}" \
  --output "${SPLITS}"

python3 "${ROOT}/code/build_hccb_p418_training_statistics.py" \
  --dataset-index "${DATASET_INDEX}" \
  --split-file "${SPLITS}" \
  --output "${TRAINING_STATISTICS}" \
  > "${RESULT_PREFIX}_learning_curve_training_statistics.log"

SPLITS="${SPLITS}" \
TRAINING_STATISTICS="${TRAINING_STATISTICS}" \
RESULT_NAMESPACE="${RESULT_NAMESPACE}" \
COMPARISON_OUTPUT_DIR="${COMPARISON_OUTPUT_DIR}" \
SPLIT_NAMES="${SPLIT_NAMES}" \
ARCHITECTURES="${ARCHITECTURES}" \
EPOCHS="${EPOCHS}" \
THREADS="${THREADS}" \
DEVICE="${DEVICE}" \
  bash "${ROOT}/code/run_hccb_p418_60_model_comparison.sh"

python3 "${ROOT}/code/summarize_hccb_p418_learning_curve_efficiency.py" \
  --comparison-csv "${COMPARISON_OUTPUT_DIR}/model_comparison.csv" \
  --split-file "${SPLITS}" \
  --matrix-root "${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3" \
  --output-dir "${COMPARISON_OUTPUT_DIR}" \
  --expected-training-counts 9 18 27 36 \
  --expected-architectures response_surface ${ARCHITECTURES} \
  --expected-split-names ${SPLIT_NAMES} \
  --expected-validation-count 12 \
  --expected-test-count 12

python3 "${ROOT}/code/plot_hccb_p418_learning_curve.py" \
  --input-csv "${COMPARISON_OUTPUT_DIR}/learning_curve_efficiency.csv" \
  --output-dir "${COMPARISON_OUTPUT_DIR}"
