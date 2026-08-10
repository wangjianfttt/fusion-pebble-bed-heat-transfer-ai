#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
EPOCHS=${EPOCHS:-1}
THREADS=${THREADS:-8}
DEVICE=${DEVICE:-cpu}

# Eight completed 200/300 s fields check the full model path only. They are
# not a formal train/test sample and must not be used to claim model accuracy.
env \
  ROOT="${ROOT}" \
  RESULT_PREFIX="${ROOT}/results/hccb_p418_mixed_endpoint_smoke" \
  RESULT_NAMESPACE="hccb_p418_mixed_endpoint_smoke_model" \
  COMPARISON_OUTPUT_DIR="${ROOT}/results/hccb_p418_mixed_endpoint_smoke_model_comparison_${EPOCHS}epoch" \
  SPLITS="${ROOT}/results/hccb_p418_mixed_endpoint_smoke_splits.json" \
  TRAINING_STATISTICS="${ROOT}/results/hccb_p418_mixed_endpoint_smoke_training_statistics.json" \
  POSTPROCESS_SUMMARY="${ROOT}/results/hccb_p418_mixed_endpoint_smoke_postprocess_summary.json" \
  EXPECTED_CASES=8 \
  SPLIT_NAMES="completed_smoke" \
  EPOCHS="${EPOCHS}" \
  THREADS="${THREADS}" \
  DEVICE="${DEVICE}" \
  ARCHITECTURES="pinn_data_only pinn graph transolver" \
  bash "${ROOT}/code/run_hccb_p418_60_model_comparison.sh"
