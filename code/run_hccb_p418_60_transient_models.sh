#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MATRIX_ROOT=${MATRIX_ROOT:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3}
THREADS=${THREADS:-16}
OBSERVABLE_DIR=${OBSERVABLE_DIR:-${ROOT}/results/hccb_p418_transient_observables_60}
TRANSFORMER_DIR=${TRANSFORMER_DIR:-${ROOT}/results/hccb_p418_transient_transformer_formal}

completed=$(find "${MATRIX_ROOT}" -mindepth 2 -maxdepth 2 -name formal_sample_complete.json | wc -l | tr -d ' ')
if [[ ${completed} -ne 60 ]]; then
  echo "P418 transient training requires 60 completed OpenFOAM cases; found ${completed}" >&2
  exit 1
fi

python3 "${ROOT}/code/export_hccb_p418_transient_observables.py" \
  --matrix-root "${MATRIX_ROOT}" \
  --output-dir "${OBSERVABLE_DIR}"

OPENBLAS_NUM_THREADS=${THREADS} OMP_NUM_THREADS=${THREADS} \
python3 "${ROOT}/code/train_hccb_p418_transient_observable_transformer.py" \
  --data "${OBSERVABLE_DIR}/hccb_p418_transient_observables.npz" \
  --splits "${ROOT}/parameters/hccb_p418_model_splits.json" \
  --split-name interleaved_all_ranges \
  --output-dir "${TRANSFORMER_DIR}" \
  --run-role formal
