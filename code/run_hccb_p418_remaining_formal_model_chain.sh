#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RESULT_ROOT=${RESULT_ROOT:-${ROOT}/results/hccb_p418_physical_steps_12}
MANIFEST=${MANIFEST:-${RESULT_ROOT}/formal_training_jobs_workstation.json}
STATE_FILE=${STATE_FILE:-${RESULT_ROOT}/remaining_formal_model_chain_state.json}
LOG_DIR=${LOG_DIR:-${RESULT_ROOT}/remaining_formal_model_chain_logs}
LOCK_FILE=${LOCK_FILE:-${RESULT_ROOT}/.remaining_formal_model_chain.lock}
WAIT_INTERVAL_S=${WAIT_INTERVAL_S:-60}
CUDA_VISIBLE_DEVICES_FOR_CHAIN=${CUDA_VISIBLE_DEVICES_FOR_CHAIN:-0}
CPU_LIST=${CPU_LIST:-110,111}
WAIT_PIDS=${WAIT_PIDS:-}
READY_FILES=${READY_FILES:-}
EXECUTE=${EXECUTE:-0}

python3 "${ROOT}/code/build_hccb_p418_formal_training_job_manifest.py" \
  --root "${ROOT}" \
  --result-dir "${RESULT_ROOT}" \
  --splits "${ROOT}/parameters/hccb_p418_step_response_splits.json" \
  --dataset-index "${RESULT_ROOT}/regional_sequences/dataset_index.json" \
  --observables "${RESULT_ROOT}/hccb_p418_transient_observables.npz" \
  --residual-geometry \
    "${ROOT}/results/hccb_p418_subface_residual_geometry_r2/subface_residual_geometry.npz" \
  --output "${MANIFEST}"

arguments=(
  --manifest "${MANIFEST}"
  --root "${ROOT}"
  --state-file "${STATE_FILE}"
  --log-dir "${LOG_DIR}"
  --lock-file "${LOCK_FILE}"
  --wait-interval-s "${WAIT_INTERVAL_S}"
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES_FOR_CHAIN}"
  --cpu-list "${CPU_LIST}"
)

for process_id in ${WAIT_PIDS//,/ }; do
  [[ -z "${process_id}" ]] || arguments+=(--wait-pid "${process_id}")
done
for path in ${READY_FILES//,/ }; do
  [[ -z "${path}" ]] || arguments+=(--required-ready-file "${path}")
done
if [[ "${EXECUTE}" == "1" ]]; then
  arguments+=(--execute)
fi

python3 "${ROOT}/code/execute_hccb_p418_formal_training_manifest.py" \
  "${arguments[@]}"
