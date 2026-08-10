#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
NP_PER_CASE=${NP_PER_CASE:-32}
CONCURRENT_CASES=${CONCURRENT_CASES:-3}
RESULT_ROOT=${RESULT_ROOT:-${ROOT}/results}
LOG=${LOG:-${RESULT_ROOT}/hccb_p418_power_recovery_$(date +%Y%m%d_%H%M%S).log}
LOCK=${LOCK:-${RESULT_ROOT}/hccb_p418_full_pipeline.lock}
PROGRESS_INTERVAL_SECONDS=${PROGRESS_INTERVAL_SECONDS:-600}
PROGRESS_JSON=${PROGRESS_JSON:-${RESULT_ROOT}/hccb_p418_runtime_progress.json}
PROGRESS_LOG=${PROGRESS_LOG:-${RESULT_ROOT}/hccb_p418_runtime_progress.log}
P418_PYTHON=${P418_PYTHON:-/data2/wangjian/venv/bin/python3}

mkdir -p "${RESULT_ROOT}"
exec 8>"${LOCK}"
if ! flock -n 8; then
  echo "P418 full pipeline is already running: ${LOCK}" >&2
  exit 1
fi

exec > >(tee -a "${LOG}") 2>&1
echo "[$(date --iso-8601=seconds)] resume P418 heat-transfer calculations"
echo "root=${ROOT}"
echo "steady parallelism=${CONCURRENT_CASES} cases x ${NP_PER_CASE} ranks"

cd "${ROOT}"

monitor_progress() {
  while true; do
    python3 "${ROOT}/code/report_hccb_p418_runtime_progress.py" \
      --matrix-root "${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3" \
      --concurrent-cases "${CONCURRENT_CASES}" \
      --parallel-ranks "${NP_PER_CASE}" \
      --output "${PROGRESS_JSON}" >> "${PROGRESS_LOG}" 2>&1 || true
    sleep "${PROGRESS_INTERVAL_SECONDS}"
  done
}

monitor_progress &
monitor_pid=$!
cleanup_monitor() {
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
}
trap cleanup_monitor EXIT INT TERM

NP_PER_CASE="${NP_PER_CASE}" CONCURRENT_CASES="${CONCURRENT_CASES}" \
  bash "${ROOT}/code/run_hccb_dense_cht_p418_matrix_parallel.sh"

completed=$(find "${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3" \
  -mindepth 2 -maxdepth 2 -name formal_sample_complete.json | wc -l | tr -d ' ')
if [[ ${completed} -ne 60 ]]; then
  echo "steady matrix incomplete: ${completed}/60" >&2
  exit 1
fi

echo "[$(date --iso-8601=seconds)] steady matrix complete; start the remaining formal calculations"
ROOT="${ROOT}" P418_PYTHON="${P418_PYTHON}" EXECUTE=1 \
NP_PER_CASE="${NP_PER_CASE}" CONCURRENT_CASES="${CONCURRENT_CASES}" \
  bash "${ROOT}/code/run_hccb_p418_formal_calculations.sh"
echo "[$(date --iso-8601=seconds)] P418 heat-transfer pipeline complete"
