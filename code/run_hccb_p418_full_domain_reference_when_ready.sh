#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
LOCAL_STEADY_ROOT=${LOCAL_STEADY_ROOT:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3}
STEP_ROOT=${STEP_ROOT:-${ROOT}/hccb_p418_physical_steps_12}
RESULT_DIR=${RESULT_DIR:-${ROOT}/results/hccb_p418_full_domain_reference}
POLL_SECONDS=${POLL_SECONDS:-600}
LOCK_FILE=${LOCK_FILE:-${ROOT}/results/hccb_p418_full_domain_reference_when_ready.lock}
WATCH_LOG=${WATCH_LOG:-${ROOT}/results/hccb_p418_full_domain_reference_when_ready.log}

if ! [[ ${POLL_SECONDS} =~ ^[1-9][0-9]*$ ]]; then
  echo "POLL_SECONDS must be a positive integer" >&2
  exit 2
fi

mkdir -p "${ROOT}/results" "${RESULT_DIR}" "${LOCAL_STEADY_ROOT}" "${STEP_ROOT}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "a full-domain reference watcher is already active"
  exit 0
fi

write_state() {
  local state=$1
  local steady_count=$2
  local step_count=$3
  local message=$4
  python3 - "${RESULT_DIR}/watcher_state.json" "${state}" \
    "${steady_count}" "${step_count}" "${message}" <<'PY'
import datetime as dt
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "steady_cases_completed": int(sys.argv[3]),
    "thermal_steps_completed": int(sys.argv[4]),
    "message": sys.argv[5],
    "updated_at": dt.datetime.now().astimezone().isoformat(),
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

while true; do
  steady_count=$(find "${LOCAL_STEADY_ROOT}" -mindepth 2 -maxdepth 2 \
    -name formal_sample_complete.json 2>/dev/null | wc -l | tr -d ' ')
  step_count=$(find "${STEP_ROOT}" -mindepth 2 -maxdepth 2 \
    -name step_response_complete.json 2>/dev/null | wc -l | tr -d ' ')

  if [[ ${steady_count} -eq 60 && ${step_count} -eq 12 ]]; then
    message="60 steady cases and 12 thermal steps are ready; starting full-domain G2 reference"
    printf '%s %s\n' "$(date --iso-8601=seconds)" "${message}" | tee -a "${WATCH_LOG}"
    write_state running_full_domain_reference "${steady_count}" "${step_count}" "${message}"
    break
  fi

  message="waiting for physical calculations: steady ${steady_count}/60, thermal steps ${step_count}/12"
  printf '%s %s\n' "$(date --iso-8601=seconds)" "${message}" >> "${WATCH_LOG}"
  write_state waiting_for_physical_calculations "${steady_count}" "${step_count}" "${message}"
  sleep "${POLL_SECONDS}"
done

set +e
ROOT="${ROOT}" LOCAL_STEADY_ROOT="${LOCAL_STEADY_ROOT}" STEP_ROOT="${STEP_ROOT}" \
RESULT_DIR="${RESULT_DIR}" bash "${ROOT}/code/run_hccb_p418_full_domain_reference.sh" \
  >> "${WATCH_LOG}" 2>&1
run_status=$?
set -e

if [[ ${run_status} -eq 0 ]]; then
  write_state full_domain_reference_completed 60 12 \
    "full-domain G2 reference and local-domain comparison completed"
  printf '%s full-domain G2 reference completed\n' "$(date --iso-8601=seconds)" \
    >> "${WATCH_LOG}"
  exit 0
fi

write_state full_domain_reference_needs_attention 60 12 \
  "full-domain reference stopped with exit code ${run_status}; inspect the mesh or solver log"
printf '%s full-domain reference stopped with exit code %s\n' \
  "$(date --iso-8601=seconds)" "${run_status}" >> "${WATCH_LOG}"
exit "${run_status}"
