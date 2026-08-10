#!/usr/bin/env bash
# Update the completed-case physical summary without touching active CFD fields.

set -uo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MATRIX_ROOT=${MATRIX_ROOT:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/results/hccb_p418_sourceflow_partial_physics}
TAIL_OUTPUT_DIR=${TAIL_OUTPUT_DIR:-${ROOT}/results/hccb_p418_sourceflow_partial_final_windows}
SUMMARY_LOCK=${SUMMARY_LOCK:-${OUTPUT_DIR}.lock}
WATCH_LOCK=${WATCH_LOCK:-${OUTPUT_DIR}.watch.lock}
STATUS_FILE=${STATUS_FILE:-${OUTPUT_DIR}/watch_status.txt}
POLL_SECONDS=${POLL_SECONDS:-120}
EXPECTED_CASES=${EXPECTED_CASES:-60}
P418_PYTHON=${P418_PYTHON:-python3}

if ! [[ ${POLL_SECONDS} =~ ^[1-9][0-9]*$ ]]; then
  echo "POLL_SECONDS must be a positive integer" >&2
  exit 1
fi
if ! [[ ${EXPECTED_CASES} =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_CASES must be a positive integer" >&2
  exit 1
fi
if ! P418_PYTHON=$(command -v "${P418_PYTHON}"); then
  echo "P418 Python executable not found" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
exec 9>"${WATCH_LOCK}"
if ! flock -n 9; then
  echo "a partial-physics watcher is already running" >&2
  exit 1
fi

last_summarized=-1
missing_runner_polls=0
while true; do
  completed=$(find "${MATRIX_ROOT}" -mindepth 2 -maxdepth 2 \
    -name formal_sample_complete.json | wc -l | tr -d ' ')
  if [[ ${completed} -gt ${EXPECTED_CASES} ]]; then
    echo "completion count ${completed} exceeds ${EXPECTED_CASES}" >&2
    exit 1
  fi

  if [[ ${completed} -gt 0 && ${completed} -ne ${last_summarized} ]]; then
    stdout_tmp="${OUTPUT_DIR}/summary_stdout.tmp.$$"
    if (
      flock -x 8
      "${P418_PYTHON}" \
        "${ROOT}/code/summarize_hccb_p418_completed_matrix_physics.py" \
        --matrix-root "${MATRIX_ROOT}" \
        --time-from-completion-marker \
        --output-dir "${OUTPUT_DIR}" \
        > "${stdout_tmp}" \
        && full_field_required=$((completed > 4 ? completed - 4 : 0)) \
        && mkdir -p "${TAIL_OUTPUT_DIR}" \
        && "${P418_PYTHON}" \
          "${ROOT}/code/summarize_hccb_p418_formal_steady_tails.py" \
          --matrix-root "${MATRIX_ROOT}" \
          --expected-case-count "${EXPECTED_CASES}" \
          --minimum-full-field-count "${full_field_required}" \
          --allow-partial \
          --output-dir "${TAIL_OUTPUT_DIR}" \
        > "${TAIL_OUTPUT_DIR}.stdout.tmp.$$" \
        && mv -f "${stdout_tmp}" "${OUTPUT_DIR}/summary_stdout.json" \
        && mv -f "${TAIL_OUTPUT_DIR}.stdout.tmp.$$" "${TAIL_OUTPUT_DIR}/summary_stdout.json"
    ) 8>"${SUMMARY_LOCK}"; then
      last_summarized=${completed}
      printf '%s summarized %s/%s completed cases\n' \
        "$(date --iso-8601=seconds)" "${completed}" "${EXPECTED_CASES}" \
        > "${STATUS_FILE}"
    else
      rm -f "${stdout_tmp}"
      rm -f "${TAIL_OUTPUT_DIR}.stdout.tmp.$$"
      printf '%s summary update failed at %s/%s; retrying\n' \
        "$(date --iso-8601=seconds)" "${completed}" "${EXPECTED_CASES}" \
        > "${STATUS_FILE}"
    fi
  fi

  if [[ ${completed} -eq ${EXPECTED_CASES} ]]; then
    printf '%s final partial summary complete at %s/%s\n' \
      "$(date --iso-8601=seconds)" "${completed}" "${EXPECTED_CASES}" \
      > "${STATUS_FILE}"
    exit 0
  fi

  if pgrep -f "run_hccb_dense_cht_p418_matrix_parallel.sh" >/dev/null; then
    missing_runner_polls=0
  else
    missing_runner_polls=$((missing_runner_polls + 1))
    if [[ ${missing_runner_polls} -ge 12 ]]; then
      printf '%s matrix process absent at %s/%s; watcher stopped\n' \
        "$(date --iso-8601=seconds)" "${completed}" "${EXPECTED_CASES}" \
        > "${STATUS_FILE}"
      exit 1
    fi
  fi
  sleep "${POLL_SECONDS}"
done
