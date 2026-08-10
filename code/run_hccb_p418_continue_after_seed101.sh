#!/usr/bin/env bash
# Wait for the 60-condition seed101 matrix, then continue the declared paper route.

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
EXECUTE=${EXECUTE:-0}
POLL_SECONDS=${POLL_SECONDS:-300}
MATRIX_ROOT=${MATRIX_ROOT:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3}
LOCK_FILE=${LOCK_FILE:-${ROOT}/results/hccb_p418_continuation_watch.lock}
STATUS_FILE=${STATUS_FILE:-${ROOT}/results/hccb_p418_continuation_watch_status.txt}
LOG_FILE=${LOG_FILE:-${ROOT}/results/hccb_p418_formal_calculations.log}
P418_PYTHON=${P418_PYTHON:-python3}

if [[ ${EXECUTE} != 0 && ${EXECUTE} != 1 ]]; then
    echo "EXECUTE must be 0 or 1" >&2
    exit 1
fi
if ! [[ ${POLL_SECONDS} =~ ^[1-9][0-9]*$ ]]; then
    echo "POLL_SECONDS must be a positive integer" >&2
    exit 1
fi

echo "P418 continuation after seed101"
echo "  matrix: ${MATRIX_ROOT}"
echo "  next: physical thermal steps, model comparison, seed202 and seed303"
if [[ ${EXECUTE} == 0 ]]; then
    echo "dry run only: no waiting or calculation was started"
    exit 0
fi

if ! resolved_python=$(command -v "${P418_PYTHON}"); then
    echo "P418 Python executable not found: ${P418_PYTHON}" >&2
    exit 1
fi
P418_PYTHON=${resolved_python}
export P418_PYTHON
export PATH="$(dirname "${P418_PYTHON}"):${PATH}"
"${P418_PYTHON}" -c 'import numpy, pandas, scipy, sklearn, torch'

mkdir -p "${ROOT}/results"
exec 8>"${LOCK_FILE}"
if ! flock -n 8; then
    echo "a P418 continuation process is already running" >&2
    exit 1
fi

while true; do
    completed=$(find "${MATRIX_ROOT}" -mindepth 2 -maxdepth 2 \
        -name formal_sample_complete.json | wc -l | tr -d ' ')
    printf '%s seed101 completed %s/60\n' "$(date --iso-8601=seconds)" "${completed}" \
        > "${STATUS_FILE}"
    if [[ ${completed} -eq 60 ]]; then
        break
    fi
    if [[ ${completed} -gt 60 ]]; then
        echo "seed101 contains more than 60 completion records" >&2
        exit 1
    fi
    if ! pgrep -f "run_hccb_dense_cht_p418_matrix_parallel.sh" >/dev/null; then
        echo "seed101 calculation stopped at ${completed}/60; continuation not started" >&2
        exit 1
    fi
    sleep "${POLL_SECONDS}"
done

# The completion files are written after each case is finalized.  Wait until the
# 60-case driver has also left its summary stage before starting the next route.
while pgrep -f "run_hccb_dense_cht_p418_matrix_parallel.sh" >/dev/null; do
    sleep "${POLL_SECONDS}"
done

printf '%s seed101 complete; starting the remaining paper calculations\n' \
    "$(date --iso-8601=seconds)" > "${STATUS_FILE}"
ROOT="${ROOT}" P418_PYTHON="${P418_PYTHON}" EXECUTE=1 \
    bash "${ROOT}/code/run_hccb_p418_formal_calculations.sh" \
    > "${LOG_FILE}" 2>&1

printf '%s all declared P418 calculations complete\n' "$(date --iso-8601=seconds)" \
    > "${STATUS_FILE}"
echo "P418 formal calculations complete"
