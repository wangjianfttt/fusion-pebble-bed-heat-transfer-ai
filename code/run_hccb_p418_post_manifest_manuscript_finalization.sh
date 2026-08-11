#!/usr/bin/env bash
# Wait for the formal model chain and both final figures, then build the paper.

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
WAIT_PIDS=${WAIT_PIDS:-}
POLL_SECONDS=${POLL_SECONDS:-60}
BUILD_SUPPLEMENT=${BUILD_SUPPLEMENT:-0}
GPU_ID=${GPU_ID:-0}
CPU_LIST=${CPU_LIST:-110,111}
THREADS=${THREADS:-2}
RESULT_ROOT=${ROOT}/results
TRANSIENT_ROOT=${RESULT_ROOT}/hccb_p418_physical_steps_12
SUMMARY=${TRANSIENT_ROOT}/model_comparison/summary.json
TRANSIENT_MARKER=${ROOT}/manuscript/generated_transient_model_comparison_validated.tex
FIELD_MARKER=${ROOT}/manuscript/generated_openfoam_model_field_comparison_validated.tex
TRANSIENT_FIGURE=${ROOT}/figures/hccb_p418_transient_model_comparison.json
FIELD_FIGURE=${ROOT}/figures/hccb_p418_openfoam_model_field_comparison.json
FINAL_RECORD=${RESULT_ROOT}/hccb_p418_manuscript_refresh_complete.json
GRAPHICAL_ABSTRACT_STEM=${ROOT}/figures/hccb_p418_graphical_abstract
GRAPHICAL_ABSTRACT_RECORD=${GRAPHICAL_ABSTRACT_STEM}.json
GRAPHICAL_ABSTRACT_LOCK=${RESULT_ROOT}/.p418_post_manifest_graphical_abstract.lock
REMAINING_VALIDATION_RECORD=${RESULT_ROOT}/hccb_p418_remaining_validation_chain_complete.json
REMAINING_VALIDATION_LOCK=${RESULT_ROOT}/.p418_remaining_validation_chain.lock
LOCK=${TRANSIENT_ROOT}/.post_manifest_manuscript_finalization.lock

if [[ -z ${WAIT_PIDS} ]]; then
    echo "WAIT_PIDS must list the formal executor and final-figure processes" >&2
    exit 2
fi
if [[ ${POLL_SECONDS} -le 0 ]]; then
    echo "POLL_SECONDS must be positive" >&2
    exit 2
fi

IFS=',' read -r -a wait_pid_array <<< "${WAIT_PIDS}"
for pid in "${wait_pid_array[@]}"; do
    if [[ ! ${pid} =~ ^[0-9]+$ ]]; then
        echo "invalid PID in WAIT_PIDS: ${pid}" >&2
        exit 2
    fi
done

if ! mkdir "${LOCK}" 2>/dev/null; then
    echo "manuscript finalization is already registered: ${LOCK}" >&2
    exit 3
fi
trap 'rmdir "${LOCK}" 2>/dev/null || true' EXIT

for pid in "${wait_pid_array[@]}"; do
    while kill -0 "${pid}" 2>/dev/null; do
        sleep "${POLL_SECONDS}"
    done
done

remaining_validation_complete() {
python3 - "${REMAINING_VALIDATION_RECORD}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(
    0 if payload.get("status") == "completed_p418_remaining_validation_chain" else 1
)
PY
}

if ! remaining_validation_complete; then
    # Another registered validation chain may already be waiting for the same
    # formal-model executor. Wait for its file lock instead of launching a
    # duplicate chain and failing the manuscript finalization.
    exec 8>"${REMAINING_VALIDATION_LOCK}"
    while ! flock -n 8; do
        sleep "${POLL_SECONDS}"
    done
    flock -u 8
fi

if ! remaining_validation_complete; then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" taskset -c "${CPU_LIST}" \
        env GPU_ID="${GPU_ID}" CPU_LIST="${CPU_LIST}" THREADS="${THREADS}" \
        bash "${ROOT}/code/run_hccb_p418_remaining_validation_chain.sh"
fi

python3 - "${SUMMARY}" "${TRANSIENT_MARKER}" "${FIELD_MARKER}" \
    "${TRANSIENT_FIGURE}" "${FIELD_FIGURE}" <<'PY'
import json
import sys
from pathlib import Path

summary, transient_marker, field_marker, transient_figure, field_figure = (
    Path(value) for value in sys.argv[1:]
)
for path in (summary, transient_marker, field_marker, transient_figure, field_figure):
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"final manuscript input is missing: {path}")

checks = (
    (
        summary,
        "completed_p418_physical_step_model_comparison",
    ),
    (
        transient_figure,
        "complete_formal_p418_transient_model_comparison_figure",
    ),
    (
        field_figure,
        "complete_same_scale_openfoam_model_field_comparison",
    ),
)
for path, expected in checks:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != expected:
        raise SystemExit(
            f"unexpected completion status in {path}: {payload.get('status')}"
        )
    if payload.get("strict_split_loss_balancing_stage") != "validation_selected":
        raise SystemExit(f"final input does not use validation-selected weights: {path}")

field_payload = json.loads(field_figure.read_text(encoding="utf-8"))
if field_payload.get("selection_data_role") != "validation":
    raise SystemExit("final field figure was not selected on validation data")
if field_payload.get("display_data_role") != "test":
    raise SystemExit("final field figure does not display the independent test data")
PY

graphical_abstract_ready() {
python3 - "${GRAPHICAL_ABSTRACT_RECORD}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(1)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, UnicodeDecodeError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(
    0 if payload.get("status") == "p418_ijhmt_graphical_abstract_ready" else 1
)
PY
}

# The separately registered graphical-abstract waiter may finish at almost the
# same time as this finalizer.  Share its directory lock and reuse a completed
# result instead of allowing both processes to write the same files.
(
    while ! mkdir "${GRAPHICAL_ABSTRACT_LOCK}" 2>/dev/null; do
        sleep "${POLL_SECONDS}"
    done
    trap 'rmdir "${GRAPHICAL_ABSTRACT_LOCK}" 2>/dev/null || true' EXIT
    if ! graphical_abstract_ready; then
        python3 "${ROOT}/code/plot_hccb_p418_graphical_abstract.py" \
            --project-root "${ROOT}" \
            --output-stem "${GRAPHICAL_ABSTRACT_STEM}"
    fi
)

python3 - "${GRAPHICAL_ABSTRACT_RECORD}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "p418_ijhmt_graphical_abstract_ready":
    raise SystemExit("graphical abstract is not ready")
if payload.get("generative_ai_used_for_image") is not False:
    raise SystemExit("graphical abstract must be generated from project data only")
PY

ROOT="${ROOT}" RESULT_ROOT="${RESULT_ROOT}" \
    BUILD_SUPPLEMENT="${BUILD_SUPPLEMENT}" \
    OUTPUT_RECORD="${FINAL_RECORD}" \
    bash "${ROOT}/code/run_hccb_p418_manuscript_refresh.sh"

python3 - "${FINAL_RECORD}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(f"final manuscript record is missing: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "completed_p418_formal_manuscript_refresh":
    raise SystemExit(f"unexpected final manuscript status: {payload.get('status')}")
PY

echo "P418 final manuscript, Chinese reader and submission checks are complete"
