#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
WAIT_PID=${WAIT_PID:-}
WAIT_VALIDATION_PID=${WAIT_VALIDATION_PID:-}
POLL_SECONDS=${POLL_SECONDS:-60}
RESULT_ROOT=${ROOT}/results/hccb_p418_physical_steps_12
SPLITS=${ROOT}/parameters/hccb_p418_step_response_splits.json
OUTPUT=${ROOT}/figures/hccb_p418_transient_model_comparison.json
MARKER=${ROOT}/manuscript/generated_transient_model_comparison_validated.tex
SUMMARY=${RESULT_ROOT}/model_comparison/summary.json
LOCK=${RESULT_ROOT}/.post_manifest_transient_figure.lock

if [[ -z ${WAIT_PID} || ! ${WAIT_PID} =~ ^[0-9]+$ ]]; then
    echo "WAIT_PID must identify the existing formal-model executor" >&2
    exit 2
fi
if [[ -z ${WAIT_VALIDATION_PID} || ! ${WAIT_VALIDATION_PID} =~ ^[0-9]+$ ]]; then
    echo "WAIT_VALIDATION_PID must identify the registered validation process" >&2
    exit 2
fi
if [[ ${POLL_SECONDS} -le 0 ]]; then
    echo "POLL_SECONDS must be positive" >&2
    exit 2
fi

if ! mkdir "${LOCK}" 2>/dev/null; then
    echo "transient-figure process is already registered: ${LOCK}" >&2
    exit 3
fi
trap 'rmdir "${LOCK}" 2>/dev/null || true' EXIT

while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
done
while kill -0 "${WAIT_VALIDATION_PID}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
done

python3 - "${SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"final model-comparison summary is missing: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "completed_p418_physical_step_model_comparison":
    raise SystemExit(
        "final model-comparison summary is incomplete: "
        + str(payload.get("status"))
    )
if payload.get("strict_split_loss_balancing_stage") != "validation_selected":
    raise SystemExit("final model comparison does not use validation-selected weights")
PY

if [[ -s ${OUTPUT} && -s ${MARKER} ]]; then
    if python3 - "${OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "complete_formal_p418_transient_model_comparison_figure":
    raise SystemExit(1)
if payload.get("split_name") != "pair_disjoint_stress_test":
    raise SystemExit(1)
if payload.get("new_physical_parameter_values_added") != []:
    raise SystemExit(1)
if payload.get("strict_split_loss_balancing_stage") != "validation_selected":
    raise SystemExit(1)
PY
    then
        exit 0
    fi
fi

python3 "${ROOT}/code/plot_hccb_p418_transient_model_comparison.py" \
    --result-dir "${RESULT_ROOT}" \
    --splits "${SPLITS}" \
    --output-dir "${ROOT}/figures"

python3 - "${OUTPUT}" "${MARKER}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
marker = Path(sys.argv[2])
payload = json.loads(output.read_text(encoding="utf-8"))
if payload.get("status") != "complete_formal_p418_transient_model_comparison_figure":
    raise SystemExit("transient figure has an unexpected completion status")
if payload.get("split_name") != "pair_disjoint_stress_test":
    raise SystemExit("transient figure was not evaluated on the fixed strict split")
if payload.get("new_physical_parameter_values_added") != []:
    raise SystemExit("transient figure introduced a physical parameter")
if payload.get("strict_split_loss_balancing_stage") != "validation_selected":
    raise SystemExit("transient figure does not use validation-selected weights")
if not marker.is_file() or marker.stat().st_size == 0:
    raise SystemExit("transient figure validation marker was not generated")
PY
