#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
WAIT_PID=${WAIT_PID:-}
WAIT_VALIDATION_PID=${WAIT_VALIDATION_PID:-}
POLL_SECONDS=${POLL_SECONDS:-60}
RESULT_ROOT=${ROOT}/results/hccb_p418_physical_steps_12
SELECTION=${ROOT}/figures/hccb_p418_openfoam_model_field_selection.json
OUTPUT=${ROOT}/figures/hccb_p418_openfoam_model_field_comparison.json
LOCK=${RESULT_ROOT}/.post_manifest_field_figure.lock

if [[ -z ${WAIT_PID} || ! ${WAIT_PID} =~ ^[0-9]+$ ]]; then
    echo "WAIT_PID must identify the existing formal-model executor" >&2
    exit 2
fi
if [[ -z ${WAIT_VALIDATION_PID} || ! ${WAIT_VALIDATION_PID} =~ ^[0-9]+$ ]]; then
    echo "WAIT_VALIDATION_PID must identify the registered validation process" >&2
    exit 2
fi

if ! mkdir "${LOCK}" 2>/dev/null; then
    echo "field-figure process is already registered: ${LOCK}" >&2
    exit 3
fi
trap 'rmdir "${LOCK}" 2>/dev/null || true' EXIT

while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
done
while kill -0 "${WAIT_VALIDATION_PID}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
done

if [[ -s ${OUTPUT} && -s ${SELECTION} ]]; then
    if python3 - "${OUTPUT}" "${SELECTION}" <<'PY'
import json
import sys
from pathlib import Path

figure = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
selection = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if figure.get("status") != "complete_same_scale_openfoam_model_field_comparison":
    raise SystemExit(1)
if figure.get("selected_model") != selection.get("selected_model"):
    raise SystemExit(1)
if figure.get("prediction_file_sha256") != selection.get("prediction_file_sha256"):
    raise SystemExit(1)
if figure.get("strict_split_loss_balancing_stage") != "validation_selected":
    raise SystemExit(1)
if selection.get("strict_split_loss_balancing_stage") != "validation_selected":
    raise SystemExit(1)
if figure.get("selection_data_role") != "validation":
    raise SystemExit(1)
if figure.get("display_data_role") != "test":
    raise SystemExit(1)
if selection.get("selection_data_role") != "validation":
    raise SystemExit(1)
if selection.get("display_data_role") != "test":
    raise SystemExit(1)
PY
    then
        exit 0
    fi
fi

bash "${ROOT}/code/build_hccb_p418_selected_field_figure.sh"

python3 - "${OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "complete_same_scale_openfoam_model_field_comparison":
    raise SystemExit("field-figure generation did not write the expected completion record")
if payload.get("new_physical_parameters") != []:
    raise SystemExit("field-figure generation introduced a physical parameter")
if payload.get("strict_split_loss_balancing_stage") != "validation_selected":
    raise SystemExit("field figure does not use validation-selected weights")
if payload.get("selection_data_role") != "validation":
    raise SystemExit("field-figure model was not selected on validation data")
if payload.get("display_data_role") != "test":
    raise SystemExit("field figure does not display a held-out test trajectory")
PY
