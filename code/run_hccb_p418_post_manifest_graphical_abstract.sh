#!/usr/bin/env bash
# Wait for the validated field figure, then create the optional IJHMT graphical abstract.

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
WAIT_PID=${WAIT_PID:-}
POLL_SECONDS=${POLL_SECONDS:-60}
FIELD_MARKER=${ROOT}/manuscript/generated_openfoam_model_field_comparison_validated.tex
FIELD_RECORD=${ROOT}/figures/hccb_p418_openfoam_model_field_comparison.json
OUTPUT_STEM=${ROOT}/figures/hccb_p418_graphical_abstract
LOCK=${ROOT}/results/.p418_post_manifest_graphical_abstract.lock

if [[ ! ${WAIT_PID} =~ ^[0-9]+$ ]]; then
    echo "WAIT_PID must be the registered final-field-figure process" >&2
    exit 2
fi
if [[ ${POLL_SECONDS} -le 0 ]]; then
    echo "POLL_SECONDS must be positive" >&2
    exit 2
fi
if ! mkdir "${LOCK}" 2>/dev/null; then
    echo "graphical abstract process is already registered: ${LOCK}" >&2
    exit 3
fi
trap 'rmdir "${LOCK}" 2>/dev/null || true' EXIT

while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
done

python3 - "${FIELD_MARKER}" "${FIELD_RECORD}" <<'PY'
import json
import sys
from pathlib import Path

marker, record = (Path(value) for value in sys.argv[1:])
for path in (marker, record):
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"validated field input is missing: {path}")
payload = json.loads(record.read_text(encoding="utf-8"))
if payload.get("status") != "complete_same_scale_openfoam_model_field_comparison":
    raise SystemExit(f"unexpected field-figure status: {payload.get('status')}")
if payload.get("selection_data_role") != "validation":
    raise SystemExit("field model was not selected on validation data")
if payload.get("display_data_role") != "test":
    raise SystemExit("field figure does not display independent test data")
PY

python3 "${ROOT}/code/plot_hccb_p418_graphical_abstract.py" \
    --project-root "${ROOT}" \
    --output-stem "${OUTPUT_STEM}"

echo "P418 graphical abstract generated from validated project figures"
