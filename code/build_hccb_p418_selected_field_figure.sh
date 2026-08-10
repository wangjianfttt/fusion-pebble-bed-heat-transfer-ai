#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RESULT_ROOT=${RESULT_ROOT:-${ROOT}/results/hccb_p418_physical_steps_12}
SELECTION=${SELECTION:-${ROOT}/figures/hccb_p418_openfoam_model_field_selection.json}
GEOMETRY=${GEOMETRY:-${RESULT_ROOT}/regional_sequences/regional_sequence_geometry.npz}
SPLITS=${SPLITS:-${ROOT}/parameters/hccb_p418_step_response_splits.json}
SPLIT_NAME=pair_disjoint_stress_test
SEQUENCE_ID=source_up_u0p15_T700

python3 - "${SPLITS}" "${SPLIT_NAME}" "${SEQUENCE_ID}" <<'PY'
import json
import sys
from pathlib import Path

path, split_name, sequence_id = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
split = json.loads(path.read_text(encoding="utf-8"))["splits"][split_name]
if sequence_id not in split["test"]:
    raise SystemExit(
        f"field-figure sequence {sequence_id!r} is not in the registered test set"
    )
if any(sequence_id in split[role] for role in ("train", "validation")):
    raise SystemExit(f"field-figure sequence {sequence_id!r} leaks into model selection")
PY

python3 "${ROOT}/code/select_hccb_p418_field_figure_model.py" \
    --result-dir "${RESULT_ROOT}" \
    --comparison-summary "${RESULT_ROOT}/model_comparison/summary.json" \
    --metrics-csv "${RESULT_ROOT}/model_comparison/physical_step_model_metrics.csv" \
    --split-name "${SPLIT_NAME}" \
    --output "${SELECTION}"

python3 "${ROOT}/code/plot_hccb_p418_field_cloud_comparison.py" \
    --selection "${SELECTION}" \
    --geometry "${GEOMETRY}" \
    --sequence-id "${SEQUENCE_ID}" \
    --time-s 25 \
    --output-dir "${ROOT}/figures"
