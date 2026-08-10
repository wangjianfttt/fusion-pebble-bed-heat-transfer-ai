#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RESULT_PREFIX=${RESULT_PREFIX:-${ROOT}/results/hccb_p418_60_sourceflow_r3}
REGIONAL_TOPOLOGY=${REGIONAL_TOPOLOGY:-${ROOT}/results/hccb_p418_regional_topology_r2/regional_topology.npz}
MODEL_GEOMETRY=${MODEL_GEOMETRY:-${RESULT_PREFIX}_model_geometry/model_geometry.npz}
ARCHITECTURE_SOURCES=${ARCHITECTURE_SOURCES:-${ROOT}/parameters/hccb_p418_ai_architecture_sources.json}
SPLIT_NAME=${SPLIT_NAME:-interleaved_all_ranges}
BASE_EPOCHS=${BASE_EPOCHS:-100}
THREADS=${THREADS:-16}
DEVICE=${DEVICE:-cpu}
SELECTION=${ROOT}/results/hccb_p418_60_diffusion_${SPLIT_NAME}_base_selection.json
OUTPUT=${ROOT}/results/hccb_p418_60_diffusion_${SPLIT_NAME}_500epoch

python3 - "${ROOT}" "${SPLIT_NAME}" "${BASE_EPOCHS}" "${SELECTION}" <<'PY'
import json
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
split = sys.argv[2]
epochs = int(sys.argv[3])
output = pathlib.Path(sys.argv[4])
candidates = []
for architecture in ("response_surface", "pinn", "graph", "transolver"):
    directory = root / "results" / f"hccb_p418_60_{architecture}_{split}_{epochs}epoch"
    summary = directory / "summary.json"
    predictions = directory / "validation_regional_predictions.npz"
    if not summary.is_file() or not predictions.is_file():
        continue
    payload = json.loads(summary.read_text(encoding="utf-8"))
    value = payload.get("evaluations", {}).get("validation", {}).get("metrics", {}).get(
        "state_normalized_rmse"
    )
    if value is None or not math.isfinite(float(value)):
        continue
    candidates.append(
        {
            "architecture": architecture,
            "validation_state_normalized_rmse": float(value),
            "prediction_directory": str(directory),
            "summary": str(summary),
        }
    )
if not candidates:
    raise SystemExit("no completed deterministic validation prediction is available")
selected = min(candidates, key=lambda row: row["validation_state_normalized_rmse"])
output.write_text(
    json.dumps({"selection_basis": "validation field RMSE", "selected": selected, "candidates": candidates}, indent=2) + "\n",
    encoding="utf-8",
)
print(selected["prediction_directory"])
PY

PREDICTION_DIR=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["prediction_directory"])' "${SELECTION}")
mkdir -p "${OUTPUT}"
python3 "${ROOT}/code/train_hccb_p418_regional_diffusion_refiner.py" \
    --prediction-dir "${PREDICTION_DIR}" \
    --regional-topology "${REGIONAL_TOPOLOGY}" \
    --model-geometry "${MODEL_GEOMETRY}" \
    --regional-level 5 \
    --architecture-sources "${ARCHITECTURE_SOURCES}" \
    --output-dir "${OUTPUT}" \
    --epochs 500 \
    --batch-size 8 \
    --hidden-dim 256 \
    --layers 5 \
    --attention-heads 8 \
    --physics-slices 32 \
    --learning-rate 0.001 \
    --weight-decay 0.00001 \
    --num-refinement-steps 3 \
    --min-noise-std 4e-7 \
    --ema-decay 0.995 \
    --threads "${THREADS}" \
    --device "${DEVICE}" \
    > "${OUTPUT}/run.log" 2>&1
