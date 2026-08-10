#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_ROOT=${DATA_ROOT:-${ROOT}/experimental_data_templates}
MODEL_OUTPUT=${MODEL_OUTPUT:?MODEL_OUTPUT is required}
MODEL_NAME=${MODEL_NAME:-learned P418 model}
SPLIT_NAME=${SPLIT_NAME:-interleaved_all_ranges}
RESULT_PREFIX=${RESULT_PREFIX:-${ROOT}/results/hccb_p418_60_sourceflow_r3}
REGIONAL_TOPOLOGY=${REGIONAL_TOPOLOGY:-${ROOT}/results/hccb_p418_regional_topology_r2/regional_topology.npz}
REFERENCE_STATE_TARGETS=${REFERENCE_STATE_TARGETS:-${RESULT_PREFIX}_regional_state_targets/regional_state_targets.npz}
MASS_TARGETS=${MASS_TARGETS:-${RESULT_PREFIX}_regional_mass_flux_targets/regional_mass_flux_targets.npz}
TRAINING_STATISTICS=${TRAINING_STATISTICS:-${RESULT_PREFIX}_training_statistics.json}
OUTPUT_DIR=${OUTPUT_DIR:-${MODEL_OUTPUT}/experimental_comparison}

mkdir -p "${OUTPUT_DIR}"
python3 "${ROOT}/code/validate_hccb_p418_experimental_data.py" \
  --schema "${ROOT}/parameters/hccb_p418_experimental_data_schema.json" \
  --data-root "${DATA_ROOT}" \
  --output "${OUTPUT_DIR}/experimental_data_validation.json"

measurement_count=$(python3 - "${OUTPUT_DIR}/experimental_data_validation.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["steady_measurement_count"] + payload["transient_measurement_count"])
PY
)
if [[ ${measurement_count} -eq 0 ]]; then
  python3 - "${OUTPUT_DIR}/summary.json" "${MODEL_NAME}" <<'PY'
import json
import pathlib
import sys
pathlib.Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "status": "no_experimental_measurements",
            "model_name": sys.argv[2],
            "measurement_count": 0,
            "interpretation_cn": "实验表为空，没有合并模型文件或生成比较数字。",
            "new_physical_parameters": [],
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
  exit 0
fi

MERGED=${OUTPUT_DIR}/all_condition_regional_predictions.npz
python3 "${ROOT}/code/merge_hccb_p418_prediction_splits.py" \
  --input "${MODEL_OUTPUT}/train_regional_predictions.npz" \
  --input "${MODEL_OUTPUT}/validation_regional_predictions.npz" \
  --input "${MODEL_OUTPUT}/test_regional_predictions.npz" \
  --output "${MERGED}"

python3 "${ROOT}/code/compare_hccb_p418_model_to_experiment.py" \
  --data-root "${DATA_ROOT}" \
  --regional-topology "${REGIONAL_TOPOLOGY}" \
  --state-file "${MERGED}" \
  --reference-state-targets "${REFERENCE_STATE_TARGETS}" \
  --training-statistics "${TRAINING_STATISTICS}" \
  --split-name "${SPLIT_NAME}" \
  --mass-targets "${MASS_TARGETS}" \
  --model-summary "${MODEL_OUTPUT}/summary.json" \
  --model-name "${MODEL_NAME}" \
  --output-dir "${OUTPUT_DIR}"
