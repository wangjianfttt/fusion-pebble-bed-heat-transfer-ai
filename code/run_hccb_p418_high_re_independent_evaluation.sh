#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${ROOT:-${SCRIPT_ROOT}}
EXECUTE=${EXECUTE:-0}
DEVICE=${DEVICE:-cuda}
RESULT_ROOT=${RESULT_ROOT:-${ROOT}/results/hccb_p418_physical_steps_12}
TRAINING_DATASET=${TRAINING_DATASET:-${RESULT_ROOT}/regional_sequences/dataset_index.json}
TEST_DATASET=${TEST_DATASET:-${ROOT}/results/hccb_p418_high_re_independent_fixed_steps_6/regional_sequences/merged/dataset_index.json}
RESIDUAL_GEOMETRY=${RESIDUAL_GEOMETRY:-${ROOT}/results/hccb_p418_subface_residual_geometry_r2/subface_residual_geometry.npz}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/results/hccb_p418_high_re_three_bounded_model_evaluation}
PAUSE_MARKER=${PAUSE_MARKER:-${ROOT}/control/PAUSE_NEW_P418_CASES_FOR_CLOUD_MIGRATION}
ALLOW_PAUSED_WORKSTATION_RUN=${ALLOW_PAUSED_WORKSTATION_RUN:-0}

LABELS=(data_only physics_constrained factorized)
SUMMARY_PATHS=(
  "${RESULT_ROOT}/regional_graph_transformer_bounded_data_only_pair_disjoint_stress_test/summary.json"
  "${RESULT_ROOT}/regional_graph_transformer_bounded_physics_pair_disjoint_stress_test/summary.json"
  "${RESULT_ROOT}/regional_graph_transformer_bounded_factorized_pair_disjoint_stress_test/summary.json"
)
OUTPUT_DIRS=(
  "${OUTPUT_ROOT}/data_only"
  "${OUTPUT_ROOT}/physics_constrained"
  "${OUTPUT_ROOT}/factorized"
)

if [[ ${EXECUTE} != 1 ]]; then
  cat <<EOF
P418高速端三模型独立测试只打印计划，没有启动训练或推理。
training_dataset=${TRAINING_DATASET}
test_dataset=${TEST_DATASET}
residual_geometry=${RESIDUAL_GEOMETRY}
models=${LABELS[*]}
output_root=${OUTPUT_ROOT}
三种模型均已在主训练集上冻结后，才在同一6条高流速曲线上测试。
这6条曲线不参与训练、归一化、模型选择或超参数调整。
全耦合启动短算不参加模型精度排名。
EOF
  exit 0
fi

if [[ -f ${PAUSE_MARKER} && ${ALLOW_PAUSED_WORKSTATION_RUN} != 1 ]]; then
  echo "P418 model inference is paused for cloud migration: ${PAUSE_MARKER}" >&2
  exit 3
fi

for required in "${TRAINING_DATASET}" "${TEST_DATASET}" "${RESIDUAL_GEOMETRY}"; do
  if [[ ! -f ${required} ]]; then
    echo "required frozen-model input is missing: ${required}" >&2
    exit 4
  fi
done

for index in "${!LABELS[@]}"; do
  label=${LABELS[$index]}
  training_summary=${SUMMARY_PATHS[$index]}
  output_dir=${OUTPUT_DIRS[$index]}
  if [[ ! -f ${training_summary} ]]; then
    echo "completed ${label} training summary is missing: ${training_summary}" >&2
    exit 4
  fi
  mkdir -p "${output_dir}"
  python3 "${ROOT}/code/evaluate_hccb_p418_frozen_independent_operator.py" \
    --mode fixed \
    --training-summary "${training_summary}" \
    --training-dataset-index "${TRAINING_DATASET}" \
    --test-dataset-index "${TEST_DATASET}" \
    --output-dir "${output_dir}" \
    --device "${DEVICE}"
  if python3 - "${output_dir}/summary.json" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
metrics = summary["aggregate_metrics"]
fractions = [
    float(metrics["predicted_fluid_temperature_outside_registered_range_fraction"]),
    float(metrics["predicted_solid_temperature_outside_registered_range_fraction"]),
]
raise SystemExit(10 if any(value > 0.0 for value in fractions) else 0)
PY
  then
    python3 "${ROOT}/code/evaluate_hccb_p418_temporal_energy_balance.py" \
      --model-summary "${output_dir}/summary.json" \
      --dataset-index "${TEST_DATASET}" \
      --residual-geometry "${RESIDUAL_GEOMETRY}" \
      --output "${output_dir}/energy_balance_summary.json" \
      --roles test \
      --device cpu
  else
    status=$?
    if [[ ${status} != 10 ]]; then
      exit "${status}"
    fi
    python3 - "${output_dir}/summary.json" "${output_dir}/energy_balance_summary.json" <<'PY'
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1]).resolve()
output_path = pathlib.Path(sys.argv[2]).resolve()
summary = json.loads(summary_path.read_text(encoding="utf-8"))
metrics = summary["aggregate_metrics"]
output = {
    "status": "p418_energy_evaluation_unavailable_outside_registered_temperature_range",
    "model_summary": str(summary_path),
    "evaluated_roles": ["test"],
    "registered_fluid_temperature_range_K": metrics[
        "registered_fluid_temperature_range_K"
    ],
    "predicted_fluid_temperature_minimum_K": metrics[
        "predicted_fluid_temperature_minimum_K"
    ],
    "predicted_fluid_temperature_maximum_K": metrics[
        "predicted_fluid_temperature_maximum_K"
    ],
    "predicted_fluid_temperature_outside_registered_range_fraction": metrics[
        "predicted_fluid_temperature_outside_registered_range_fraction"
    ],
    "registered_solid_temperature_range_K": metrics[
        "registered_solid_temperature_range_K"
    ],
    "predicted_solid_temperature_minimum_K": metrics[
        "predicted_solid_temperature_minimum_K"
    ],
    "predicted_solid_temperature_maximum_K": metrics[
        "predicted_solid_temperature_maximum_K"
    ],
    "predicted_solid_temperature_outside_registered_range_fraction": metrics[
        "predicted_solid_temperature_outside_registered_range_fraction"
    ],
    "reason": (
        "The frozen prediction leaves at least one registered phase-temperature "
        "range; the thermophysical equations were not extrapolated."
    ),
    "new_physical_parameters": [],
}
output_path.write_text(
    json.dumps(output, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
  fi
done

python3 "${ROOT}/code/summarize_hccb_p418_high_re_model_comparison.py" \
  --data-only-summary "${OUTPUT_DIRS[0]}/summary.json" \
  --physics-summary "${OUTPUT_DIRS[1]}/summary.json" \
  --factorized-summary "${OUTPUT_DIRS[2]}/summary.json" \
  --data-only-energy "${OUTPUT_DIRS[0]}/energy_balance_summary.json" \
  --physics-energy "${OUTPUT_DIRS[1]}/energy_balance_summary.json" \
  --factorized-energy "${OUTPUT_DIRS[2]}/energy_balance_summary.json" \
  --output-dir "${OUTPUT_ROOT}/comparison" \
  --latex-output "${ROOT}/manuscript/generated_high_re_comparison.tex"
