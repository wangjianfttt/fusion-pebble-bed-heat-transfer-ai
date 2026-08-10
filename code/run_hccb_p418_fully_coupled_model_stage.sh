#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${ROOT:-${SCRIPT_ROOT}}
EXECUTE=${EXECUTE:-0}
DEVICE=${DEVICE:-cuda}
TORCH_THREADS=${TORCH_THREADS:-4}
SEED=${SEED:-20260723}
DATASET_INDEX=${DATASET_INDEX:-${ROOT}/results/hccb_p418_fully_coupled_steps_12/regional_sequences/dataset_index.json}
SPLITS=${SPLITS:-${ROOT}/parameters/hccb_p418_step_response_splits.json}
SPLIT_NAME=${SPLIT_NAME:-pair_disjoint_stress_test}
RESIDUAL_GEOMETRY=${RESIDUAL_GEOMETRY:-${ROOT}/results/hccb_p418_subface_residual_geometry_r2/subface_residual_geometry.npz}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/results/hccb_p418_fully_coupled_model_comparison}
PAUSE_MARKER=${PAUSE_MARKER:-${ROOT}/control/PAUSE_NEW_P418_CASES_FOR_CLOUD_MIGRATION}
ALLOW_PAUSED_WORKSTATION_RUN=${ALLOW_PAUSED_WORKSTATION_RUN:-0}

PROTOCOL=(
  python3 "${ROOT}/code/run_hccb_p418_loss_balancing_protocol.py"
  --dataset-index "${DATASET_INDEX}"
  --splits "${SPLITS}"
  --split-name "${SPLIT_NAME}"
  --residual-geometry "${RESIDUAL_GEOMETRY}"
  --output-root "${OUTPUT_ROOT}"
  --device "${DEVICE}"
  --torch-threads "${TORCH_THREADS}"
  --seed "${SEED}"
)

if [[ ${EXECUTE} != 1 ]]; then
  "${PROTOCOL[@]}" plan
  cat <<EOF
P418全流热耦合模型只打印运行顺序，没有开始训练。
第一阶段：只用训练曲线和检查曲线比较固定等权与3组文献ReLoBRaLo设置。
第二阶段：固定损失组合后，只读取一次独立测试曲线。
设置 EXECUTE=1 后才会正式运行。
EOF
  exit 0
fi

if [[ -f ${PAUSE_MARKER} && ${ALLOW_PAUSED_WORKSTATION_RUN} != 1 ]]; then
  echo "new P418 calculations are paused for cloud migration: ${PAUSE_MARKER}" >&2
  exit 3
fi

for required in "${DATASET_INDEX}" "${SPLITS}" "${RESIDUAL_GEOMETRY}"; do
  if [[ ! -f ${required} ]]; then
    echo "required full-coupled model input is missing: ${required}" >&2
    exit 4
  fi
done

SELECTION_RECORD=${OUTPUT_ROOT}/selected_loss_balancing_method.json
if [[ ! -f ${SELECTION_RECORD} ]]; then
  "${PROTOCOL[@]}" selection
fi

if ! find "${OUTPUT_ROOT}" -mindepth 2 -maxdepth 2 -name final_summary.json -print -quit |
  grep -q .; then
  "${PROTOCOL[@]}" final
fi

FINAL_SUMMARY=$(find "${OUTPUT_ROOT}" -mindepth 2 -maxdepth 2 -name final_summary.json -print -quit)
if [[ -z ${FINAL_SUMMARY} ]]; then
  echo "full-coupled final model summary was not produced" >&2
  exit 5
fi

printf 'P418全流热耦合模型已完成：%s\n' "${FINAL_SUMMARY}"
