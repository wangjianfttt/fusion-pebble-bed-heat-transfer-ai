#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MATRIX_ROOT=${MATRIX_ROOT:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3}
CONFIG=${CONFIG:-${ROOT}/parameters/hccb_p418_thermal_step_timestep_sensitivity.json}
BASE_PLAN=${BASE_PLAN:-${ROOT}/parameters/hccb_p418_transient_step_plan.json}
STEP_ROOT=${STEP_ROOT:-${ROOT}/hccb_p418_thermal_timestep_sensitivity}
RESULT_ROOT=${RESULT_ROOT:-${ROOT}/results/hccb_p418_thermal_timestep_sensitivity}
NP_PER_CASE=${NP_PER_CASE:-32}

mkdir -p "${STEP_ROOT}" "${RESULT_ROOT}"

mapfile -t DELTA_T_VALUES < <(python3 -c 'import json,sys; print("\n".join(map(str,json.load(open(sys.argv[1]))["delta_t_s"])))' "${CONFIG}")
for delta_t in "${DELTA_T_VALUES[@]}"; do
  label=$(python3 -c 'import sys; value=("%.12g"%float(sys.argv[1])).replace("-","m").replace("+","").replace(".","p"); print("dt_"+value)' "${delta_t}")
  plan="${RESULT_ROOT}/${label}_plan.json"
  python3 - "${BASE_PLAN}" "${CONFIG}" "${delta_t}" "${plan}" <<'PY'
import json, sys
base=json.load(open(sys.argv[1]))
cfg=json.load(open(sys.argv[2]))
dt=float(sys.argv[3])
sequence=next(row for row in base["sequences"] if row["sequence_id"]==cfg["sequence_id"])
base["sequences"]=[sequence]
base["numerical_time_design"]["duration_s"]=cfg["duration_s"]
base["numerical_time_design"]["delta_t_s"]=dt
base["numerical_time_design"]["field_write_interval_s"]=cfg["field_write_interval_s"]
base["numerical_time_design"]["field_write_schedule"]=cfg["field_write_schedule"]
formal=cfg["formal_time_step_schedule"]
formal_initial=float(formal[0]["delta_t_s"])
scale=dt/formal_initial
base["numerical_time_design"]["time_step_schedule"]=[
    {**row, "delta_t_s": float(row["delta_t_s"])*scale}
    for row in formal
]
base["purpose"]="Numerical time-step sensitivity for one representative P418 thermal step."
base["new_physical_parameters"]=[]
open(sys.argv[4],"w").write(json.dumps(base,indent=2)+"\n")
PY
  ROOT="${ROOT}" MATRIX_ROOT="${MATRIX_ROOT}" \
    STEP_ROOT="${STEP_ROOT}/${label}" RESULT_DIR="${RESULT_ROOT}/${label}" \
    PLAN="${plan}" NP_PER_CASE="${NP_PER_CASE}" CONCURRENT_CASES=1 \
    RUN_MODEL_TRAINING=0 \
    bash "${ROOT}/code/run_hccb_p418_step_responses.sh"
done

python3 "${ROOT}/code/compare_hccb_p418_thermal_timestep_sensitivity.py" \
  --config "${CONFIG}" \
  --result-root "${RESULT_ROOT}" \
  --output-dir "${RESULT_ROOT}"
