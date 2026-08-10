#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${ROOT:-${SCRIPT_ROOT}}
MATRIX_ROOT=${MATRIX_ROOT:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3}
CONFIG=${CONFIG:-${ROOT}/parameters/hccb_p418_fully_coupled_timestep_sensitivity.json}
BASE_PLAN=${BASE_PLAN:-${ROOT}/parameters/hccb_p418_fully_coupled_step_plan.json}
STEP_ROOT=${STEP_ROOT:-${ROOT}/hccb_p418_fully_coupled_timestep_sensitivity}
RESULT_ROOT=${RESULT_ROOT:-${ROOT}/results/hccb_p418_fully_coupled_timestep_sensitivity}
NP_PER_CASE=${NP_PER_CASE:-32}
EXECUTE=${EXECUTE:-0}
ALLOW_PAUSED_WORKSTATION_RUN=${ALLOW_PAUSED_WORKSTATION_RUN:-0}

if [[ ${EXECUTE} != 1 ]]; then
  python3 - "${CONFIG}" "${STEP_ROOT}" "${RESULT_ROOT}" "${NP_PER_CASE}" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
print("P418 fully coupled time-step study only; no OpenFOAM command was started.")
print(f"sequence={cfg['sequence_id']}")
print("initial_delta_t_s=" + ",".join(str(value) for value in cfg["delta_t_s"]))
print(f"step_root={sys.argv[2]}")
print(f"result_root={sys.argv[3]}")
print(f"mpi_ranks_per_case={sys.argv[4]}")
print("Set EXECUTE=1 only after both steady endpoints are complete.")
PY
  exit 0
fi

mkdir -p "${STEP_ROOT}" "${RESULT_ROOT}"
mapfile -t DELTA_T_VALUES < <(
  python3 -c 'import json,sys; print("\n".join(map(str,json.load(open(sys.argv[1]))["delta_t_s"])))' \
    "${CONFIG}"
)
for delta_t in "${DELTA_T_VALUES[@]}"; do
  label=$(python3 -c 'import sys; value=("%.12g"%float(sys.argv[1])).replace("-","m").replace("+","").replace(".","p"); print("dt_"+value)' "${delta_t}")
  plan="${RESULT_ROOT}/${label}_plan.json"
  python3 - "${BASE_PLAN}" "${CONFIG}" "${delta_t}" "${plan}" <<'PY'
import json, sys
base = json.load(open(sys.argv[1]))
cfg = json.load(open(sys.argv[2]))
dt = float(sys.argv[3])
sequence = next(
    row for row in base["sequences"] if row["sequence_id"] == cfg["sequence_id"]
)
formal = cfg["formal_time_step_schedule"]
scale = dt / float(formal[0]["delta_t_s"])
base["analysis_kind"] = "fully_coupled_time_step_sensitivity"
base["sequences"] = [sequence]
base["numerical_time_design"] = {
    "duration_s": cfg["duration_s"],
    "delta_t_s": dt,
    "time_step_schedule": [
        {**row, "delta_t_s": float(row["delta_t_s"]) * scale}
        for row in formal
    ],
    "field_write_interval_s": cfg["field_write_interval_s"],
    "field_write_schedule": cfg["field_write_schedule"],
    "ddt_scheme": "Euler",
    "write_control": "runTime",
    "origin": (
        "Predeclared fully coupled numerical time-step sensitivity; no physical "
        "parameter or operating endpoint is changed."
    ),
}
base["new_physical_parameters"] = []
open(sys.argv[4], "w").write(json.dumps(base, ensure_ascii=False, indent=2) + "\n")
PY
  ROOT="${ROOT}" MATRIX_ROOT="${MATRIX_ROOT}" \
    STEP_ROOT="${STEP_ROOT}/${label}" RESULT_DIR="${RESULT_ROOT}/${label}" \
    PLAN="${plan}" NP_PER_CASE="${NP_PER_CASE}" CONCURRENT_CASES=1 \
    EXECUTE=1 ALLOW_PAUSED_WORKSTATION_RUN="${ALLOW_PAUSED_WORKSTATION_RUN}" \
    REQUIRE_TIMESTEP_SENSITIVITY=0 COMPARE_FIXED=0 \
    bash "${ROOT}/code/run_hccb_p418_fully_coupled_step_responses.sh"
done

python3 "${ROOT}/code/compare_hccb_p418_thermal_timestep_sensitivity.py" \
  --config "${CONFIG}" \
  --result-root "${RESULT_ROOT}" \
  --output-dir "${RESULT_ROOT}" \
  --analysis-kind fully_coupled_flow_heat
