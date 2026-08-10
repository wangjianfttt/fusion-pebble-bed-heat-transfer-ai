#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 FULLY_COUPLED_STEP_CASE" >&2
  exit 2
fi

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CASE_DIR=$(cd "$1" && pwd)
METADATA=${CASE_DIR}/fully_coupled_step_metadata.json

if [[ ! -f ${METADATA} ]]; then
  echo "missing ${METADATA}" >&2
  exit 1
fi

set +u
source /opt/openfoam13/etc/bashrc || true
set -u

times=$(python3 - "${METADATA}" <<'PY'
import json, sys
row = json.load(open(sys.argv[1]))
print(",".join(str(value) for value in row["snapshot_times_s"]))
PY
)
final_time=$(python3 - "${METADATA}" <<'PY'
import json, sys
row = json.load(open(sys.argv[1]))
print(row["duration_s"])
PY
)

reconstructPar -case "${CASE_DIR}" -allRegions -time "${times}" \
  > "${CASE_DIR}/log.reconstructPar.fully_coupled" 2>&1
python3 "${ROOT}/code/compute_hccb_gmsh_boundary_heat_flows.py" \
  --case "${CASE_DIR}" --time "${final_time}" \
  --output "${CASE_DIR}/boundary_heat_flows_${final_time}.json" \
  > "${CASE_DIR}/log.boundary_heat_flows.${final_time}"
python3 "${ROOT}/code/summarize_hccb_gmsh_cht_result.py" \
  --case "${CASE_DIR}" \
  --log "${CASE_DIR}/log.foamMultiRun.fully_coupled" \
  --boundary-heat-flows "${CASE_DIR}/boundary_heat_flows_${final_time}.json" \
  --output "${CASE_DIR}/cht_result_summary_${final_time}.json" \
  > "${CASE_DIR}/log.result_summary.${final_time}"

python3 - "${CASE_DIR}" "${final_time}" <<'PY'
import json
import math
import pathlib
import sys

case = pathlib.Path(sys.argv[1])
final_time = sys.argv[2]
metadata = json.load(open(case / "fully_coupled_step_metadata.json"))
summary = json.load(open(case / f"cht_result_summary_{final_time}.json"))
if not summary["solver_finished"]:
    raise SystemExit("fully coupled OpenFOAM log does not contain a finished solution")
mass_difference = float(summary["flow"]["relative_mass_difference"])
energy_difference = float(summary["heat_balance"]["relative_energy_difference"])
if not math.isfinite(mass_difference) or not math.isfinite(energy_difference):
    raise SystemExit("fully coupled mass or energy result is not finite")
payload = {
    "status": "completed_p418_fully_coupled_flow_heat_step_response",
    "transient_model": metadata["transient_model"],
    "sequence_id": metadata["sequence_id"],
    "source_condition_id": metadata["source_condition_id"],
    "target_condition_id": metadata["target_condition_id"],
    "changed_physical_input": metadata["changed_physical_input"],
    "duration_s": metadata["duration_s"],
    "solver_finished": True,
    "relative_mass_difference": mass_difference,
    "relative_energy_difference": energy_difference,
    "new_physical_parameters": [],
}
(case / "fully_coupled_step_response_complete.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

if [[ ${REMOVE_PROCESSORS_AFTER_EXPORT:-1} == 1 ]]; then
  rm -rf "${CASE_DIR}"/processor*
fi
