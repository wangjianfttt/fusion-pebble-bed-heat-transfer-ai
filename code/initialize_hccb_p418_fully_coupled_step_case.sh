#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 FULLY_COUPLED_STEP_CASE" >&2
  exit 2
fi

CASE_DIR=$(cd "$1" && pwd)
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
METADATA=${CASE_DIR}/fully_coupled_step_metadata.json

if [[ ! -f ${METADATA} ]]; then
  echo "missing ${METADATA}" >&2
  exit 1
fi

read_metadata() {
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {"d": d}))' \
    "${METADATA}" "$1"
}

SOURCE_CASE=$(read_metadata 'd["source_case"]')
SOURCE_TIME=$(read_metadata 'd["source_final_time_s"]')
TARGET_CASE=$(read_metadata 'd["target_case"]')
TARGET_TIME=$(read_metadata 'd["target_final_time_s"]')
TARGET_U=$(read_metadata 'd["target_parameters"].get("pore_opening_boundary_velocity_m_s", d["target_parameters"]["inlet_velocity_m_s"])')
TARGET_T=$(read_metadata 'd["target_parameters"]["inlet_temperature_K"]')
TARGET_INLET_PHI=$(
  foamDictionary "${TARGET_CASE}/${TARGET_TIME}/fluid/phi" -writePrecision 17 \
    -entry 'boundaryField/inlet/value' -value
)
if [[ -z ${TARGET_INLET_PHI} ]]; then
  echo "target inlet phi is empty: ${TARGET_CASE}/${TARGET_TIME}/fluid/phi" >&2
  exit 1
fi

for field in U p p_rgh phi T; do
  cp "${SOURCE_CASE}/${SOURCE_TIME}/fluid/${field}" "${CASE_DIR}/0/fluid/${field}"
done
cp "${SOURCE_CASE}/${SOURCE_TIME}/solid/T" "${CASE_DIR}/0/solid/T"

for field in U T p p_rgh phi; do
  foamDictionary "${CASE_DIR}/0/fluid/${field}" -writePrecision 10 \
    -entry 'FoamFile/location' -set '"0/fluid"'
done
foamDictionary "${CASE_DIR}/0/solid/T" -writePrecision 10 \
  -entry 'FoamFile/location' -set '"0/solid"'

# The complete source state is retained internally. At the inlet, U, T and phi
# are taken from the target endpoint so the imposed boundary velocity and mass
# flux are consistent from t=0 onward.
foamDictionary "${CASE_DIR}/0/fluid/U" -writePrecision 10 \
  -entry 'boundaryField/inlet/value' -set "uniform (0 0 ${TARGET_U})"
foamDictionary "${CASE_DIR}/0/fluid/T" -writePrecision 10 \
  -entry 'boundaryField/inlet/value' -set "uniform ${TARGET_T}"
foamDictionary "${CASE_DIR}/0/fluid/T" -writePrecision 10 \
  -entry 'boundaryField/outlet/inletValue' -set "uniform ${TARGET_T}"
foamDictionary "${CASE_DIR}/0/fluid/phi" -writePrecision 17 \
  -entry 'boundaryField/inlet/value' -set "${TARGET_INLET_PHI}"

python3 "${ROOT}/code/verify_hccb_p418_fully_coupled_step_initialization.py" \
  --case "${CASE_DIR}" --write-record

echo "prepared fully coupled source state and target boundary: $(basename "${CASE_DIR}")"
