#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OPENFOAM_BASHRC=${OPENFOAM_BASHRC:-/opt/openfoam13/etc/bashrc}
REFERENCE_MESH=${REFERENCE_MESH:-${ROOT}/hccb_dense_snappy_g2_nativezone_r2}
PACKING=${PACKING:-${ROOT}/data/apd006_hccb_source_sequence_target_packings/seed101_s80_xlo_ycentre/packing.npz}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/work/hccb_p418_gci_mesh_preflight_20260729}
CELLS_PER_DIAMETER=${CELLS_PER_DIAMETER:?set the background cells per particle diameter}
LABEL=${LABEL:?set a unique mesh label}
OUTPUT_DIR=${OUTPUT_ROOT}/${LABEL}

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "candidate already exists: ${OUTPUT_DIR}" >&2
  exit 2
fi
if [[ ! -f "${OPENFOAM_BASHRC}" ]]; then
  echo "OpenFOAM environment is missing: ${OPENFOAM_BASHRC}" >&2
  exit 2
fi
if [[ ! -f "${REFERENCE_MESH}/constant/triSurface/solid.obj" ]]; then
  echo "reference particle surface is missing" >&2
  exit 2
fi
if [[ ! -f "${PACKING}" ]]; then
  echo "packing input is missing: ${PACKING}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"

python3 "${ROOT}/code/build_hccb_dense_snappy_case.py" \
  --packing "${PACKING}" \
  --output-dir "${OUTPUT_DIR}" \
  --crop-box-dp 1.234 5.157 3.921 8.163 2.906 6.396 \
  --cells-per-diameter "${CELLS_PER_DIAMETER}" \
  --sphere-subdivisions 3 \
  --surface-refinement 2 \
  --cells-between-levels 2 \
  --reuse-surface-case "${REFERENCE_MESH}" \
  --solid-cell-source snappy-zone \
  > "${OUTPUT_DIR}.build.json"

set +u
# shellcheck disable=SC1090
source "${OPENFOAM_BASHRC}"
set -u

(
  cd "${OUTPUT_DIR}"
  ./Allmesh
  checkMesh -region fluid > log.checkMesh.fluid.basic 2>&1
  checkMesh -region solid > log.checkMesh.solid.basic 2>&1
)

python3 "${ROOT}/code/summarize_hccb_dense_mesh_check.py" \
  --case "${OUTPUT_DIR}" \
  --fluid-log "${OUTPUT_DIR}/log.checkMesh.fluid.basic" \
  --solid-log "${OUTPUT_DIR}/log.checkMesh.solid.basic" \
  --output "${OUTPUT_DIR}/mesh_summary.json"
