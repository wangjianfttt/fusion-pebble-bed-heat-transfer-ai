#!/usr/bin/env bash
set -eo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PILOT_ROOT=${PILOT_ROOT:-${ROOT}/hccb_dense_cht_p418_pilot}
MESH_CASE=${MESH_CASE:-${ROOT}/hccb_dense_cht_native_r2}
SAMPLE=${SAMPLE:-${PILOT_ROOT}/u0p20_T700_q6p85/training_sample_300/fields_and_topology.npz}
OUTPUT=${OUTPUT:-${ROOT}/results/hccb_p418_regional_topology_r2}

while [[ $(find "${PILOT_ROOT}" -name formal_sample_complete.json 2>/dev/null | wc -l) -lt 5 ]]; do
  sleep 60
done

if [[ ! -f "${MESH_CASE}/native_multiregion_graph/native_multiregion_graph.npz" ]]; then
  python3 "${ROOT}/code/export_openfoam_multiregion_native_graph.py" \
    --case "${MESH_CASE}" --fluid-region fluid --solid-region solid \
    --interface-summary "${MESH_CASE}/interface_pairs/summary.json" \
    --output-dir "${MESH_CASE}/native_multiregion_graph"
fi

python3 "${ROOT}/code/build_hccb_p418_regional_topology.py" \
  --shared-topology "${SAMPLE}" \
  --native-graph "${MESH_CASE}/native_multiregion_graph/native_multiregion_graph.npz" \
  --levels 6 --subsample-factor 4 \
  --output-dir "${OUTPUT}"
