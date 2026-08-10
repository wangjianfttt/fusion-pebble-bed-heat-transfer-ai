#!/usr/bin/env bash
set -eo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MATRIX_ROOT=${MATRIX_ROOT:-${ROOT}/hccb_dense_cht_p418_pilot}
REFERENCE=${REFERENCE:-u0p20_T700_q6p85}
TIME_NAME=${TIME_NAME:-300}
INTERFACE_PAIRS=${INTERFACE_PAIRS:-${ROOT}/hccb_dense_cht_native_r2/interface_pairs/interface_face_pairs.npz}
PARAMETERS=${PARAMETERS:-${ROOT}/parameters/literature_parameter_manifest.csv}
SAMPLE_NAME=training_sample_${TIME_NAME}_schema3

conditions=(
  u0p20_T700_q6p85
  u0p05_T300_q8p85
  u0p05_T900_q8p85
  u0p25_T300_q4p85
  u0p25_T900_q4p85
)

reference_sample=${MATRIX_ROOT}/${REFERENCE}/${SAMPLE_NAME}/fields_and_topology.npz
reference_metadata=${MATRIX_ROOT}/${REFERENCE}/${SAMPLE_NAME}/metadata.json
if [[ ! -f ${reference_sample} || ! -f ${reference_metadata} ]]; then
  echo "completed schema-v3 reference sample is required: ${reference_sample}" >&2
  exit 1
fi
python3 - "${reference_metadata}" <<'PY'
import json, sys
if json.load(open(sys.argv[1])).get("schema_version") != 3:
    raise SystemExit("reference sample is not schema version 3")
PY

for condition in "${conditions[@]}"; do
  case_dir=${MATRIX_ROOT}/${condition}
  output_dir=${case_dir}/${SAMPLE_NAME}
  if [[ ${condition} != ${REFERENCE} ]] \
      && [[ ! -f ${output_dir}/fields_and_topology.npz \
            || ! -f ${output_dir}/metadata.json ]]; then
    rm -rf "${output_dir}"
    python3 "${ROOT}/code/export_hccb_cht_training_sample.py" \
      --case "${case_dir}" --time "${TIME_NAME}" \
      --parameter-manifest "${PARAMETERS}" \
      --interface-pairs "${INTERFACE_PAIRS}" \
      --reuse-boundary-geometry "${reference_sample}" \
      --output-dir "${output_dir}" \
      > "${case_dir}/log.training_export.${TIME_NAME}.schema3"
  fi
  python3 "${ROOT}/code/summarize_hccb_training_sample_boundary.py" \
    --sample "${output_dir}/fields_and_topology.npz" \
    --metadata "${output_dir}/metadata.json" \
    --output "${output_dir}/boundary_summary.json" \
    > "${output_dir}/log.boundary_summary"
done

dataset=${ROOT}/hccb_dense_cht_p418_pilot_dataset_schema3
rm -rf "${dataset}"
python3 "${ROOT}/code/build_hccb_p418_shared_mesh_dataset.py" \
  --matrix-root "${MATRIX_ROOT}" \
  --sample-directory-name "${SAMPLE_NAME}" \
  --output-dir "${dataset}" \
  > "${ROOT}/results/hccb_p418_pilot_dataset_schema3.log"
