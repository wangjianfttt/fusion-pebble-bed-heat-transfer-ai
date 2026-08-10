#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MATRIX_ROOT=${MATRIX_ROOT:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3}
INPUT_CHECK=${INPUT_CHECK:-${ROOT}/results/hccb_p418_60_actual_case_input_check/summary.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/cloud_migration_build}
PACKAGE_NAME=${PACKAGE_NAME:-p418_openfoam13_pending_46}
PACKAGE_DIR=${OUTPUT_ROOT}/${PACKAGE_NAME}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}

case "${OUTPUT_ROOT}" in
  /home|/home/*)
    echo "cloud package output must not be under /home: ${OUTPUT_ROOT}" >&2
    exit 2
    ;;
esac

python3 "${ROOT}/code/build_hccb_p418_openfoam13_batch_inputs.py" \
  --matrix-root "${MATRIX_ROOT}" \
  --cloud-table "${ROOT}/cloud_migration/cloud_case_matrix.csv" \
  --input-check "${INPUT_CHECK}" \
  --output-dir "${PACKAGE_DIR}"

mkdir -p "${PACKAGE_DIR}/scripts" "${PACKAGE_DIR}/parameters"
cp -a "${ROOT}/cloud_migration/README_CN.md" "${PACKAGE_DIR}/"
cp -a "${ROOT}/cloud_migration/VERSION_DEPENDENCIES_CN.md" "${PACKAGE_DIR}/"
cp -a "${ROOT}/cloud_migration/cloud_case_matrix.csv" "${PACKAGE_DIR}/"
cp -a "${ROOT}/cloud_migration/cloud_case_matrix_summary.json" "${PACKAGE_DIR}/"
cp -a "${ROOT}/cloud_migration/pending_case_ids.txt" "${PACKAGE_DIR}/"
cp -a "${ROOT}/cloud_migration/run_openfoam13_case.sh" "${PACKAGE_DIR}/scripts/"
cp -a "${ROOT}/cloud_migration/run_openfoam13_batch_case.sh" "${PACKAGE_DIR}/scripts/"
cp -a "${ROOT}/cloud_migration/postprocess_openfoam13_case.sh" "${PACKAGE_DIR}/scripts/"
cp -a "${ROOT}/code/compute_hccb_gmsh_boundary_heat_flows.py" \
  "${PACKAGE_DIR}/scripts/"
cp -a "${ROOT}/code/summarize_hccb_gmsh_cht_result.py" \
  "${PACKAGE_DIR}/scripts/"
cp -a "${ROOT}/code/compare_hccb_p418_cloud_reference.py" \
  "${PACKAGE_DIR}/scripts/"
cp -a "${ROOT}/code/run_with_resource_monitor.py" \
  "${PACKAGE_DIR}/scripts/"
cp -a "${ROOT}/code/summarize_openfoam_cloud_resources.py" \
  "${PACKAGE_DIR}/scripts/"
cp -a "${ROOT}/cloud_migration/submit_slurm_array_example.sh" "${PACKAGE_DIR}/scripts/"
cp -a "${ROOT}/parameters/hccb_p418_physical_parameter_sources.csv" \
  "${PACKAGE_DIR}/parameters/"
cp -a "${ROOT}/parameters/literature_parameter_manifest.csv" \
  "${PACKAGE_DIR}/parameters/"
chmod +x "${PACKAGE_DIR}/scripts/"*.sh

if find "${PACKAGE_DIR}" -mindepth 1 \
  \( -name 'processor*' -o -name dynamicCode -o -name postProcessing \
     -o -name training_sample_200_schema3 -o -name 200 \) \
  -print -quit | grep -q .; then
  echo "solver output entered the pending-case package" >&2
  exit 1
fi

(
  cd "${PACKAGE_DIR}"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)

if [[ ${CREATE_ARCHIVE} == 1 ]]; then
  if command -v zstd >/dev/null 2>&1; then
    archive=${OUTPUT_ROOT}/${PACKAGE_NAME}.tar.zst
    rm -f "${archive}"
    tar --zstd -cf "${archive}" -C "${OUTPUT_ROOT}" "${PACKAGE_NAME}"
  else
    archive=${OUTPUT_ROOT}/${PACKAGE_NAME}.tar.gz
    rm -f "${archive}"
    tar -czf "${archive}" -C "${OUTPUT_ROOT}" "${PACKAGE_NAME}"
  fi
  sha256sum "${archive}" > "${archive}.sha256"
  echo "archive=${archive}"
fi
echo "package_dir=${PACKAGE_DIR}"
