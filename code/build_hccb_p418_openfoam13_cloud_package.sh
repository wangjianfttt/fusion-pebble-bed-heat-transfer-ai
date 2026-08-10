#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SOURCE_CASE=${SOURCE_CASE:-${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3/u0p05_T300_q4p85}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/cloud_migration_build}
PACKAGE_NAME=${PACKAGE_NAME:-p418_openfoam13_minimal_case}
PACKAGE_DIR=${OUTPUT_ROOT}/${PACKAGE_NAME}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}

for path in "${OUTPUT_ROOT}" "${PACKAGE_DIR}"; do
  case "${path}" in
    /home|/home/*)
      echo "cloud package output must not be under /home: ${path}" >&2
      exit 2
      ;;
  esac
done
for required in \
  "${SOURCE_CASE}/0" \
  "${SOURCE_CASE}/constant" \
  "${SOURCE_CASE}/system" \
  "${SOURCE_CASE}/cht_smoke_metadata.json" \
  "${SOURCE_CASE}/formal_sample_complete.json" \
  "${SOURCE_CASE}/cht_result_summary_200.json" \
  "${ROOT}/parameters/hccb_p418_physical_parameter_sources.csv" \
  "${ROOT}/parameters/literature_parameter_manifest.csv" \
  "${ROOT}/cloud_migration/cloud_case_matrix.csv" \
  "${ROOT}/cloud_migration/cloud_case_matrix_summary.json" \
  "${ROOT}/cloud_migration/pending_case_ids.txt" \
  "${ROOT}/code/run_with_resource_monitor.py" \
  "${ROOT}/code/summarize_openfoam_cloud_resources.py"; do
  if [[ ! -e ${required} ]]; then
    echo "required cloud-package input is missing: ${required}" >&2
    exit 1
  fi
done

rm -rf "${PACKAGE_DIR}"
mkdir -p "${PACKAGE_DIR}/case_template" "${PACKAGE_DIR}/source_record" \
  "${PACKAGE_DIR}/parameters" "${PACKAGE_DIR}/scripts"
cp -a "${SOURCE_CASE}/0" "${PACKAGE_DIR}/case_template/"
cp -a "${SOURCE_CASE}/constant" "${PACKAGE_DIR}/case_template/"
cp -a "${SOURCE_CASE}/system" "${PACKAGE_DIR}/case_template/"
cp -a "${SOURCE_CASE}/cht_smoke_metadata.json" "${PACKAGE_DIR}/case_template/"
cp -a "${SOURCE_CASE}/cht_smoke_metadata.json" "${PACKAGE_DIR}/source_record/"
cp -a "${SOURCE_CASE}/formal_sample_complete.json" "${PACKAGE_DIR}/source_record/"
cp -a "${SOURCE_CASE}/cht_result_summary_200.json" "${PACKAGE_DIR}/source_record/"
cp -a "${ROOT}/parameters/hccb_p418_physical_parameter_sources.csv" \
  "${PACKAGE_DIR}/parameters/"
cp -a "${ROOT}/parameters/literature_parameter_manifest.csv" \
  "${PACKAGE_DIR}/parameters/"
cp -a "${ROOT}/cloud_migration/README_CN.md" "${PACKAGE_DIR}/"
cp -a "${ROOT}/cloud_migration/VERSION_DEPENDENCIES_CN.md" "${PACKAGE_DIR}/"
cp -a "${ROOT}/cloud_migration/cloud_case_matrix.csv" "${PACKAGE_DIR}/"
cp -a "${ROOT}/cloud_migration/cloud_case_matrix_summary.json" "${PACKAGE_DIR}/"
cp -a "${ROOT}/cloud_migration/pending_case_ids.txt" "${PACKAGE_DIR}/"
cp -a "${ROOT}/cloud_migration/run_openfoam13_case.sh" \
  "${PACKAGE_DIR}/scripts/"
cp -a "${ROOT}/cloud_migration/run_openfoam13_smoke.sh" \
  "${PACKAGE_DIR}/scripts/"
cp -a "${ROOT}/cloud_migration/run_openfoam13_formal.sh" \
  "${PACKAGE_DIR}/scripts/"
cp -a "${ROOT}/cloud_migration/postprocess_openfoam13_case.sh" \
  "${PACKAGE_DIR}/scripts/"
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
cp -a "${ROOT}/cloud_migration/submit_slurm_example.sh" \
  "${PACKAGE_DIR}/scripts/"
chmod +x "${PACKAGE_DIR}/scripts/"*.sh

if find "${PACKAGE_DIR}" -mindepth 1 \
    \( -name 'processor*' -o -name dynamicCode -o -name training_sample_200_schema3 \
       -o -name 200 -o -name postProcessing \) -print -quit | grep -q .; then
  echo "excluded result or machine-specific files entered the cloud package" >&2
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
