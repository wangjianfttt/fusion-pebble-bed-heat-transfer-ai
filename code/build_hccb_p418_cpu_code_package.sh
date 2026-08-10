#!/usr/bin/env bash
set -euo pipefail
export COPYFILE_DISABLE=1

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/cloud_migration_build}
PACKAGE_NAME=${PACKAGE_NAME:-p418_cpu_preprocess_model_code}
ARCHIVE=${OUTPUT_ROOT}/${PACKAGE_NAME}.tar.zst
TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/${PACKAGE_NAME}.XXXXXX")
STAGE=${TMP_ROOT}/${PACKAGE_NAME}

cleanup() {
  rm -rf "${TMP_ROOT}"
}
trap cleanup EXIT

mkdir -p "${OUTPUT_ROOT}" "${STAGE}"

rsync -rlt \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '._*' \
  --exclude '.DS_Store' \
  --exclude 'CPU_MIGRATION_READONLY_STATUS_20260723_CN.md' \
  --exclude 'cloud_submission_manifest.json' \
  --exclude '易算云提交清单_简明_CN.md' \
  "${ROOT}/code" \
  "${ROOT}/parameters" \
  "${ROOT}/tests" \
  "${ROOT}/algorithms" \
  "${ROOT}/cloud_migration" \
  "${STAGE}/"

for file in \
  Makefile \
  README.md \
  CURRENT_STATUS_CN.md \
  PROCESS_LOG_CN.md \
  RESEARCH_MAINLINE_CN.md \
  ALGORITHM_ARCHITECTURE_NOTES_CN.md \
  EXPERIMENTAL_VALIDATION_PLAN_CN.md \
  实验实施步骤_简明_CN.md \
  研究主线_简明版_CN.md \
  requirements-p418.txt \
  requirements-pinn.txt \
  requirements-tools.txt; do
  if [[ -f "${ROOT}/${file}" ]]; then
    cp "${ROOT}/${file}" "${STAGE}/${file}"
  fi
done

# The experiment interface tests use the blank CSV templates. They contain
# headers only and no measurements or fitted physical values.
rsync -rlt \
  --exclude '._*' \
  --exclude '.DS_Store' \
  "${ROOT}/experimental_data_templates" \
  "${STAGE}/"

# Keep the small PREMUX/TESOMEX observation tensors and their exact source
# files so the cloud package can rebuild and verify the experimental/computed
# observation-source separation without the full OpenFOAM or literature tree.
rsync -rlt \
  --exclude '._*' \
  --exclude '.DS_Store' \
  "${ROOT}/data/hccb_heat_ai_external_observations" \
  "${STAGE}/data/"
for file in \
  data/apd006_premux_steady_thermocouples/premux_steady_thermocouples_digitized.csv \
  data/apd006_premux_nominal_2d_geometry/thermocouple_coordinates_nominal.csv \
  cases/apd006_premux_moose_steady_vhi/medium/premux_vhi.msh \
  data/external/tesomex/figure_5_8_radial_profiles/digitized_profiles.csv \
  results/apd006_tesomex_1d_transient_baseline/profile_comparison.csv \
  results/apd006_tesomex_1d_transient_baseline/summary.json; do
  if [[ ! -f "${ROOT}/${file}" ]]; then
    echo "missing external observation source file: ${ROOT}/${file}" >&2
    exit 2
  fi
  mkdir -p "${STAGE}/$(dirname "${file}")"
  cp "${ROOT}/${file}" "${STAGE}/${file}"
done

# Keep the manuscript source and result-source table because the final refresh
# tests and high-Re comparison now write directly into the paper. Exclude only
# local LaTeX build products; the cloud package can regenerate them.
rsync -rlt \
  --exclude 'main.pdf' \
  --exclude 'main.aux' \
  --exclude 'main.bbl' \
  --exclude 'main.blg' \
  --exclude 'main.fdb_latexmk' \
  --exclude 'main.fls' \
  --exclude 'main.log' \
  --exclude 'main.out' \
  --exclude 'main.spl' \
  --exclude 'main.synctex.gz' \
  --exclude '._*' \
  --exclude '.DS_Store' \
  "${ROOT}/manuscript" \
  "${STAGE}/"

# Keep the current plain-Chinese findings reproducible without copying the
# multi-gigabyte OpenFOAM fields. These are small derived summaries only.
for file in \
  SCIENTIFIC_FINDINGS_CN.md \
  results/hccb_p418_sourceflow_partial_physics/summary.json \
  results/hccb_p418_sourceflow_partial_physics/completed_case_physics.csv \
  results/hccb_p418_sourceflow_partial_pressure_correlation/summary.json \
  results/hccb_p418_sourceflow_partial_pressure_correlation/pressure_correlation.csv \
  results/hccb_p418_sourceflow_partial_boundary_heat/summary.json \
  results/hccb_p418_sourceflow_partial_dimensionless_heat_transfer_with_flux/summary.json \
  results/hccb_p418_sourceflow_partial_dimensionless_heat_transfer_with_flux/dimensionless_heat_transfer.csv \
  results/hccb_p418_sourceflow_partial_relations/summary.json \
  results/hccb_p418_sourceflow_partial_relations/P418_14工况物理关系_CN.md \
  results/hccb_p418_sourceflow_partial_relations/hccb_p418_partial_physics_relations.pdf \
  results/hccb_p418_sourceflow_partial_relations/hccb_p418_partial_physics_relations.png \
  results/hccb_p418_local_transport_model_support/summary.json \
  results/hccb_p418_local_transport_model_support/local_transport_model_support.csv \
  results/hccb_p418_local_transport_model_support/P418_局部三维模型数据说明_CN.md \
  results/hccb_p418_local_transport_model_sensitivity/summary.json \
  results/hccb_p418_local_transport_model_sensitivity/P418_局部流场输入敏感性_CN.md \
  results/hccb_p418_60_actual_case_input_check/summary.json \
  results/hccb_p418_training_data_coverage_partial/summary.json \
  results/hccb_heat_ai_external_evidence/summary.json \
  results/hccb_p418_velocity_step_time_scales/summary.json \
  results/hccb_p418_inlet_dimensionless_envelope/inlet_dimensionless_conditions.csv \
  results/hccb_p418_inlet_dimensionless_envelope/summary.json \
  results/hccb_p418_step_split_coverage/summary.json \
  results/hccb_p418_step_split_coverage/curve_coverage.csv \
  results/hccb_p418_step_split_coverage/P418_热阶跃训练测试范围_CN.md \
  results/hccb_p418_high_re_independent_plan/summary.json \
  results/hccb_p418_high_re_independent_plan/P418_高流速独立测试_CN.md \
  results/hccb_p418_actual_spatiotemporal_operator_56time_gpu_factorized/summary.json \
  results/hccb_p418_actual_temporal_diffusion_56time_gpu_batch1_bfloat16_chunk2048/summary.json \
  results/hccb_p418_diffusion_physical_state/summary.json \
  results/hccb_p418_fully_coupled_training_interface/summary.json \
  results/hccb_p418_model_data_preparation/summary.json \
  results/hccb_p418_end_to_end_plan/summary.json \
  results/hccb_p418_fused_preflight/summary.json \
  results/hccb_p418_fused_preflight/P418_融合模型计算前检查_CN.md \
  results/hccb_p418_end_to_end_model_interface/summary.json \
  results/hccb_p418_end_to_end_model_interface/P418_融合模型接口检查_CN.md \
  results/hccb_p418_parameter_evidence/summary.json \
  results/hccb_p418_parameter_evidence/P418_物理参数简明说明_CN.md \
  results/hccb_p418_parameter_use/summary.json \
  results/hccb_p418_parameter_use/P418_参数怎样进入研究_CN.md \
  results/hccb_p418_research_route_completeness/summary.json \
  results/hccb_p418_research_route_completeness/P418_研究方案完成情况_CN.md \
  results/hccb_p418_experimental_data_validation.json \
  results/hccb_p418_experimental_observation_sources/summary.json \
  results/hccb_p418_experimental_observation_sources/P418_实验观测量与模型对应_CN.md; do
  if [[ ! -f "${ROOT}/${file}" ]]; then
    echo "missing scientific-findings input: ${ROOT}/${file}" >&2
    exit 2
  fi
  mkdir -p "${STAGE}/$(dirname "${file}")"
  cp "${ROOT}/${file}" "${STAGE}/${file}"
done

# Keep the three small, source-backed packing realizations required to rebuild
# the 60+9+9 cross-packing plan after the package is unpacked. These are input
# geometries (about 176 kB in total), not generated OpenFOAM fields.
PACKING_INPUT_ROOT=data/apd006_hccb_source_sequence_target_packings
if [[ ! -d "${ROOT}/${PACKING_INPUT_ROOT}" ]]; then
  echo "missing cross-packing input directory: ${ROOT}/${PACKING_INPUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${STAGE}/data"
rsync -rlt \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '._*' \
  --exclude '.DS_Store' \
  "${ROOT}/${PACKING_INPUT_ROOT}" \
  "${STAGE}/data/"

# The staged time-step studies invoke the Celik three-resolution GCI method
# directly. Its saved source record is not a pebble-bed parameter, but it must
# travel with the code so the method check works outside the full repository.
for file in \
  literature/raw/numerical_methods/Celik_2008_GCI_crossref.json; do
  if [[ ! -f "${ROOT}/${file}" ]]; then
    echo "missing numerical-method source file: ${ROOT}/${file}" >&2
    exit 2
  fi
  mkdir -p "${STAGE}/$(dirname "${file}")"
  cp "${ROOT}/${file}" "${STAGE}/${file}"
done

# Copy only the local source files referenced by the 22-parameter evidence
# table. This keeps the CPU package self-contained without carrying the full
# project literature library.
while IFS= read -r relative; do
  [[ -n ${relative} ]] || continue
  source_path=${ROOT}/${relative}
  if [[ ! -f ${source_path} ]]; then
    echo "missing parameter evidence file: ${source_path}" >&2
    exit 2
  fi
  mkdir -p "${STAGE}/$(dirname "${relative}")"
  cp "${source_path}" "${STAGE}/${relative}"
done < <(
  python3 - "${ROOT}/parameters/hccb_p418_physical_parameter_evidence_files.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8-sig") as stream:
    paths = {
        item.strip()
        for row in csv.DictReader(stream)
        for item in row["local_evidence_paths"].split(";")
        if item.strip()
    }
for path in sorted(paths):
    print(path)
PY
)

# The architecture registry stores the main papers and implementations, while
# the numerical-settings table also cites optimizer/configuration sources.
# Include every local source path from that table so the cloud package can
# reproduce the same current numerical-setting check as the main project.
while IFS= read -r relative; do
  [[ -n ${relative} ]] || continue
  source_path=${ROOT}/${relative}
  if [[ ! -f ${source_path} ]]; then
    echo "missing model-setting source file: ${source_path}" >&2
    exit 2
  fi
  mkdir -p "${STAGE}/$(dirname "${relative}")"
  cp "${source_path}" "${STAGE}/${relative}"
done < <(
  python3 - "${ROOT}/parameters/hccb_p418_model_numerical_settings.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8-sig") as stream:
    paths = {
        row["source_path"].strip()
        for row in csv.DictReader(stream)
        if row["source_path"].strip()
    }
for path in sorted(paths):
    print(path)
PY
)

# Copy only the archived algorithm sources and small recorded results that are
# referenced by the architecture registry. This makes the source checks work on
# the cloud without copying the full 621 MB third_party tree.
while IFS= read -r relative; do
  [[ -n ${relative} ]] || continue
  source_path=${ROOT}/${relative}
  if [[ ! -f ${source_path} ]]; then
    echo "missing architecture source file: ${source_path}" >&2
    exit 2
  fi
  mkdir -p "${STAGE}/$(dirname "${relative}")"
  cp "${source_path}" "${STAGE}/${relative}"
done < <(
  python3 - "${ROOT}" \
    "${ROOT}/parameters/hccb_p418_ai_architecture_sources.json" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
registry = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
paths: set[str] = set()


def collect(value: object) -> None:
    if isinstance(value, dict):
        for item in value.values():
            collect(item)
        return
    if isinstance(value, list):
        for item in value:
            collect(item)
        return
    if not isinstance(value, str):
        return
    if value.startswith(("http://", "https://")) or len(value) >= 300:
        return
    if "/" not in value:
        return
    candidate = pathlib.Path(value)
    if candidate.is_absolute():
        return
    source = (root / candidate).resolve()
    try:
        source.relative_to(root)
    except ValueError:
        return
    try:
        if source.is_file():
            paths.add(candidate.as_posix())
    except OSError:
        return


collect(registry)
for path in sorted(paths):
    print(path)
PY
)

(
  cd "${STAGE}"
  find . -type f ! -name SHA256SUMS -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 shasum -a 256 > SHA256SUMS
)

rm -f "${ARCHIVE}" "${ARCHIVE}.sha256"
if command -v zstd >/dev/null 2>&1; then
  COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata \
    -cf - -C "${TMP_ROOT}" "${PACKAGE_NAME}" \
    | zstd -T0 -10 -o "${ARCHIVE}"
else
  echo "zstd is required to build ${ARCHIVE}" >&2
  exit 2
fi

(
  cd "${OUTPUT_ROOT}"
  shasum -a 256 "$(basename "${ARCHIVE}")" > "$(basename "${ARCHIVE}").sha256"
)
printf 'archive=%s\n' "${ARCHIVE}"
printf 'bytes=%s\n' "$(wc -c < "${ARCHIVE}")"
cat "${ARCHIVE}.sha256"
