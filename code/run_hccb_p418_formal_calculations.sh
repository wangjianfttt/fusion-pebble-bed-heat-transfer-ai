#!/usr/bin/env bash
# Complete the declared seed101, seed202 and seed303 calculation sequence.
# The default is a dry run. EXECUTE=1 starts expensive calculations.

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
EXECUTE=${EXECUTE:-0}
NP_PER_CASE=${NP_PER_CASE:-32}
CONCURRENT_CASES=${CONCURRENT_CASES:-3}
RESULT_ROOT=${RESULT_ROOT:-${ROOT}/results}
FORMAL_LOCK=${FORMAL_LOCK:-${RESULT_ROOT}/hccb_p418_formal_calculations.lock}
MODEL_SOURCES=${MODEL_SOURCES:-${RESULT_ROOT}/hccb_p418_cross_packing_seed101_model_sources.json}
SELECTION_FILE=${SELECTION_FILE:-${RESULT_ROOT}/hccb_p418_cross_packing_seed202_model_comparison/architecture_selection.json}
P418_PYTHON=${P418_PYTHON:-python3}
OPENFOAM_BASHRC=${OPENFOAM_BASHRC:-/opt/openfoam13/etc/bashrc}

if [[ ${EXECUTE} != 0 && ${EXECUTE} != 1 ]]; then
    echo "EXECUTE must be 0 or 1" >&2
    exit 1
fi

echo "P418 formal heat-transfer calculation sequence"
echo "  1. seed101: 60 steady conditions and 12 fixed-flow thermal steps"
echo "  2. fully coupled time-step study, 12 trajectories and frozen model"
echo "  3. seed202: nine independent-packing conditions and model comparison"
echo "  4. freeze architecture using seed202 only"
echo "  5. seed303: nine zero-shot conditions with the frozen model"
echo "  6. six fixed-flow and six fully coupled high-Re histories"
echo "  7. source-sized domain comparison and manuscript refresh"

if [[ ${EXECUTE} == 0 ]]; then
    if [[ -f ${ROOT}/code/plan_hccb_p418_end_to_end_research.py ]]; then
        "${P418_PYTHON}" "${ROOT}/code/plan_hccb_p418_end_to_end_research.py" \
            --project-root "${ROOT}"
    fi
    echo "dry run only: no mesh, OpenFOAM solver or model training was started"
    exit 0
fi

mkdir -p "${RESULT_ROOT}"
exec 7>"${FORMAL_LOCK}"
if ! flock -n 7; then
    echo "another P418 formal calculation route already holds ${FORMAL_LOCK}" >&2
    exit 1
fi

if ! resolved_python=$(command -v "${P418_PYTHON}"); then
    echo "P418 Python executable not found: ${P418_PYTHON}" >&2
    exit 1
fi
P418_PYTHON=${resolved_python}
export P418_PYTHON
# Nested calculation scripts call `python3`; put the selected scientific
# environment first so they all use the same NumPy/PyTorch installation.
export PATH="$(dirname "${P418_PYTHON}"):${PATH}"
"${P418_PYTHON}" - <<'PY'
import numpy
import pandas
import scipy
import sklearn
import torch

if not torch.cuda.is_available():
    raise SystemExit("the formal P418 model route requires an available CUDA device")
print(f"formal P418 Python: {torch.__version__}; CUDA available")
PY
if [[ -f ${OPENFOAM_BASHRC} ]]; then
    # shellcheck disable=SC1090
    source "${OPENFOAM_BASHRC}"
    export PATH="$(dirname "${P418_PYTHON}"):${PATH}"
fi

# Rebuild the calculation list from the current literature-backed description
# before any expensive stage starts.  This prevents an older 20x3 packing plan
# or the retired P051--P055 conditions from being reintroduced by a stale file.
mkdir -p "${RESULT_ROOT}/hccb_p418_formal_route_checks"
RUNTIME_ENVIRONMENT=${RESULT_ROOT}/hccb_p418_formal_route_checks/runtime_environment.json
python3 "${ROOT}/code/capture_hccb_p418_runtime_environment.py" \
    --output "${RUNTIME_ENVIRONMENT}" \
    > "${RESULT_ROOT}/hccb_p418_formal_route_checks/runtime_environment.log"
python3 "${ROOT}/code/build_hccb_p418_cross_packing_plan.py" \
    > "${RESULT_ROOT}/hccb_p418_formal_route_checks/cross_packing_plan.json"
python3 "${ROOT}/code/build_hccb_pore_resolved_cht_case_matrix.py" \
    > "${RESULT_ROOT}/hccb_p418_formal_route_checks/case_plan.json"
python3 "${ROOT}/code/audit_hccb_pore_resolved_cht_case_matrix.py" \
    > "${RESULT_ROOT}/hccb_p418_formal_route_checks/case_plan_check.json"
python3 "${ROOT}/code/audit_hccb_pore_resolved_cht_dataset_contract.py" \
    > "${RESULT_ROOT}/hccb_p418_formal_route_checks/data_description_check.json"
python3 "${ROOT}/code/audit_hccb_pore_resolved_ml_tensor_contract.py" \
    > "${RESULT_ROOT}/hccb_p418_formal_route_checks/model_description_check.json"
echo "confirmed current calculation route: seed101 60 + seed202 9 + seed303 9"

completed=$(find "${ROOT}/hccb_dense_cht_p418_60_sourceflow_r3" -mindepth 2 -maxdepth 2 \
    -name formal_sample_complete.json | wc -l | tr -d ' ')
if [[ ${completed} -ne 60 ]]; then
    echo "seed101 must contain 60 completed steady cases; found ${completed}" >&2
    exit 1
fi

POSTSTEADY_RECORD=${RESULT_ROOT}/hccb_p418_poststeady_pipeline_complete.json
POSTSTEADY_CHECK=${RESULT_ROOT}/hccb_p418_poststeady_pipeline_current.json
poststeady_current=0
if [[ -f ${POSTSTEADY_RECORD} ]]; then
    if python3 "${ROOT}/code/verify_hccb_p418_poststeady_completion.py" \
        --record "${POSTSTEADY_RECORD}" --output "${POSTSTEADY_CHECK}" \
        > /dev/null; then
        poststeady_current=1
    else
        echo "existing post-steady record is incomplete or refers to changed results; continuing calculations"
    fi
fi
if [[ ${poststeady_current} -eq 0 ]]; then
    ROOT="${ROOT}" bash "${ROOT}/code/run_hccb_p418_poststeady_pipeline.sh"
    python3 "${ROOT}/code/verify_hccb_p418_poststeady_completion.py" \
        --record "${POSTSTEADY_RECORD}" --output "${POSTSTEADY_CHECK}" \
        > /dev/null
else
    echo "reuse completed seed101 post-steady calculations"
fi

FULL_TIMESTEP_SUMMARY=${RESULT_ROOT}/hccb_p418_fully_coupled_timestep_sensitivity/fully_coupled_timestep_sensitivity.json
if [[ ! -f ${FULL_TIMESTEP_SUMMARY} ]]; then
    ROOT="${ROOT}" EXECUTE=1 NP_PER_CASE="${NP_PER_CASE}" \
        bash "${ROOT}/code/run_hccb_p418_fully_coupled_timestep_sensitivity.sh"
else
    echo "reuse completed fully coupled time-step comparison"
fi

FULL_STEP_DATASET=${RESULT_ROOT}/hccb_p418_fully_coupled_steps_12/regional_sequences/dataset_index.json
if [[ ! -f ${FULL_STEP_DATASET} ]]; then
    ROOT="${ROOT}" EXECUTE=1 NP_PER_CASE="${NP_PER_CASE}" \
        CONCURRENT_CASES=1 \
        bash "${ROOT}/code/run_hccb_p418_fully_coupled_step_responses.sh"
else
    echo "reuse completed fully coupled thermal trajectories"
fi

FIXED_STEP_OBSERVABLES=${RESULT_ROOT}/hccb_p418_physical_steps_12/hccb_p418_transient_observables.npz
FULL_STEP_OBSERVABLES=${RESULT_ROOT}/hccb_p418_fully_coupled_steps_12/hccb_p418_transient_observables.npz
FIXED_FULL_COMPARISON_DIR=${RESULT_ROOT}/hccb_p418_fully_coupled_steps_12/fixed_vs_fully_coupled
FIXED_FULL_COMPARISON=${FIXED_FULL_COMPARISON_DIR}/summary.json
FIXED_FULL_CSV=${FIXED_FULL_COMPARISON_DIR}/fixed_vs_fully_coupled_steps.csv
FIXED_FULL_TABLE=${ROOT}/manuscript/generated_fixed_vs_fully_coupled_steps.tex
FIXED_FULL_TEXT=${ROOT}/manuscript/generated_fixed_vs_fully_coupled_text.tex
if [[ ! -f ${FIXED_FULL_COMPARISON} || ! -f ${FIXED_FULL_CSV} ]]; then
    python3 "${ROOT}/code/compare_hccb_p418_fixed_and_fully_coupled_steps.py" \
        --fixed-observables "${FIXED_STEP_OBSERVABLES}" \
        --fully-coupled-observables "${FULL_STEP_OBSERVABLES}" \
        --output-dir "${FIXED_FULL_COMPARISON_DIR}"
fi
python3 "${ROOT}/code/build_hccb_p418_fixed_vs_fully_coupled_manuscript.py" \
    --summary "${FIXED_FULL_COMPARISON}" \
    --csv "${FIXED_FULL_CSV}" \
    --table-output "${FIXED_FULL_TABLE}" \
    --text-output "${FIXED_FULL_TEXT}"

FULL_MODEL_ROOT=${RESULT_ROOT}/hccb_p418_fully_coupled_model_comparison
FULL_MODEL_FINAL=$(find "${FULL_MODEL_ROOT}" -mindepth 2 -maxdepth 2 \
    -name final_summary.json -print -quit 2>/dev/null || true)
if [[ -z ${FULL_MODEL_FINAL} ]]; then
    ROOT="${ROOT}" EXECUTE=1 DEVICE=cuda \
        bash "${ROOT}/code/run_hccb_p418_fully_coupled_model_stage.sh"
    FULL_MODEL_FINAL=$(find "${FULL_MODEL_ROOT}" -mindepth 2 -maxdepth 2 \
        -name final_summary.json -print -quit)
else
    echo "reuse completed fully coupled frozen model"
fi
if [[ -z ${FULL_MODEL_FINAL} || ! -f ${FULL_MODEL_FINAL} ]]; then
    echo "fully coupled final model is missing" >&2
    exit 1
fi

if [[ ! -f ${MODEL_SOURCES} ]]; then
    python3 "${ROOT}/code/build_hccb_p418_cross_packing_model_sources.py" \
        --project-root "${ROOT}" \
        --initial-epochs 100 \
        --split-name interleaved_all_ranges \
        --followup-plan "${RESULT_ROOT}/hccb_p418_60_source_epoch_followup/epoch_followup_plan.json" \
        --output "${MODEL_SOURCES}"
fi
python3 - "${MODEL_SOURCES}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") != "cross_packing_seed101_model_sources_selected":
    raise SystemExit("seed101 model sources are not ready for cross-packing evaluation")
if payload.get("independent_test_used_for_selection") is not False:
    raise SystemExit("seed101 model sources used independent test conditions")
PY

seed202_manifest=${ROOT}/hccb_dense_cht_p418_cross_packing_seed202_screen9/matrix_manifest.json
seed303_manifest=${ROOT}/hccb_dense_cht_p418_cross_packing_seed303_screen9/matrix_manifest.json
if [[ ! -f ${seed202_manifest} || ! -f ${seed303_manifest} ]]; then
    ROOT="${ROOT}" EXECUTE=1 bash "${ROOT}/code/run_hccb_p418_cross_packing_setup.sh"
else
    echo "reuse prepared seed202 and seed303 meshes and case directories"
fi

# Regenerate the geometry comparison even when both meshes were prepared by an
# earlier run. The values come from the actual packing and local mesh records,
# not from the numerical seed labels.
CROSS_PACKING_GEOMETRY_DIR=${RESULT_ROOT}/hccb_p418_cross_packing_geometry
CROSS_PACKING_GEOMETRY_SUMMARY=${CROSS_PACKING_GEOMETRY_DIR}/summary.json
CROSS_PACKING_GEOMETRY_TABLE=${ROOT}/manuscript/generated_cross_packing_geometry.tex
geometry_args=()
physical_csv=${RESULT_ROOT}/hccb_p418_60_sourceflow_r3_completed_physics/completed_case_physics.csv
if [[ -f ${physical_csv} ]]; then
    geometry_args+=(--physical-csv "${physical_csv}")
fi
python3 "${ROOT}/code/summarize_hccb_p418_cross_packing_geometry.py" \
    --root "${ROOT}" \
    --plan "${ROOT}/parameters/hccb_p418_cross_packing_plan.json" \
    --manifest-seed101 "${ROOT}/hccb_dense_snappy_g2_nativezone_r2/case_manifest.json" \
    --manifest-seed202 "${ROOT}/hccb_dense_snappy_g2_nativezone_r2_seed202/case_manifest.json" \
    --manifest-seed303 "${ROOT}/hccb_dense_snappy_g2_nativezone_r2_seed303/case_manifest.json" \
    --output-dir "${CROSS_PACKING_GEOMETRY_DIR}" \
    --tex-output "${CROSS_PACKING_GEOMETRY_TABLE}" \
    "${geometry_args[@]}"

if [[ ! -f ${RESULT_ROOT}/hccb_p418_cross_packing_seed202_complete.json ]]; then
    ROOT="${ROOT}" SEED=202 EXECUTE=1 NP_PER_CASE="${NP_PER_CASE}" \
    CONCURRENT_CASES="${CONCURRENT_CASES}" \
        bash "${ROOT}/code/run_hccb_p418_cross_packing_matrix.sh"
else
    echo "reuse completed seed202 OpenFOAM matrix"
fi

if [[ ! -f ${SELECTION_FILE} ]]; then
    ROOT="${ROOT}" RESULT_ROOT="${RESULT_ROOT}" STAGE=development EXECUTE=1 \
    MODEL_SOURCES="${MODEL_SOURCES}" \
        bash "${ROOT}/code/run_hccb_p418_cross_packing_model_stage.sh"
else
    echo "reuse architecture fixed from seed202"
fi

if [[ ! -f ${RESULT_ROOT}/hccb_p418_cross_packing_seed303_complete.json ]]; then
    ROOT="${ROOT}" SEED=303 EXECUTE=1 NP_PER_CASE="${NP_PER_CASE}" \
    CONCURRENT_CASES="${CONCURRENT_CASES}" \
        bash "${ROOT}/code/run_hccb_p418_cross_packing_matrix.sh"
else
    echo "reuse completed seed303 OpenFOAM matrix"
fi

selected_architecture=$(python3 - "${SELECTION_FILE}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") != "seed202_architecture_fixed_before_seed303":
    raise SystemExit("seed202 architecture selection is not complete")
print(payload["selected_architecture"])
PY
)
final_summary=${RESULT_ROOT}/hccb_p418_cross_packing_seed303_final_${selected_architecture}/summary.json
if [[ ! -f ${final_summary} ]]; then
    ROOT="${ROOT}" RESULT_ROOT="${RESULT_ROOT}" STAGE=final EXECUTE=1 \
    MODEL_SOURCES="${MODEL_SOURCES}" SELECTION_FILE="${SELECTION_FILE}" \
        bash "${ROOT}/code/run_hccb_p418_cross_packing_model_stage.sh"
else
    echo "reuse completed frozen-model seed303 prediction"
fi
CROSS_PACKING_TEXT=${ROOT}/manuscript/generated_cross_packing_result_text.tex
CROSS_PACKING_TEXT_SUMMARY=${RESULT_ROOT}/hccb_p418_cross_packing_seed303_final_${selected_architecture}/manuscript_text.json

HIGH_RE_FIXED_SUMMARY=${RESULT_ROOT}/hccb_p418_high_re_independent_fixed_model_evaluation/summary.json
if [[ ! -f ${HIGH_RE_FIXED_SUMMARY} ]]; then
    ROOT="${ROOT}" MODE=fixed EXECUTE=1 NP_PER_CASE="${NP_PER_CASE}" \
        bash "${ROOT}/code/run_hccb_p418_high_re_independent_steps.sh"
else
    echo "reuse completed fixed-flow high-Re reference and frozen prediction"
fi

HIGH_RE_FULL_SUMMARY=${RESULT_ROOT}/hccb_p418_high_re_independent_fully_coupled_model_evaluation/summary.json
HIGH_RE_COMPARISON=${RESULT_ROOT}/hccb_p418_high_re_independent_model_comparison/summary.json
if [[ ! -f ${HIGH_RE_FULL_SUMMARY} || ! -f ${HIGH_RE_COMPARISON} ]]; then
    ROOT="${ROOT}" MODE=fully_coupled EXECUTE=1 NP_PER_CASE="${NP_PER_CASE}" \
        bash "${ROOT}/code/run_hccb_p418_high_re_independent_steps.sh"
else
    echo "reuse completed fully coupled high-Re reference and joint comparison"
fi
for required in "${HIGH_RE_FIXED_SUMMARY}" "${HIGH_RE_FULL_SUMMARY}" \
    "${HIGH_RE_COMPARISON}" \
    "${ROOT}/manuscript/generated_high_re_comparison.tex"; do
    if [[ ! -f ${required} ]]; then
        echo "high-Re frozen-model comparison is missing: ${required}" >&2
        exit 1
    fi
done

# The source-sized CHT calculation starts automatically after the 12 thermal
# steps and normally runs while the GPU models are being trained.  This call
# acquires the same run lock, so it either reuses the completed comparison or
# waits for the background calculation before the final manuscript is built.
FULL_DOMAIN_RESULT_DIR=${RESULT_ROOT}/hccb_p418_full_domain_reference
FULL_DOMAIN_COMPLETION=${FULL_DOMAIN_RESULT_DIR}/completion.json
FULL_DOMAIN_COMPARISON=${FULL_DOMAIN_RESULT_DIR}/full_vs_local_comparison.json
FULL_DOMAIN_TABLE=${ROOT}/manuscript/generated_full_domain_reference.tex
ROOT="${ROOT}" RESULT_DIR="${FULL_DOMAIN_RESULT_DIR}" \
    MANUSCRIPT_TABLE="${FULL_DOMAIN_TABLE}" \
    bash "${ROOT}/code/run_hccb_p418_full_domain_reference.sh"
for required in "${FULL_DOMAIN_COMPLETION}" "${FULL_DOMAIN_COMPARISON}" \
    "${FULL_DOMAIN_TABLE}"; do
    if [[ ! -f ${required} ]]; then
        echo "source-sized domain comparison is missing: ${required}" >&2
        exit 1
    fi
done

python3 - "${RESULT_ROOT}/hccb_p418_formal_calculations_complete.json" \
    "${MODEL_SOURCES}" "${SELECTION_FILE}" "${final_summary}" \
    "${CROSS_PACKING_GEOMETRY_SUMMARY}" "${CROSS_PACKING_GEOMETRY_TABLE}" \
    "${CROSS_PACKING_TEXT}" "${CROSS_PACKING_TEXT_SUMMARY}" \
    "${RUNTIME_ENVIRONMENT}" "${FULL_DOMAIN_COMPLETION}" \
    "${FULL_DOMAIN_COMPARISON}" "${FULL_DOMAIN_TABLE}" \
    "${FULL_TIMESTEP_SUMMARY}" "${FULL_STEP_DATASET}" "${FULL_MODEL_FINAL}" \
    "${FIXED_FULL_COMPARISON}" "${FIXED_FULL_CSV}" \
    "${FIXED_FULL_TABLE}" "${FIXED_FULL_TEXT}" \
    "${HIGH_RE_FIXED_SUMMARY}" "${HIGH_RE_FULL_SUMMARY}" \
    "${HIGH_RE_COMPARISON}" \
    "${ROOT}/manuscript/generated_high_re_comparison.tex" <<'PY'
import hashlib
import json
import pathlib
import sys

output, model_sources, selection, final, geometry_summary, geometry_table, cross_text, cross_text_summary, runtime_environment, full_domain_completion, full_domain_comparison, full_domain_table, full_timestep_summary, full_step_dataset, full_model_final, fixed_full_comparison, fixed_full_csv, fixed_full_table, fixed_full_text, high_re_fixed_summary, high_re_full_summary, high_re_comparison, high_re_table = map(
    pathlib.Path, sys.argv[1:]
)
for path in (
    model_sources,
    selection,
    final,
    geometry_summary,
    geometry_table,
    cross_text,
    cross_text_summary,
    runtime_environment,
    full_domain_completion,
    full_domain_comparison,
    full_domain_table,
    full_timestep_summary,
    full_step_dataset,
    full_model_final,
    fixed_full_comparison,
    fixed_full_csv,
    fixed_full_table,
    fixed_full_text,
    high_re_fixed_summary,
    high_re_full_summary,
    high_re_comparison,
    high_re_table,
):
    if not path.is_file():
        raise SystemExit(f"missing final calculation record: {path}")
selected = json.loads(selection.read_text(encoding="utf-8"))
output.write_text(
    json.dumps(
        {
            "status": "p418_seed101_seed202_seed303_formal_calculations_complete",
            "selected_architecture": selected["selected_architecture"],
            "seed101_model_sources": str(model_sources.resolve()),
            "seed101_model_sources_sha256": hashlib.sha256(model_sources.read_bytes()).hexdigest(),
            "selection_file": str(selection.resolve()),
            "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
            "final_seed303_summary": str(final.resolve()),
            "final_seed303_summary_sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
            "cross_packing_geometry_summary": str(geometry_summary.resolve()),
            "cross_packing_geometry_summary_sha256": hashlib.sha256(geometry_summary.read_bytes()).hexdigest(),
            "cross_packing_geometry_table": str(geometry_table.resolve()),
            "cross_packing_geometry_table_sha256": hashlib.sha256(geometry_table.read_bytes()).hexdigest(),
            "cross_packing_manuscript_text": str(cross_text.resolve()),
            "cross_packing_manuscript_text_sha256": hashlib.sha256(cross_text.read_bytes()).hexdigest(),
            "cross_packing_manuscript_text_summary": str(cross_text_summary.resolve()),
            "cross_packing_manuscript_text_summary_sha256": hashlib.sha256(cross_text_summary.read_bytes()).hexdigest(),
            "runtime_environment": str(runtime_environment.resolve()),
            "runtime_environment_sha256": hashlib.sha256(runtime_environment.read_bytes()).hexdigest(),
            "full_domain_completion": str(full_domain_completion.resolve()),
            "full_domain_completion_sha256": hashlib.sha256(full_domain_completion.read_bytes()).hexdigest(),
            "full_domain_comparison": str(full_domain_comparison.resolve()),
            "full_domain_comparison_sha256": hashlib.sha256(full_domain_comparison.read_bytes()).hexdigest(),
            "full_domain_manuscript_table": str(full_domain_table.resolve()),
            "full_domain_manuscript_table_sha256": hashlib.sha256(full_domain_table.read_bytes()).hexdigest(),
            "fully_coupled_timestep_summary": str(full_timestep_summary.resolve()),
            "fully_coupled_timestep_summary_sha256": hashlib.sha256(full_timestep_summary.read_bytes()).hexdigest(),
            "fully_coupled_dataset_index": str(full_step_dataset.resolve()),
            "fully_coupled_dataset_index_sha256": hashlib.sha256(full_step_dataset.read_bytes()).hexdigest(),
            "fully_coupled_final_model_summary": str(full_model_final.resolve()),
            "fully_coupled_final_model_summary_sha256": hashlib.sha256(full_model_final.read_bytes()).hexdigest(),
            "fixed_vs_fully_coupled_comparison": str(fixed_full_comparison.resolve()),
            "fixed_vs_fully_coupled_comparison_sha256": hashlib.sha256(fixed_full_comparison.read_bytes()).hexdigest(),
            "fixed_vs_fully_coupled_csv": str(fixed_full_csv.resolve()),
            "fixed_vs_fully_coupled_csv_sha256": hashlib.sha256(fixed_full_csv.read_bytes()).hexdigest(),
            "fixed_vs_fully_coupled_manuscript_table": str(fixed_full_table.resolve()),
            "fixed_vs_fully_coupled_manuscript_table_sha256": hashlib.sha256(fixed_full_table.read_bytes()).hexdigest(),
            "fixed_vs_fully_coupled_manuscript_text": str(fixed_full_text.resolve()),
            "fixed_vs_fully_coupled_manuscript_text_sha256": hashlib.sha256(fixed_full_text.read_bytes()).hexdigest(),
            "high_re_fixed_frozen_summary": str(high_re_fixed_summary.resolve()),
            "high_re_fixed_frozen_summary_sha256": hashlib.sha256(high_re_fixed_summary.read_bytes()).hexdigest(),
            "high_re_fully_coupled_frozen_summary": str(high_re_full_summary.resolve()),
            "high_re_fully_coupled_frozen_summary_sha256": hashlib.sha256(high_re_full_summary.read_bytes()).hexdigest(),
            "high_re_joint_comparison": str(high_re_comparison.resolve()),
            "high_re_joint_comparison_sha256": hashlib.sha256(high_re_comparison.read_bytes()).hexdigest(),
            "high_re_manuscript_table": str(high_re_table.resolve()),
            "high_re_manuscript_table_sha256": hashlib.sha256(high_re_table.read_bytes()).hexdigest(),
            "new_physical_parameter_values_added": [],
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(output)
PY

ROOT="${ROOT}" RESULT_ROOT="${RESULT_ROOT}" \
    bash "${ROOT}/code/run_hccb_p418_manuscript_refresh.sh"

echo "P418 formal heat-transfer calculations are complete"
