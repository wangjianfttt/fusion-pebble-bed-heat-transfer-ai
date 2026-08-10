#!/usr/bin/env bash
# Evaluate trained seed101 models on seed202 or one fixed model on seed303.
# Default is a dry run and therefore reads no independent packing field.

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
STAGE=${STAGE:-development}
EXECUTE=${EXECUTE:-0}
SPLIT_NAME=${SPLIT_NAME:-interleaved_all_ranges}
DEVICE=${DEVICE:-cuda}
MICROBATCH_SIZE=${MICROBATCH_SIZE:-1}
PROTOCOL=${PROTOCOL:-${ROOT}/parameters/hccb_p418_cross_packing_model_protocol.json}
RESULT_ROOT=${RESULT_ROOT:-${ROOT}/results}
MODEL_SOURCES=${MODEL_SOURCES:-${RESULT_ROOT}/hccb_p418_cross_packing_seed101_model_sources.json}
SELECTION_FILE=${SELECTION_FILE:-${RESULT_ROOT}/hccb_p418_cross_packing_seed202_model_comparison/architecture_selection.json}

if [[ ${STAGE} != development && ${STAGE} != final ]]; then
    echo "STAGE must be development or final" >&2
    exit 1
fi
if [[ ${EXECUTE} != 0 && ${EXECUTE} != 1 ]]; then
    echo "EXECUTE must be 0 or 1" >&2
    exit 1
fi

if [[ ${STAGE} == development ]]; then
    seed=202
    architectures=(pinn_data_only pinn graph transolver)
else
    seed=303
    if [[ ! -f ${SELECTION_FILE} ]]; then
        echo "final stage requires the seed202 selection file: ${SELECTION_FILE}" >&2
        exit 1
    fi
    fixed_architecture=$(python3 - "${SELECTION_FILE}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") != "seed202_architecture_fixed_before_seed303":
    raise SystemExit("seed202 architecture selection is not complete")
if payload.get("seed303_fields_read") is not False:
    raise SystemExit("selection file does not prove that seed303 was unseen")
print(payload["selected_architecture"])
PY
    )
    if [[ -n ${SELECTED_ARCHITECTURE:-} && ${SELECTED_ARCHITECTURE} != "${fixed_architecture}" ]]; then
        echo "SELECTED_ARCHITECTURE disagrees with frozen seed202 selection" >&2
        exit 1
    fi
    SELECTED_ARCHITECTURE=${fixed_architecture}
    case ${SELECTED_ARCHITECTURE} in
        pinn_data_only|pinn|graph|transolver) ;;
        *) echo "unsupported SELECTED_ARCHITECTURE=${SELECTED_ARCHITECTURE}" >&2; exit 1 ;;
    esac
    architectures=("${SELECTED_ARCHITECTURE}")
fi

echo "cross-packing model stage: ${STAGE}, seed${seed}"
echo "  split: ${SPLIT_NAME}"
echo "  models: ${architectures[*]}"
echo "  seed101 normalization and validation-selected checkpoints remain fixed"

if [[ ${EXECUTE} == 0 ]]; then
    echo "dry run only: no independent packing field was loaded"
    exit 0
fi

if [[ ! -f ${MODEL_SOURCES} ]]; then
    echo "validation-selected seed101 model sources are missing: ${MODEL_SOURCES}" >&2
    exit 1
fi

if [[ ${STAGE} == final ]]; then
    development_result=${RESULT_ROOT}/hccb_p418_cross_packing_seed202_${SELECTED_ARCHITECTURE}_${SPLIT_NAME}.json
    python3 "${ROOT}/code/verify_hccb_p418_cross_packing_fixed_model.py" \
        --selection "${SELECTION_FILE}" \
        --model-sources "${MODEL_SOURCES}" \
        --project-root "${ROOT}" \
        --architecture "${SELECTED_ARCHITECTURE}" \
        --seed202-result "${development_result}" \
        --output "${RESULT_ROOT}/hccb_p418_seed303_fixed_model_check.json"
fi

for architecture in "${architectures[@]}"; do
    mapfile -t source_files < <(
        python3 - "${MODEL_SOURCES}" "${ROOT}" "${architecture}" "${SPLIT_NAME}" <<'PY'
import json
import hashlib
import pathlib
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") != "cross_packing_seed101_model_sources_selected":
    raise SystemExit("seed101 model-source map is not ready")
if payload.get("split_name") != sys.argv[4]:
    raise SystemExit("model-source map uses a different condition split")
if payload.get("independent_test_used_for_selection") is not False:
    raise SystemExit("model-source selection used independent test conditions")
record = payload.get("models", {}).get(sys.argv[3])
if not record:
    raise SystemExit(f"model-source map lacks {sys.argv[3]}")
root = pathlib.Path(sys.argv[2])
for key, hash_key in (
    ("selected_checkpoint", "selected_checkpoint_sha256"),
    ("selected_summary", "selected_summary_sha256"),
):
    path = pathlib.Path(record[key])
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        raise SystemExit(f"validation-selected file is missing: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != record.get(hash_key):
        raise SystemExit(f"validation-selected file changed after source selection: {path}")
    print(path)
print(record["selected_epochs"])
PY
    )
    if [[ ${#source_files[@]} -ne 3 ]]; then
        echo "failed to read validation-selected files for ${architecture}" >&2
        exit 1
    fi
    checkpoint=${source_files[0]}
    training_summary=${source_files[1]}
    source_epochs=${source_files[2]}
    output=${RESULT_ROOT}/hccb_p418_cross_packing_seed${seed}_${architecture}_${SPLIT_NAME}.json
    if [[ ! -f ${checkpoint} || ! -f ${training_summary} ]]; then
        echo "trained seed101 ${architecture} files are missing" >&2
        exit 1
    fi
    echo "  ${architecture}: validation-selected ${source_epochs}-epoch run"
    if [[ -e ${output} ]]; then
        echo "refusing to replace existing first-pass result: ${output}" >&2
        exit 1
    fi
    python3 "${ROOT}/code/evaluate_hccb_p418_cross_packing_conservative_operator.py" \
        --project-root "${ROOT}" \
        --protocol "${PROTOCOL}" \
        --packing-seed "${seed}" \
        --checkpoint "${checkpoint}" \
        --training-summary "${training_summary}" \
        --architecture "${architecture}" \
        --microbatch-size "${MICROBATCH_SIZE}" \
        --device "${DEVICE}" \
        --output "${output}"
done

summary_inputs=()
if [[ ${STAGE} == development ]]; then
    for architecture in "${architectures[@]}"; do
        summary_inputs+=(
            "${RESULT_ROOT}/hccb_p418_cross_packing_seed202_${architecture}_${SPLIT_NAME}.json"
        )
    done
    summary_dir=${RESULT_ROOT}/hccb_p418_cross_packing_seed202_model_comparison
else
    development=${RESULT_ROOT}/hccb_p418_cross_packing_seed202_${SELECTED_ARCHITECTURE}_${SPLIT_NAME}.json
    final=${RESULT_ROOT}/hccb_p418_cross_packing_seed303_${SELECTED_ARCHITECTURE}_${SPLIT_NAME}.json
    if [[ ! -f ${development} ]]; then
        echo "matching seed202 result is missing: ${development}" >&2
        exit 1
    fi
    summary_inputs=("${development}" "${final}")
    summary_dir=${RESULT_ROOT}/hccb_p418_cross_packing_seed303_final_${SELECTED_ARCHITECTURE}
fi

python3 "${ROOT}/code/summarize_hccb_p418_cross_packing_models.py" \
    --input "${summary_inputs[@]}" \
    --output-dir "${summary_dir}"

if [[ ${STAGE} == development ]]; then
    python3 "${ROOT}/code/select_hccb_p418_cross_packing_architecture.py" \
        --summary "${summary_dir}/summary.json" \
        --model-sources "${MODEL_SOURCES}" \
        --output "${summary_dir}/architecture_selection.json" \
        --chinese-output "${summary_dir}/P418_seed202模型选择_CN.md"
else
    development_summary=${RESULT_ROOT}/hccb_p418_cross_packing_seed202_model_comparison/summary.json
    python3 "${ROOT}/code/plot_hccb_p418_cross_packing_results.py" \
        --development-summary "${development_summary}" \
        --selection "${SELECTION_FILE}" \
        --final-summary "${summary_dir}/summary.json" \
        --output-dir "${ROOT}/figures"
    python3 "${ROOT}/code/build_hccb_p418_cross_packing_result_text.py" \
        --development-summary "${development_summary}" \
        --selection "${SELECTION_FILE}" \
        --final-summary "${summary_dir}/summary.json" \
        --output "${ROOT}/manuscript/generated_cross_packing_result_text.tex" \
        --summary "${summary_dir}/manuscript_text.json"
fi

echo "cross-packing ${STAGE} model results are complete: ${summary_dir}"
