#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RESULTS=${ROOT}/results
STAGE_PREFIX=hccb_p418_60_seedrobust_v2
SPLIT_NAME=interleaved_all_ranges
EPOCHS=100
SEEDS=(20260717 20260718 20260719)
THREADS=${THREADS:-8}

STATE_TARGETS=${RESULTS}/hccb_p418_60_sourceflow_r3_regional_state_targets/regional_state_targets.npz
MASS_TARGETS=${RESULTS}/hccb_p418_60_sourceflow_r3_regional_mass_flux_targets/regional_mass_flux_targets.npz
ENERGY_TARGETS=${RESULTS}/hccb_p418_60_sourceflow_r3_regional_energy_flux_targets/regional_energy_flux_targets.npz
REGIONAL_TOPOLOGY=${RESULTS}/hccb_p418_regional_topology_r2/regional_topology.npz
MODEL_GEOMETRY=${RESULTS}/hccb_p418_60_sourceflow_r3_model_geometry/model_geometry.npz
SPLITS=${ROOT}/parameters/hccb_p418_model_splits.json
TRAINING_STATISTICS=${RESULTS}/hccb_p418_60_sourceflow_r3_training_statistics.json
OUTPUT_DIR=${RESULTS}/hccb_p418_60_steady_seed_robustness_100epoch
TEX_OUTPUT=${ROOT}/manuscript/generated_steady_seed_robustness.tex
TEXT_OUTPUT=${ROOT}/manuscript/generated_steady_seed_robustness_text.tex

for path in \
  "${STATE_TARGETS}" "${MASS_TARGETS}" "${ENERGY_TARGETS}" \
  "${REGIONAL_TOPOLOGY}" "${MODEL_GEOMETRY}" "${SPLITS}" \
  "${TRAINING_STATISTICS}"; do
  test -f "${path}" || { echo "missing input: ${path}" >&2; exit 2; }
done

# The three existing repetitions for these architectures were each produced
# with one implementation fingerprint. Keep them read-only and expose them
# under a dedicated robustness namespace.
for architecture in pinn_data_only pinn graph; do
  for seed in "${SEEDS[@]}"; do
    suffix=""
    if [[ ${seed} != 20260717 ]]; then suffix="_seed${seed}"; fi
    source=${RESULTS}/hccb_p418_60_${architecture}_${SPLIT_NAME}_${EPOCHS}epoch${suffix}
    target=${RESULTS}/${STAGE_PREFIX}_${architecture}_${SPLIT_NAME}_${EPOCHS}epoch${suffix}
    test -f "${source}/summary.json" || { echo "missing source: ${source}" >&2; exit 3; }
    if [[ -L ${target} ]]; then
      [[ $(readlink "${target}") == "${source}" ]] || { echo "wrong link: ${target}" >&2; exit 4; }
    elif [[ -e ${target} ]]; then
      echo "refusing to replace existing path: ${target}" >&2
      exit 4
    else
      ln -s "${source}" "${target}"
    fi
  done
done

pids=()
for seed in "${SEEDS[@]}"; do
  suffix=""
  if [[ ${seed} != 20260717 ]]; then suffix="_seed${seed}"; fi
  output=${RESULTS}/${STAGE_PREFIX}_transolver_${SPLIT_NAME}_${EPOCHS}epoch${suffix}
  if [[ -f ${output}/summary.json ]]; then
    echo "reuse completed transolver seed ${seed}"
    continue
  fi
  if [[ -e ${output} ]]; then
    echo "incomplete output already exists: ${output}" >&2
    exit 5
  fi
  mkdir -p "${output}"
  (
    exec python3 "${ROOT}/code/train_hccb_p418_conservative_mixed_operator.py" \
      --regional-topology "${REGIONAL_TOPOLOGY}" \
      --model-geometry "${MODEL_GEOMETRY}" \
      --state-targets "${STATE_TARGETS}" \
      --mass-targets "${MASS_TARGETS}" \
      --energy-targets "${ENERGY_TARGETS}" \
      --split-file "${SPLITS}" \
      --training-statistics "${TRAINING_STATISTICS}" \
      --split-name "${SPLIT_NAME}" \
      --regional-level 5 \
      --architecture transolver \
      --epochs "${EPOCHS}" \
      --threads "${THREADS}" \
      --device cpu \
      --seed "${seed}" \
      --output-dir "${output}"
  ) > "${output}/run.log" 2>&1 &
  pids+=("$!")
  echo "started transolver seed ${seed}: PID $!"
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
[[ ${failed} == 0 ]] || { echo "at least one transolver seed failed" >&2; exit 6; }

python3 "${ROOT}/code/summarize_hccb_p418_steady_seed_robustness.py" \
  --results-root "${RESULTS}" \
  --result-prefix "${STAGE_PREFIX}" \
  --split-file "${SPLITS}" \
  --split-name "${SPLIT_NAME}" \
  --epochs "${EPOCHS}" \
  --primary-seed 20260717 \
  --seeds "${SEEDS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --tex-output "${TEX_OUTPUT}" \
  --text-output "${TEXT_OUTPUT}"

echo "steady three-seed robustness complete"
