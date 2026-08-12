#!/usr/bin/env bash
# Run the self-contained scientific checks shipped in the public source package.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON:-python3}"

tests=(
  tests/test_analyze_hccb_p418_fully_coupled_failure_scale.py
  tests/test_analyze_hccb_p418_dimensionless_heat_transfer.py
  tests/test_analyze_hccb_p418_fixed_flow_loss_scale.py
  tests/test_analyze_hccb_p418_pressure_correlation.py
  tests/test_build_hccb_p418_reproducibility_manifest.py
  tests/test_build_hccb_p418_steady_result_text.py
  tests/test_build_hccb_p418_scope_limit_text.py
  tests/test_hccb_p418_steady_seed_robustness.py
  tests/test_hccb_p418_physical_parameter_sources.py
  tests/test_hccb_p418_model_splits.py
  tests/test_hccb_p418_transient_step_plan.py
  tests/test_hccb_p418_repository_metadata_consistency.py
  tests/test_hccb_p418_public_record_paths.py
  tests/test_hccb_p418_comparison_contract.py
  tests/test_verify_hccb_p418_transient_thermo_correspondence.py
  tests/test_hccb_p418_pressure_density_consistency.py
)

for test_file in "${tests[@]}"; do
  if [[ ! -f "${test_file}" ]]; then
    printf 'Missing public test: %s\n' "${test_file}" >&2
    exit 1
  fi
done

"${PYTHON_BIN}" -m pytest -q "${tests[@]}"
