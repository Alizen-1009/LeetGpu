#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMPL="${1:-float4}"
SIZE="${2:-16777216}"
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))

OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/ncu-reports}"
REPORT_BASE="${REPORT_BASE:-${OUTPUT_DIR}/reduce_${IMPL}_${SIZE}}"
REPORT_FILE="${REPORT_BASE}.ncu-rep"
TEXT_FILE="${REPORT_BASE}.txt"
WARMUP="${WARMUP:-20}"
PROFILE_ITERS="${PROFILE_ITERS:-1}"
ARCH="${ARCH:-}"
PYTHON_BIN="${PYTHON:-python}"

has_ncu_arg() {
  local expected="$1"
  shift
  for arg in "$@"; do
    if [[ "${arg}" == "${expected}" || "${arg}" == "${expected}="* ]]; then
      return 0
    fi
  done
  return 1
}

PY_ARGS=(
  "${PYTHON_BIN}"
  Nsight-learn/profile_reduce_ncu.py
  --impl "${IMPL}"
  --size "${SIZE}"
  --warmup "${WARMUP}"
  --profile-iters "${PROFILE_ITERS}"
  --check
)

if [ -n "${ARCH}" ]; then
  PY_ARGS+=(--arch "${ARCH}")
fi

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

NCU_ARGS=(
  --profile-from-start off
  --target-processes all
  --force-overwrite
  --kernel-name-base function
  -o "${REPORT_BASE}"
)

if [[ "${IMPL}" != "torch" ]] && ! has_ncu_arg "--kernel-name" "$@" && ! has_ncu_arg "-k" "$@"; then
  NCU_ARGS+=(--kernel-name "regex:(reduce_sum_kernel|sum_kernel)")
fi

ncu \
  "${NCU_ARGS[@]}" \
  "$@" \
  "${PY_ARGS[@]}"

ncu --import "${REPORT_FILE}" --page details --print-kernel-base function > "${TEXT_FILE}"

echo "Saved report: ${REPORT_FILE}"
echo "Saved text summary: ${TEXT_FILE}"
echo "Open in UI: ncu-ui ${REPORT_FILE}"
