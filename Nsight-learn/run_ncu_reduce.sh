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

PY_ARGS=(
  python
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

ncu \
  --profile-from-start off \
  --target-processes all \
  --force-overwrite \
  -o "${REPORT_BASE}" \
  "$@" \
  "${PY_ARGS[@]}"

ncu --import "${REPORT_FILE}" --page details > "${TEXT_FILE}"

echo "Saved report: ${REPORT_FILE}"
echo "Saved text summary: ${TEXT_FILE}"
echo "Open in UI: ncu-ui ${REPORT_FILE}"
