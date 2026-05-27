#!/usr/bin/env bash
set -euo pipefail

IMPL="${1:-float4}"
SIZE="${2:-16777216}"
OUTPUT_DIR="${OUTPUT_DIR:-Nsight-learn/ncu-reports}"
REPORT_NAME="${REPORT_NAME:-reduce_${IMPL}_${SIZE}}"
WARMUP="${WARMUP:-20}"
PROFILE_ITERS="${PROFILE_ITERS:-1}"
ARCH="${ARCH:-}"
KERNEL_DIR="${KERNEL_DIR:-reduction}"

EXTRA_NCU_ARGS=()
if [ "$#" -gt 2 ]; then
  EXTRA_NCU_ARGS=("${@:3}")
fi

mkdir -p "${OUTPUT_DIR}"

PY_ARGS=(
  python
  Nsight-learn/profile_reduce_ncu.py
  --mode profile
  --impl "${IMPL}"
  --size "${SIZE}"
  --warmup "${WARMUP}"
  --profile-iters "${PROFILE_ITERS}"
  --kernel-dir "${KERNEL_DIR}"
  --check
)

if [ -n "${ARCH}" ]; then
  PY_ARGS+=(--arch "${ARCH}")
fi

ncu \
  --profile-from-start off \
  --target-processes all \
  --force-overwrite true \
  -o "${OUTPUT_DIR}/${REPORT_NAME}" \
  "${EXTRA_NCU_ARGS[@]}" \
  "${PY_ARGS[@]}"
