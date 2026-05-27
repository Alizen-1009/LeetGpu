#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${1:-/home/alizen/models/Qwen3.5-0.8B}"
MAX_TOKENS="${2:-64}"
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))

PYTHON_BIN="${PYTHON:-/home/alizen/miniconda3/envs/vllm/bin/python}"
PYTHON_BINDIR="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/nsys-reports}"
REPORT_BASE="${REPORT_BASE:-${OUTPUT_DIR}/vllm_qwen_request}"
REPORT_FILE="${REPORT_BASE}.nsys-rep"
TEXT_FILE="${REPORT_BASE}.txt"

PROMPT="${PROMPT:-请用三句话解释 CUDA kernel launch overhead 是什么。}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WARMUP="${WARMUP:-1}"
PROFILE_ITERS="${PROFILE_ITERS:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
DTYPE="${DTYPE:-auto}"

PY_ARGS=(
  "${PYTHON_BIN}"
  Nsight-learn/profile_vllm_nsys.py
  --model "${MODEL}"
  --prompt "${PROMPT}"
  --batch-size "${BATCH_SIZE}"
  --warmup "${WARMUP}"
  --profile-iters "${PROFILE_ITERS}"
  --max-tokens "${MAX_TOKENS}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --dtype "${DTYPE}"
  --cuda-profiler
)

if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  PY_ARGS+=(--enforce-eager)
fi

if [[ "${PRINT_OUTPUT:-0}" == "1" ]]; then
  PY_ARGS+=(--print-output)
fi

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"
export PATH="${PYTHON_BINDIR}:${PATH}"

nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --cuda-trace-scope=process-tree \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --cuda-graph-trace=graph \
  --sample=none \
  --cpuctxsw=none \
  --output "${REPORT_BASE}" \
  "$@" \
  "${PY_ARGS[@]}"

nsys stats \
  --force-overwrite=true \
  --report nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
  --format column \
  --output - \
  "${REPORT_FILE}" > "${TEXT_FILE}"

echo "Saved report: ${REPORT_FILE}"
echo "Saved text summary: ${TEXT_FILE}"
echo "Open in UI: nsys-ui ${REPORT_FILE}"
