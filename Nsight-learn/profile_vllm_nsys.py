import argparse
import ctypes
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from vllm import LLM, SamplingParams


DEFAULT_MODEL = Path.home() / "models" / "Qwen3.5-0.8B"
DEFAULT_PROMPT = "请用三句话解释 CUDA kernel launch overhead 是什么。"


def parse_args():
    parser = argparse.ArgumentParser(description="Profile one vLLM generate request with Nsight Systems.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--profile-iters", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="auto", choices=["auto", "half", "float16", "bfloat16", "float", "float32"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-profiler", action="store_true", help="Call cudaProfilerStart/Stop around profiled requests.")
    parser.add_argument("--print-output", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative.")
    if args.profile_iters <= 0:
        parser.error("--profile-iters must be positive.")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive.")
    return args


@contextmanager
def nvtx_range(name):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def get_cudart():
    if hasattr(torch.cuda, "cudart"):
        return torch.cuda.cudart()
    return ctypes.CDLL("libcudart.so")


def maybe_cuda_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def ensure_python_bin_on_path():
    python_bin = Path(sys.executable).resolve().parent
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if str(python_bin) not in path_parts:
        os.environ["PATH"] = str(python_bin) + os.pathsep + os.environ.get("PATH", "")


def build_prompts(prompt, batch_size):
    if batch_size == 1:
        return [prompt]
    return [f"{prompt}\nRequest index: {i}" for i in range(batch_size)]


def count_tokens(outputs):
    prompt_tokens = 0
    output_tokens = 0
    for item in outputs:
        prompt_token_ids = getattr(item, "prompt_token_ids", None)
        if prompt_token_ids is not None:
            prompt_tokens += len(prompt_token_ids)
        for completion in item.outputs:
            output_tokens += len(completion.token_ids)
    return prompt_tokens, output_tokens


def generate_once(llm, prompts, sampling_params, name):
    with nvtx_range(name):
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        maybe_cuda_synchronize()
        elapsed = time.perf_counter() - start
    return outputs, elapsed


def main():
    args = parse_args()
    ensure_python_bin_on_path()
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)

    model_path = Path(args.model).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    print(f"model={model_path}")
    print(f"batch_size={args.batch_size}")
    print(f"warmup={args.warmup}")
    print(f"profile_iters={args.profile_iters}")
    print(f"max_tokens={args.max_tokens}")
    print(f"max_model_len={args.max_model_len}")
    print(f"enforce_eager={args.enforce_eager}")
    print(f"cuda_profiler={args.cuda_profiler}")
    if torch.cuda.is_available():
        print(f"device={torch.cuda.current_device()}:{torch.cuda.get_device_name(torch.cuda.current_device())}")

    prompts = build_prompts(args.prompt, args.batch_size)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    with nvtx_range("vllm:init"):
        llm = LLM(
            model=str(model_path),
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            enforce_eager=args.enforce_eager,
            seed=args.seed,
        )

    for i in range(args.warmup):
        outputs, elapsed = generate_once(llm, prompts, sampling_params, f"vllm:warmup_{i}")
        prompt_tokens, output_tokens = count_tokens(outputs)
        print(f"warmup_{i}_seconds={elapsed:.6f}")
        print(f"warmup_{i}_prompt_tokens={prompt_tokens}")
        print(f"warmup_{i}_output_tokens={output_tokens}")

    cudart = get_cudart()
    if args.cuda_profiler and cudart.cudaProfilerStart() != 0:
        raise RuntimeError("cudaProfilerStart failed.")

    total_elapsed = 0.0
    total_prompt_tokens = 0
    total_output_tokens = 0
    last_outputs = None
    try:
        with nvtx_range("vllm:profile"):
            for i in range(args.profile_iters):
                outputs, elapsed = generate_once(llm, prompts, sampling_params, f"vllm:request_{i}")
                prompt_tokens, output_tokens = count_tokens(outputs)
                total_elapsed += elapsed
                total_prompt_tokens += prompt_tokens
                total_output_tokens += output_tokens
                last_outputs = outputs
                print(f"request_{i}_seconds={elapsed:.6f}")
                print(f"request_{i}_prompt_tokens={prompt_tokens}")
                print(f"request_{i}_output_tokens={output_tokens}")
    finally:
        if args.cuda_profiler and cudart.cudaProfilerStop() != 0:
            raise RuntimeError("cudaProfilerStop failed.")

    print(f"profile_total_seconds={total_elapsed:.6f}")
    print(f"profile_prompt_tokens={total_prompt_tokens}")
    print(f"profile_output_tokens={total_output_tokens}")
    if total_elapsed > 0:
        print(f"profile_output_tokens_per_second={total_output_tokens / total_elapsed:.6f}")

    if args.print_output and last_outputs:
        for request_id, item in enumerate(last_outputs):
            text = item.outputs[0].text
            print(f"output[{request_id}]={text!r}")


if __name__ == "__main__":
    main()
