import time
import numpy as np
import torch
from torch.utils.cpp_extension import load
import os

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="transpose_lib",
    sources=["transpose.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)


def run_benchmark(
    perf_func1: callable,
    perf_func2: callable,
    a: torch.Tensor,
    tag1: str,
    tag2: str,
    warmup: int = 10,
    iters: int = 100,
):
    # Warmup and benchmark perf_func1
    for _ in range(warmup):
        out1 = perf_func1(a)  # warmup
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        out1 = perf_func1(a)
    torch.cuda.synchronize()
    end = time.time()
    total_time1 = (end - start) * 1000  # ms
    mean_time1 = total_time1 / iters
    out_val1 = out1.tolist()  # Convert to Python list for comparison

    # Warmup and benchmark perf_func2
    for _ in range(warmup):
        out2 = perf_func2(a)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        out2 = perf_func2(a)
    torch.cuda.synchronize()
    end = time.time()
    total_time2 = (end - start) * 1000  # ms
    mean_time2 = total_time2 / iters
    out_val2 = out2.tolist()

    # Calculate differences (assuming out1 and out2 are same shape)
    out1_np = np.array(out_val1)
    out2_np = np.array(out_val2)
    abs_diff = np.abs(out1_np - out2_np)
    mean_diff = np.mean(abs_diff)
    max_diff = np.max(abs_diff)

    # Print results
    print(f"\n{'Benchmark Results':^60}")
    print("-" * 60)
    print(f"{'Function':<14} |  {'Time (ms)':>10}  |   {'Output Shape':>15}")
    print("-" * 60)
    print(f"{tag1:<14} | {mean_time1:>10.6f}   |   {str(out1.shape):>15}")
    print(f"{tag2:<14} | {mean_time2:>10.6f}   |   {str(out2.shape):>15}")
    print("-" * 60)
    print(f"{'Mean Difference':<14}: {mean_diff:.8f}")
    print(f"{'Max  Difference':<14}: {max_diff:.8f}")
    print("-" * 60)

    return {
        "out1": out1,
        "out2": out2,
        "time1": mean_time1,
        "time2": mean_time2,
        "mean_diff": mean_diff,
        "max_diff": max_diff,
    }


Ss = [1024, 2048, 4096]

for S in Ss:
    print("=" * 60)
    print(" " * 27 + f"S={S}")
    a = torch.randn((S, S)).cuda().float()
    run_benchmark(
        lib.transpose, lambda x: torch.transpose(x, 0, 1), a, "f32", "f32_torch"
    )
