import time
import numpy as np
import torch
from torch.utils.cpp_extension import load
import os
import sys

# --------------------------
# CUDA 检查 & 环境变量设置
# --------------------------
if not torch.cuda.is_available():
    print("CUDA is not available. Please check your driver or PyTorch installation.")
    sys.exit(1)

os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_grad_enabled(False)

# --------------------------
# 加载 CUDA 扩展（2dconv.cu）
# --------------------------
try:
    lib = load(
        name="conv_2d_lib",
        sources=["2dconv.cu"],
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
        verbose=True,
    )
except Exception as e:
    print(f"CUDA extension failed to build: {e}")
    sys.exit(1)


# --------------------------
# Benchmark 工具函数
# --------------------------
def run_benchmark(
    perf_func1: callable,
    perf_func2: callable,
    a: torch.Tensor,
    kernel: torch.Tensor,
    tag1: str,
    tag2: str,
    warmup: int = 10,
    iters: int = 100,
):
    # Warmup and benchmark perf_func1
    for _ in range(warmup):
        out1 = perf_func1(a, kernel)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        out1 = perf_func1(a, kernel)
    torch.cuda.synchronize()
    end = time.time()
    total_time1 = (end - start) * 1000
    mean_time1 = total_time1 / iters

    # Warmup and benchmark perf_func2
    for _ in range(warmup):
        out2 = perf_func2(a)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        out2 = perf_func2(a)
    torch.cuda.synchronize()
    end = time.time()
    total_time2 = (end - start) * 1000
    mean_time2 = total_time2 / iters

    # Compare results
    out1_np = out1.detach().cpu().numpy()
    out2_np = out2.detach().cpu().numpy()
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


# --------------------------
# 主运行逻辑
# --------------------------
Ss = [1024, 2048, 4096]  # 可选：8192

for S in Ss:
    print("=" * 60)
    print(" " * 25 + f"S = {S}")

    # [H, W] 2D tensor 给 CUDA kernel
    input_tensor = torch.randn((S, S), dtype=torch.float32).to("cuda")
    kernel = torch.randn((3, 3), dtype=torch.float32).to("cuda")

    def torch_conv2d_like_custom(input_2d: torch.Tensor):
        # PyTorch conv2d 需要 4D 输入和 kernel
        input_4d = input_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        output = torch.nn.functional.conv2d(input_4d, kernel_4d, padding=0)  # 保持大小
        return output.squeeze(0).squeeze(0)  # 回到 [H, W]

    run_benchmark(
        lambda x, k: lib.conv_2d(x, k),  # 自定义 CUDA kernel（接收 [H, W], [K, K]）
        torch_conv2d_like_custom,  # PyTorch baseline（也输出 [H, W]）
        input_tensor,
        kernel,
        "custom_conv2d",
        "torch_conv2d",
    )
