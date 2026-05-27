import argparse
import ctypes
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
KERNEL_ROOT = REPO_ROOT / "reduction"
SOURCE_MAP = {
    "naive": KERNEL_ROOT / "reduction.cu",
    "float4": KERNEL_ROOT / "reduction_float4.cu",
    "float4_atomic": KERNEL_ROOT / "reduction_float4_atomic.cu",
    "torch": None,
}

CUDA_FLAGS = [
    "-O3",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark and profile reduction kernels with Nsight Compute."
    )
    parser.add_argument(
        "--impl",
        choices=sorted(SOURCE_MAP.keys()),
        default="float4",
        help="Reduction implementation to run.",
    )
    parser.add_argument(
        "--mode",
        choices=["bench", "profile"],
        default="bench",
        help="bench: time the op with CUDA events; profile: bracket the op with cudaProfilerStart/Stop.",
    )
    parser.add_argument("--size", type=int, default=1 << 24, help="1D tensor length.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=200, help="Benchmark iterations.")
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=1,
        help="Iterations inside the profiler start/stop range.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument(
        "--arch",
        default=None,
        help='Optional TORCH_CUDA_ARCH_LIST override, for example "8.9".',
    )
    parser.add_argument(
        "--build-dir",
        default=str(ROOT / ".torch_extensions"),
        help="Directory used by torch extension build cache.",
    )
    parser.add_argument(
        "--kernel-dir",
        default=str(KERNEL_ROOT),
        help="Directory containing the reduction CUDA source files.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare the final result with torch.sum and print the max abs diff.",
    )
    parser.add_argument(
        "--verbose-build",
        action="store_true",
        help="Print verbose torch extension build logs.",
    )
    return parser.parse_args()


def ensure_cuda(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the current Python environment.")
    if args.device >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {args.device} is unavailable, only found {torch.cuda.device_count()} device(s)."
        )
    torch.cuda.set_device(args.device)


def load_impl(args):
    if args.impl == "torch":
        return None

    if args.arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = args.arch

    kernel_dir = Path(args.kernel_dir).resolve()
    source_map = {
        "naive": kernel_dir / "reduction.cu",
        "float4": kernel_dir / "reduction_float4.cu",
        "float4_atomic": kernel_dir / "reduction_float4_atomic.cu",
        "torch": None,
    }
    source_path = source_map[args.impl]
    if not source_path.exists():
        raise FileNotFoundError(f"CUDA source file not found: {source_path}")

    build_dir = Path(args.build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    impl_build_dir = build_dir / args.impl
    impl_build_dir.mkdir(parents=True, exist_ok=True)

    return load(
        name=f"reduction_{args.impl}_lib",
        sources=[str(source_path)],
        build_directory=str(impl_build_dir),
        extra_cuda_cflags=CUDA_FLAGS,
        extra_cflags=["-std=c++17"],
        verbose=args.verbose_build,
    )


def make_input(args):
    torch.manual_seed(args.seed)
    return torch.randn(args.size, device=f"cuda:{args.device}", dtype=torch.float32) * 400.0


def make_runner(args, module):
    if args.impl == "torch":
        return lambda x: torch.sum(x, dtype=torch.float32).reshape(1)
    return module.reduce_sum


def compare_with_torch(x, out):
    ref = torch.sum(x, dtype=torch.float32).reshape(1)
    max_abs_diff = torch.max(torch.abs(out - ref)).item()
    return ref, max_abs_diff


def benchmark(runner, x, warmup, iters):
    for _ in range(warmup):
        runner(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    out = None
    for _ in range(iters):
        out = runner(x)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    return out, total_ms / iters


def get_cudart():
    if hasattr(torch.cuda, "cudart"):
        return torch.cuda.cudart()
    return ctypes.CDLL("libcudart.so")


def cuda_profiler_start():
    result = get_cudart().cudaProfilerStart()
    if result != 0:
        raise RuntimeError(f"cudaProfilerStart failed with error code {result}.")


def cuda_profiler_stop():
    result = get_cudart().cudaProfilerStop()
    if result != 0:
        raise RuntimeError(f"cudaProfilerStop failed with error code {result}.")


def profile_region(runner, x, warmup, profile_iters):
    for _ in range(warmup):
        runner(x)
    torch.cuda.synchronize()

    cuda_profiler_start()
    out = None
    for _ in range(profile_iters):
        out = runner(x)
    torch.cuda.synchronize()
    cuda_profiler_stop()
    return out


def format_bandwidth_gbps(x, mean_ms):
    bytes_read = x.numel() * x.element_size()
    return bytes_read / (mean_ms / 1e3) / 1e9


def main():
    args = parse_args()
    ensure_cuda(args)

    module = load_impl(args)
    x = make_input(args)
    runner = make_runner(args, module)

    print(f"impl={args.impl}")
    print(f"mode={args.mode}")
    print(f"device={torch.cuda.current_device()}:{torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"size={args.size}")
    print(f"dtype={x.dtype}")

    if args.mode == "bench":
        out, mean_ms = benchmark(runner, x, args.warmup, args.iters)
        print(f"mean_latency_ms={mean_ms:.6f}")
        print(f"approx_read_bandwidth_gbps={format_bandwidth_gbps(x, mean_ms):.3f}")
    else:
        out = profile_region(runner, x, args.warmup, args.profile_iters)
        print(f"profile_iters={args.profile_iters}")
        print("profile_region=cudaProfilerStart/Stop")

    if args.check:
        ref, max_abs_diff = compare_with_torch(x, out)
        print(f"result={out.item():.8f}")
        print(f"torch_ref={ref.item():.8f}")
        print(f"max_abs_diff={max_abs_diff:.8e}")
    else:
        print(f"result={out.item():.8f}")


if __name__ == "__main__":
    main()
