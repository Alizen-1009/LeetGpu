import argparse
import ctypes
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / ".torch_extensions"
SOURCE_MAP = {
    "naive": ROOT.parent / "reduction" / "reduction.cu",
    "float4": ROOT.parent / "reduction" / "reduction_float4.cu",
    "float4_atomic": ROOT.parent / "reduction" / "reduction_float4_atomic.cu",
    "torch": None,
}
CUDA_FLAGS = [
    "-O3",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Profile one reduction implementation with Nsight Compute.")
    parser.add_argument("--impl", choices=sorted(SOURCE_MAP), default="float4")
    parser.add_argument("--size", type=int, default=1 << 24)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--profile-iters", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--arch", default=None, help='Optional TORCH_CUDA_ARCH_LIST, for example "8.9".')
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--verbose-build", action="store_true")
    return parser.parse_args()


def setup_cuda(device):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the current Python environment.")
    if device >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device} is unavailable.")
    torch.cuda.set_device(device)


def load_runner(args):
    if args.impl == "torch":
        return lambda x: torch.sum(x, dtype=torch.float32).reshape(1)

    if args.arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = args.arch

    source = SOURCE_MAP[args.impl]
    if not source or not source.is_file():
        raise FileNotFoundError(f"CUDA source file not found: {source}")

    build_dir = BUILD_DIR / args.impl
    build_dir.mkdir(parents=True, exist_ok=True)

    module = load(
        name=f"reduction_{args.impl}_lib",
        sources=[str(source)],
        build_directory=str(build_dir),
        extra_cuda_cflags=CUDA_FLAGS,
        extra_cflags=["-std=c++17"],
        verbose=args.verbose_build,
    )
    return module.reduce_sum


def get_cudart():
    if hasattr(torch.cuda, "cudart"):
        return torch.cuda.cudart()
    return ctypes.CDLL("libcudart.so")


def run_profile(runner, x, warmup, profile_iters):
    for _ in range(warmup):
        runner(x)
    torch.cuda.synchronize()

    cudart = get_cudart()
    if cudart.cudaProfilerStart() != 0:
        raise RuntimeError("cudaProfilerStart failed.")

    out = None
    for _ in range(profile_iters):
        out = runner(x)

    torch.cuda.synchronize()
    if cudart.cudaProfilerStop() != 0:
        raise RuntimeError("cudaProfilerStop failed.")
    return out


def main():
    args = parse_args()
    setup_cuda(args.device)

    runner = load_runner(args)
    torch.manual_seed(0)
    x = torch.randn(args.size, device=f"cuda:{args.device}", dtype=torch.float32) * 400.0
    out = run_profile(runner, x, args.warmup, args.profile_iters)

    print(f"impl={args.impl}")
    print(f"device={torch.cuda.current_device()}:{torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"size={args.size}")
    print(f"profile_iters={args.profile_iters}")
    print(f"result={out.item():.8f}")

    if args.check:
        ref = torch.sum(x, dtype=torch.float32).reshape(1)
        diff = torch.max(torch.abs(out - ref)).item()
        print(f"torch_ref={ref.item():.8f}")
        print(f"max_abs_diff={diff:.8e}")


if __name__ == "__main__":
    main()
