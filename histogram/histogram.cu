#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>

#include <algorithm>
#include <vector>

#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<const float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<const int4 *>(&(value))[0])

// #include "solve.h"
#define CEIL(a, b) ((a + b - 1) / (b))

__global__ void histogram_kernel(const int *a, int *hist, int N, int bins) {
    extern __shared__ int shared_hist[];
    int tid = threadIdx.x;
    for (int i = tid; i < bins; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    __syncthreads();
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < N) {
        int4 reg = LDST128BITS(a[idx]);
        atomicAdd(&shared_hist[reg.x], 1);
        atomicAdd(&shared_hist[reg.y], 1);
        atomicAdd(&shared_hist[reg.z], 1);
        atomicAdd(&shared_hist[reg.w], 1);
    } else if (idx < N) {
        for (int i = idx; i < N; i++) {
            atomicAdd(&shared_hist[a[i]], 1);
        }
    }
    __syncthreads();
    for (int i = tid; i < bins; i += blockDim.x) {
        atomicAdd(&hist[i], shared_hist[i]);
    }
}
void solve(const int *input, int *histogram, int N, int num_bins) {
    const int threads_per_block = 256;
    const int blocks = CEIL(N, threads_per_block * 4);
    const int shared_memory_size = num_bins * sizeof(int);
    histogram_kernel<<<blocks, threads_per_block, shared_memory_size>>>(input, histogram, N,
                                                                        num_bins);
}

torch::Tensor histogram(torch::Tensor a, int bins) {
    if (a.dtype() != torch::kInt32) {
        throw std::runtime_error("Input tensor must be of type int32");
    }
    auto hist = torch::zeros({bins}, torch::kInt32).cuda();
    solve(a.data_ptr<int>(), hist.data_ptr<int>(), a.numel(), bins);
    return hist;
}

#define STRINGFY(x) #x
#define TORCH_BINDING_COMMON_EXTENSION(func) m.def(STRINGFY(func), &func, STRINGFY(func));
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { TORCH_BINDING_COMMON_EXTENSION(histogram) }
