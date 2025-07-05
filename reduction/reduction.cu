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
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// #include "solve.h"
#define CEIL(a, b) ((a + b - 1) / (b))

template <const int WARP_SIZE = 32>
__device__ __forceinline__ float WarpReduceSum(float value) {
    for (int mask = WARP_SIZE >> 1; mask; mask >>= 1) {
        value += __shfl_xor_sync(0xffffffff, value, mask);
    }
    return value;
}

template <const int WARP_SIZE = 32>
__device__ __forceinline__ float BlockReduceSum(float value) {
    int NUM_WARPS = CEIL(blockDim.x, WARP_SIZE);
    static __shared__ float shared[32];
    value = WarpReduceSum(value);
    int laneid = threadIdx.x % WARP_SIZE;
    int warpid = threadIdx.x / WARP_SIZE;

    if (laneid == 0) shared[warpid] = value;
    __syncthreads();

    value = (threadIdx.x < NUM_WARPS) ? shared[laneid] : 0.0f;
    if (warpid == 0) value = WarpReduceSum(value);
    return value;
}
template <const int THREAD_PER_BLOCK = 256, const int WARP_SIZE = 32>
__global__ void reduce_sum_kernel(const float *d_input, float *d_output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (idx < N) val = d_input[idx];
    val = BlockReduceSum(val);
    if (threadIdx.x == 0) d_output[blockIdx.x] = val;
}
template <const int THREAD_PER_BLOCK = 256, const int WARP_SIZE = 32>
__global__ void sum_kernel(float *d_input, float *d_output, int N) {
    float val = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        val += d_input[i];
    }
    val = BlockReduceSum(val);
    if (threadIdx.x == 0) *d_output = val;
}
void solve(const float *input, float *output, int N) {
    float *d_input, *d_output;
    float *d_sum_partials;

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_sum_partials, blocksPerGrid * sizeof(float));
    cudaMalloc((void **)&d_output, sizeof(float));
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    reduce_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_sum_partials, N);
    sum_kernel<<<1, threadsPerBlock>>>(d_sum_partials, d_output, blocksPerGrid);

    cudaMemcpy(output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_sum_partials);
}

torch::Tensor reduce_sum(torch::Tensor a) {
    TORCH_CHECK(a.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(a.dim() == 1, "Input tensor must be 1-dimensional");

    const int64_t N = a.numel();

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 确保中间张量也在同一设备上
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(a.device());
    torch::Tensor d_sum_partials = torch::empty({blocksPerGrid}, options);
    torch::Tensor d_sum = torch::empty({1}, options);

    // 获取原始指针
    const float *input_ptr = a.data_ptr<float>();
    float *sum_partials_ptr = d_sum_partials.data_ptr<float>();
    float *sum_ptr = d_sum.data_ptr<float>();

    reduce_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_ptr, sum_partials_ptr, N);
    sum_kernel<<<1, threadsPerBlock>>>(sum_partials_ptr, sum_ptr, blocksPerGrid);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return d_sum;
}

#define STRINGFY(x) #x
#define TORCH_BINDING_COMMON_EXTENSION(func) m.def(STRINGFY(func), &func, STRINGFY(func));
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { TORCH_BINDING_COMMON_EXTENSION(reduce_sum) }