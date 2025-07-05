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

#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

// #include "solve.h"
#define CEIL(a, b) ((a + b - 1) / (b))

template <const int WARP_SIZE = 32>
__device__ __forceinline__ float WarpReduceMax(float val) {
    for (unsigned int mask = WARP_SIZE >> 1; mask; mask >>= 1) {
        float other = __shfl_down_sync(0xffffffff, val, mask);
        val = fmaxf(val, other);
    }
    return val;
}

template <const int WARP_SIZE = 32>
__device__ __forceinline__ float WarpReduceSum(float val) {
    for (unsigned int mask = WARP_SIZE >> 1; mask; mask >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return val;
}

template <const int THREAD_PER_BLOCK = 256, const int WARP_SIZE = 32>
__device__ __forceinline__ float BlockReduceMax(float val) {
    const int WARPNUM = CEIL(THREAD_PER_BLOCK, WARP_SIZE);
    static __shared__ float shared[WARPNUM];
    int laneid = threadIdx.x % WARP_SIZE;
    int warpid = threadIdx.x / WARP_SIZE;

    val = WarpReduceMax(val);
    if (laneid == 0) shared[warpid] = val;
    __syncthreads();

    val = (threadIdx.x < WARPNUM) ? shared[laneid] : -FLT_MAX;
    if (warpid == 0) val = WarpReduceMax(val);
    return val;
}

template <const int THREAD_PER_BLOCK = 256, const int WARP_SIZE = 32>
__device__ __forceinline__ float BlockReduceSum(float val) {
    const int WARPNUM = CEIL(THREAD_PER_BLOCK, WARP_SIZE);
    static __shared__ float shared[WARPNUM];
    int laneid = threadIdx.x % WARP_SIZE;
    int warpid = threadIdx.x / WARP_SIZE;

    val = WarpReduceSum(val);
    if (laneid == 0) shared[warpid] = val;
    __syncthreads();

    val = (threadIdx.x < WARPNUM) ? shared[laneid] : 0.0f;
    if (warpid == 0) val = WarpReduceSum(val);
    return val;
}

template <const int THREAD_PER_BLOCK = 256, const int WARP_SIZE = 32>
__global__ void reduce_max_kernel(const float* input, float* max_warps, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float max_val = idx < N ? input[idx] : -FLT_MAX;
    max_val = BlockReduceMax(max_val);
    if (threadIdx.x == 0) max_warps[blockIdx.x] = max_val;
}

template <const int THREAD_PER_BLOCK = 256, const int WARP_SIZE = 32>
__global__ void max_kernel(float* max_warps, float* result, int BlocksPerGrid) {
    int idx = threadIdx.x;
    float max_val = -FLT_MAX;
    for (int i = idx; i < BlocksPerGrid; i += blockDim.x) {
        max_val = fmaxf(max_val, max_warps[i]);
    }
    max_val = BlockReduceMax(max_val);
    if (threadIdx.x == 0) *result = max_val;
}

template <const int THREAD_PER_BLOCK = 256, const int WARP_SIZE = 32>
__global__ void reduce_sum_kernel(const float* input, float* output, float* sum_warps, int N,
                                  float* max_val_tmp) {
    float max_val = *max_val_tmp;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_val = idx < N ? __expf(input[idx] - max_val) : 0.0f;
    output[idx] = sum_val;
    sum_val = BlockReduceSum(sum_val);
    if (threadIdx.x == 0) sum_warps[blockIdx.x] = sum_val;
}

template <const int THREAD_PER_BLOCK = 256, const int WARP_SIZE = 32>
__global__ void sum_kernel(float* sum_warps, float* result, int BlocksPerGrid) {
    int idx = threadIdx.x;
    float sum_val = 0.0f;
    for (int i = idx; i < BlocksPerGrid; i += blockDim.x) {
        sum_val += sum_warps[i];
    }
    sum_val = BlockReduceSum(sum_val);
    if (threadIdx.x == 0) *result = sum_val;
}

template <const int THREAD_PER_BLOCK = 256, const int WARP_SIZE = 32>
__global__ void softmax_kernel(const float* input, float* output, int N, float* sum_val_tmp) {
    float sum_val = *sum_val_tmp;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_val_div = __fdividef(1.0f, sum_val);
    if (idx < N) {
        output[idx] *= sum_val_div;
    }
}

void solve(const float* d_input, float* d_output, int N) {
    float *d_input, *d_output;
    float *d_max_partials, *d_max;
    float *d_sum_partials, *d_sum;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_max_partials, blocksPerGrid * sizeof(float));
    cudaMalloc((void**)&d_max, sizeof(float));
    cudaMalloc((void**)&d_sum_partials, blocksPerGrid * sizeof(float));
    cudaMalloc((void**)&d_sum, sizeof(float));

    reduce_max_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_max_partials, N);
    max_kernel<<<1, threadsPerBlock>>>(d_max_partials, d_max, blocksPerGrid);
    reduce_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_sum_partials, N,
                                                          d_max);
    sum_kernel<<<1, threadsPerBlock>>>(d_sum_partials, d_sum, blocksPerGrid);
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, d_sum);

    cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_max_partials);
    cudaFree(d_max);
    cudaFree(d_sum_partials);
    cudaFree(d_sum);
}

torch::Tensor softmax(torch::Tensor a) {
    TORCH_CHECK(a.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(a.dim() == 1, "Input tensor must be 1-dimensional");

    const int64_t N = a.numel();
    torch::Tensor output = torch::empty_like(a);

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 确保中间张量也在同一设备上
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(a.device());
    torch::Tensor d_max_partials = torch::empty({blocksPerGrid}, options);
    torch::Tensor d_max = torch::empty({1}, options);
    torch::Tensor d_sum_partials = torch::empty({blocksPerGrid}, options);
    torch::Tensor d_sum = torch::empty({1}, options);

    // 获取原始指针
    const float* input_ptr = a.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    float* max_partials_ptr = d_max_partials.data_ptr<float>();
    float* max_ptr = d_max.data_ptr<float>();
    float* sum_partials_ptr = d_sum_partials.data_ptr<float>();
    float* sum_ptr = d_sum.data_ptr<float>();

    reduce_max_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_ptr, max_partials_ptr, N);
    max_kernel<<<1, threadsPerBlock>>>(max_partials_ptr, max_ptr, blocksPerGrid);
    reduce_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_ptr, output_ptr, sum_partials_ptr,
                                                          N, max_ptr);
    sum_kernel<<<1, threadsPerBlock>>>(sum_partials_ptr, sum_ptr, blocksPerGrid);
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_ptr, output_ptr, N, sum_ptr);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

#define STRINGFY(x) #x
#define TORCH_BINDING_COMMON_EXTENSION(func) m.def(STRINGFY(func), &func, STRINGFY(func));
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { TORCH_BINDING_COMMON_EXTENSION(softmax) }