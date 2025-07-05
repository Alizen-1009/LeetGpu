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
#define FLOAT4(value) (reinterpret_cast<const float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

// #include "solve.h"
#define CEIL(a, b) ((a + b - 1) / (b))

template <const int WARP_SIZE = 32>
__device__ __forceinline__ float WarpReduceMax(float val) {
    for (unsigned int mask = WARP_SIZE >> 1; mask; mask >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, val, mask);
        val = fmaxf(val, other);
    }

    return val;
}

template <const int WARP_SIZE = 32>
__device__ __forceinline__ float WarpReduceSum(float val) {
    for (unsigned int mask = WARP_SIZE >> 1; mask; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
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
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    float max_val = -FLT_MAX;

    // 修正边界检查：确保所有4个元素都在范围内
    if (idx + 3 < N) {
        float4 reg1 = FLOAT4(input[idx]);
        max_val = fmaxf(max_val, reg1.x);
        max_val = fmaxf(max_val, reg1.y);
        max_val = fmaxf(max_val, reg1.z);
        max_val = fmaxf(max_val, reg1.w);
    } else if (idx < N) {
        // 处理边界情况，逐个读取剩余元素
        for (int i = idx; i < N; i++) {
            max_val = fmaxf(max_val, input[i]);
        }
    }

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
__global__ void softmax_and_sum_kernel(const float* input, float* output, float* result, int N,
                                       float* max_val_tmp) {
    float max_val = *max_val_tmp;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    float sum_val = 0.0f;

    // 修正边界检查和内存访问
    if (idx + 3 < N) {
        float4 reg1 = FLOAT4(input[idx]);
        reg1.x = __expf(reg1.x - max_val);
        reg1.y = __expf(reg1.y - max_val);
        reg1.z = __expf(reg1.z - max_val);
        reg1.w = __expf(reg1.w - max_val);

        // 使用LDST128BITS进行对齐写入
        LDST128BITS(output[idx]) = reg1;
        sum_val += reg1.x + reg1.y + reg1.z + reg1.w;
    } else if (idx < N) {
        // 处理边界情况，逐个处理剩余元素
        for (int i = idx; i < N; i++) {
            float val = __expf(input[i] - max_val);
            output[i] = val;
            sum_val += val;
        }
    }

    sum_val = BlockReduceSum(sum_val);
    if (threadIdx.x == 0) atomicAdd(result, sum_val);
    __threadfence();

    // 归一化阶段
    float sum_val_div = __fdividef(1.0f, *result);
    if (idx + 3 < N) {
        float4 reg2 = FLOAT4(output[idx]);
        reg2.x *= sum_val_div;
        reg2.y *= sum_val_div;
        reg2.z *= sum_val_div;
        reg2.w *= sum_val_div;
        LDST128BITS(output[idx]) = reg2;
    } else if (idx < N) {
        for (int i = idx; i < N; i++) {
            output[i] *= sum_val_div;
        }
    }
}

void solve(const float* d_input, float* d_output, int N) {
    float *d_max_partials, *d_max;
    float* d_sum;
    const int threadsPerBlock = 256;
    const int numsPerBlock = 256 * 4;
    int blocksPerGrid = (N + numsPerBlock - 1) / numsPerBlock;

    // 移除N%4==0的强制要求
    cudaMalloc((void**)&d_max_partials, blocksPerGrid * sizeof(float));
    cudaMalloc((void**)&d_max, sizeof(float));
    cudaMalloc((void**)&d_sum, sizeof(float));

    // 初始化sum为0
    cudaMemset(d_sum, 0, sizeof(float));

    reduce_max_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_max_partials, N);
    max_kernel<<<1, threadsPerBlock>>>(d_max_partials, d_max, blocksPerGrid);
    softmax_and_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_sum, N, d_max);

    cudaFree(d_max_partials);
    cudaFree(d_max);
    cudaFree(d_sum);
}

torch::Tensor softmax(torch::Tensor a) {
    TORCH_CHECK(a.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(a.dim() == 1, "Input tensor must be 1-dimensional");

    const int64_t N = a.numel();
    torch::Tensor output = torch::empty_like(a);

    const int threadsPerBlock = 256;
    const int numsPerBlock = 256 * 4;
    const int blocksPerGrid = (N + numsPerBlock - 1) / numsPerBlock;

    // 移除N%4==0的强制要求，但建议使用对齐的大小以获得最佳性能
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(a.device());
    torch::Tensor d_max_partials = torch::empty({blocksPerGrid}, options);
    torch::Tensor d_max = torch::empty({1}, options);
    torch::Tensor d_sum = torch::zeros({1}, options);  // 使用zeros而不是empty

    // 获取原始指针
    const float* input_ptr = a.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    float* max_partials_ptr = d_max_partials.data_ptr<float>();
    float* max_ptr = d_max.data_ptr<float>();
    float* sum_ptr = d_sum.data_ptr<float>();

    reduce_max_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_ptr, max_partials_ptr, N);
    max_kernel<<<1, threadsPerBlock>>>(max_partials_ptr, max_ptr, blocksPerGrid);
    softmax_and_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_ptr, output_ptr, sum_ptr, N,
                                                               max_ptr);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

#define STRINGFY(x) #x
#define TORCH_BINDING_COMMON_EXTENSION(func) m.def(STRINGFY(func), &func, STRINGFY(func));
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { TORCH_BINDING_COMMON_EXTENSION(softmax) }