// 定义FLOAT4向量化读取宏

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

#define CEIL(a, b) ((a + b - 1) / (b))
#define FETCH_FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define FETCH_CFLOAT4(value) (reinterpret_cast<const float4 *>(&(value))[0])

template <const int SIZE>
__device__ __forceinline__ float DynamicWarpReduceSum(float val) {
    for (int mask = SIZE >> 1; mask; mask >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return val;
}

template <const int ROW_PER_WARP, const int WARP_SIZE = 32>
__global__ void sgemv_16_kernel(const float *A, const float *x, float *res, int M, int N) {
    int bid = blockIdx.x;
    const int ROW_NUMS = WARP_SIZE / ROW_PER_WARP;
    int col = threadIdx.x % ROW_NUMS;
    int row = (bid * blockDim.y + threadIdx.y) * ROW_PER_WARP + threadIdx.x / ROW_NUMS;

    if (row < M) {
        float sum = (col < N) ? A[row * N + col] * x[col] : 0.f;
        sum = DynamicWarpReduceSum<ROW_NUMS>(sum);
        if (col == 0) res[row] = sum;
    }
}

template <const int WARP_SIZE = 32>
__global__ void sgemv_32_kernel(const float *A, const float *x, float *res, int M, int N) {
    int bid = blockIdx.x;
    int row = bid * blockDim.y + threadIdx.y;

    if (row < M) {
        float sum = 0;
        for (int col = threadIdx.x; col < N; col += blockDim.x) {
            sum += A[row * N + col] * x[col];
        }
        sum = DynamicWarpReduceSum<WARP_SIZE>(sum);
        if (threadIdx.x == 0) res[row] = sum;
    }
}

template <const int WARP_SIZE = 32>
__global__ void sgemv_128_kernel(const float *A, const float *x, float *res, int M, int N) {
    int bid = blockIdx.x;
    int row = bid * blockDim.y + threadIdx.y;

    if (row < M) {
        float sum = 0;
        for (int col = threadIdx.x * 4; col < N; col += blockDim.x * 4) {
            if (col + 3 < N) {
                float4 reg_A = FETCH_CFLOAT4(A[row * N + col]);
                float4 reg_x = FETCH_CFLOAT4(x[col]);
                sum += reg_A.x * reg_x.x;
                sum += reg_A.y * reg_x.y;
                sum += reg_A.z * reg_x.z;
                sum += reg_A.w * reg_x.w;
            } else {
                for (int idx = col; idx < N; idx++) {
                    sum += A[row * N + idx] * x[idx];
                }
            }
        }
        sum = DynamicWarpReduceSum<WARP_SIZE>(sum);
        if (threadIdx.x == 0) res[row] = sum;
    }
}

void solve(const float *A, const float *x, float *res, int M, int N) {
    if (N <= 16) {
        const int WARP_SIZE = 32;
        const int ROW_PER_WARP = 2;
        const int THREAD_PER_BLOCK = 128;
        const int WARP_PER_BLOCK = THREAD_PER_BLOCK / WARP_SIZE;
        const int ROW_PER_BLOCK = WARP_PER_BLOCK * ROW_PER_WARP;
        dim3 grid(CEIL(M, ROW_PER_BLOCK));
        dim3 block(WARP_SIZE, THREAD_PER_BLOCK / WARP_SIZE);
        sgemv_16_kernel<ROW_PER_WARP><<<grid, block>>>(A, x, res, M, N);
    } else if (N < 128) {
        const int WARP_SIZE = 32;
        const int THREAD_PER_BLOCK = 128;
        const int WARP_PER_BLOCK = THREAD_PER_BLOCK / WARP_SIZE;
        const int ROW_PER_BLOCK = WARP_PER_BLOCK;
        dim3 grid(CEIL(M, ROW_PER_BLOCK));
        dim3 block(WARP_SIZE, THREAD_PER_BLOCK / WARP_SIZE);
        sgemv_32_kernel<<<grid, block>>>(A, x, res, M, N);
    } else {
        const int WARP_SIZE = 32;
        const int THREAD_PER_BLOCK = 128;
        const int WARP_PER_BLOCK = THREAD_PER_BLOCK / WARP_SIZE;
        const int ROW_PER_BLOCK = WARP_PER_BLOCK;
        dim3 grid(CEIL(M, ROW_PER_BLOCK));
        dim3 block(WARP_SIZE, THREAD_PER_BLOCK / WARP_SIZE);
        sgemv_32_kernel<<<grid, block>>>(A, x, res, M, N);
    }
}

torch::Tensor gemv(torch::Tensor A, torch::Tensor x) {
    int M = A.size(0);
    int N = A.size(1);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(A.device());
    torch::Tensor res = torch::zeros({M}, options);
    const float *A_ptr = A.data_ptr<float>();
    const float *x_ptr = x.data_ptr<float>();
    float *res_ptr = res.data_ptr<float>();
    solve(A_ptr, x_ptr, res_ptr, M, N);
    return res;
}

#define STRINGFY(x) #x
#define TORCH_BINDING_COMMON_EXTENSION(func) m.def(STRINGFY(func), &func, STRINGFY(func));
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { TORCH_BINDING_COMMON_EXTENSION(gemv) }