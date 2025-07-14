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
template <const int BM = 128, const int BN = 128, const int BK = 8, const int TY = 8, const int TX = 8>
__global__ void matmul_kernel(const float *__restrict__ A_ptr, const float *__restrict__ B_ptr,
                              float *__restrict__ C_ptr, const int M, const int N, const int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    __shared__ float a_shared[2][BK][BM];
    __shared__ float b_shared[2][BK][BN];
    float acc_m[TY][TX] = {0.f};
    float reg_a[TY] = {0.f};
    float reg_b[TX] = {0.f};
    float ldg_a[4] = {0.f};

    const float *A_start_ptr = A_ptr + blockIdx.y * BM * K;
    const float *B_start_ptr = B_ptr + blockIdx.x * BN;

    int A_row_per_thread = BK / 4;
    int B_row_per_thread = BN / 4;

    int A_chunk_y = tid / A_row_per_thread;
    int A_chunk_x = tid % A_row_per_thread;
    int B_chunk_y = tid / B_row_per_thread;
    int B_chunk_x = tid % B_row_per_thread;

    FLOAT4(ldg_a[0]) = FETCH_CFLOAT4(A_start_ptr[A_chunk_y * K + A_chunk_x * 4]);
    FLOAT4(b_shared[0][B_chunk_y][B_chunk_x * 4]) = FETCH_CFLOAT4(B_start_ptr[B_chunk_y * N + B_chunk_x * 4]);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        a_shared[0][A_chunk_x * 4 + i][A_chunk_y] = ldg_a[i];
    }
    __syncthreads();

    int write_stage_idx = 1;
    for (int tile = BK; tile < K; tile += BK) {
        FLOAT4(ldg_a[0]) = FETCH_CFLOAT4(A_start_ptr[A_chunk_y * K + A_chunk_x * 4 + tile]);
        FLOAT4(b_shared[write_stage_idx][B_chunk_y][B_chunk_x * 4]) =
            FETCH_CFLOAT4(B_start_ptr[(B_chunk_y + tile) * N + B_chunk_x * 4]);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            a_shared[write_stage_idx][A_chunk_x * 4 + i][A_chunk_y] = ldg_a[i];
        }

        write_stage_idx ^= 1;
#pragma unroll
        for (int k = 0; k < BK; k++) {
            FLOAT4(reg_a[0]) = FLOAT4(a_shared[write_stage_idx][k][ty * TY]);
            FLOAT4(reg_a[4]) = FLOAT4(a_shared[write_stage_idx][k][ty * TY + 4]);
            FLOAT4(reg_b[0]) = FLOAT4(b_shared[write_stage_idx][k][tx * TX]);
            FLOAT4(reg_b[4]) = FLOAT4(b_shared[write_stage_idx][k][tx * TX + 4]);
#pragma unroll
            for (int i = 0; i < TY; i++) {
#pragma unroll
                for (int j = 0; j < TX; j++) {
                    acc_m[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();
    }

    write_stage_idx ^= 1;
#pragma unroll
    for (int k = 0; k < BK; k++) {
        FLOAT4(reg_a[0]) = FLOAT4(a_shared[write_stage_idx][k][ty * TY]);
        FLOAT4(reg_a[4]) = FLOAT4(a_shared[write_stage_idx][k][ty * TY + 4]);
        FLOAT4(reg_b[0]) = FLOAT4(b_shared[write_stage_idx][k][tx * TX]);
        FLOAT4(reg_b[4]) = FLOAT4(b_shared[write_stage_idx][k][tx * TX + 4]);

#pragma unroll
        for (int i = 0; i < TY; i++) {
#pragma unroll
            for (int j = 0; j < TX; j++) {
                acc_m[i][j] += reg_a[i] * reg_b[j];
            }
        }
    }

    float *C_ptr_start = C_ptr + N * by * BM + bx * BN;
    for (int i = 0; i < TY; i++) {
        FLOAT4(C_ptr_start[N * (ty * TY + i) + tx * TX]) = FLOAT4(acc_m[i][0]);
        FLOAT4(C_ptr_start[N * (ty * TY + i) + tx * TX + 4]) = FLOAT4(acc_m[i][4]);
    }
}

__global__ void matmul_naive_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < K) {
        float value = 0.0f;
        for (int k = 0; k < K; k++) {
            value += A[row * N + k] * B[k * M + col];
        }
        C[row * K + col] = value;
    }
}
void solve(const float *A, const float *B, float *C, int M, int N, int K) {
    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_Y = 8;
    const int THREAD_SIZE_X = 8;

    dim3 block(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 grid(CEIL(N, BLOCK_SIZE_N), CEIL(M, BLOCK_SIZE_M));
    matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);
    // cuda_sgemm<<<grid, block>>>(A, B, C, M, N, K);

    cudaDeviceSynchronize();
}

torch::Tensor matmul(torch::Tensor a, torch::Tensor b) {
    // 修正矩阵维度获取
    int M = a.size(0);
    int K = a.size(1);  // 修正：应该是A的列数
    int N = b.size(1);

    // 检查维度匹配
    TORCH_CHECK(b.size(0) == K, "Matrix dimensions don't match for multiplication");
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, "Both tensors must be float32");
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Both tensors must be on CUDA device");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "Both tensors must be contiguous");

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(a.device());
    torch::Tensor c = torch::zeros({M, N}, options);

    const float *A_ptr = a.data_ptr<float>();
    const float *B_ptr = b.data_ptr<float>();
    float *C_ptr = c.data_ptr<float>();

    solve(A_ptr, B_ptr, C_ptr, M, N, K);
    return c;
}

#define STRINGFY(x) #x
#define TORCH_BINDING_COMMON_EXTENSION(func) m.def(STRINGFY(func), &func, STRINGFY(func));
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { TORCH_BINDING_COMMON_EXTENSION(matmul) }