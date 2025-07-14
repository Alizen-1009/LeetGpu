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
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 &>(pointer))
#define FETCH_CFLOAT4(pointer) (reinterpret_cast<const float4 &>(pointer))

template <const int BLOCK_SIZE_M = 128, const int BLOCK_SIZE_N = 128, const int BLOCK_SIZE_K = 8,
          const int THREAD_SIZE_Y = 8, const int THREAD_SIZE_X = 8>
__global__ void matmul_kernel(const float *__restrict__ A_ptr, const float *__restrict__ B_ptr,
                              float *__restrict__ C_ptr, const int M, const int N, const int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int tid = ty * blockDim.x + tx;
    __shared__ float a_shared[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float b_shared[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0.f};
    float reg_a[THREAD_SIZE_Y] = {0.f};
    float reg_b[THREAD_SIZE_X] = {0.f};
    float ldg_a_reg[4] = {0.f};

    const float *A_ptr_start = A_ptr + blockIdx.y * BLOCK_SIZE_M * K;
    const float *B_ptr_start = B_ptr + blockIdx.x * BLOCK_SIZE_N;

    const int A_tile_thread_per_row = BLOCK_SIZE_K / 4;
    const int B_tile_thread_per_row = BLOCK_SIZE_N / 4;

    const int A_tile_tid_x = tid % A_tile_thread_per_row;
    const int A_tile_tid_y = tid / A_tile_thread_per_row;
    const int B_tile_tid_x = tid % B_tile_thread_per_row;
    const int B_tile_tid_y = tid / B_tile_thread_per_row;

    auto safe_load_A_float4 = [&](const float *src_ptr, float *dst_reg, int row, int col_base) {
        if (row < M && col_base + 3 < K) {
            FETCH_FLOAT4(dst_reg[0]) = FETCH_CFLOAT4(src_ptr[0]);
        } else {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                dst_reg[i] = (row < M && col_base + i < K) ? src_ptr[i] : 0.0f;
            }
        }
    };

    auto safe_load_B_float4 = [&](const float *src_ptr, float *dst_reg, int row, int col_base) {
        if (row < K && col_base + 3 < N) {
            FETCH_FLOAT4(dst_reg[0]) = FETCH_CFLOAT4(src_ptr[0]);
        } else {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                dst_reg[i] = (row < K && col_base + i < N) ? src_ptr[i] : 0.0f;
            }
        }
    };

    // 初始化循环变量
    int A_row = by * BLOCK_SIZE_M + A_tile_tid_y;
    int A_col_base = A_tile_tid_x * 4;
    int B_row = B_tile_tid_y;
    int B_col_base = bx * BLOCK_SIZE_N + B_tile_tid_x * 4;

    // 加载第一个tile
    safe_load_A_float4(A_ptr_start + A_tile_tid_y * K + A_tile_tid_x * 4, ldg_a_reg, A_row, A_col_base);
    safe_load_B_float4(B_ptr_start + B_tile_tid_y * N + B_tile_tid_x * 4,
                       &b_shared[0][B_tile_tid_y][B_tile_tid_x * 4], B_row, B_col_base);

    // 存储到共享内存
#pragma unroll
    for (int i = 0; i < 4; i++) {
        a_shared[0][A_tile_tid_x * 4 + i][A_tile_tid_y] = ldg_a_reg[i];
    }
    __syncthreads();

    int write_stage_idx = 1;
    for (int tile_k = BLOCK_SIZE_K; tile_k < K; tile_k += BLOCK_SIZE_K) {
        // 更新加载位置
        A_col_base = tile_k + A_tile_tid_x * 4;
        B_row = tile_k + B_tile_tid_y;

        // 预加载下一个tile
        safe_load_A_float4(A_ptr_start + A_tile_tid_y * K + A_col_base, ldg_a_reg, A_row, A_col_base);
        safe_load_B_float4(B_ptr_start + B_row * N + B_tile_tid_x * 4,
                           &b_shared[write_stage_idx][B_tile_tid_y][B_tile_tid_x * 4], B_row, B_col_base);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            a_shared[write_stage_idx][A_tile_tid_x * 4 + i][A_tile_tid_y] = ldg_a_reg[i];
        }
        write_stage_idx ^= 1;
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; k++) {
            // 从共享内存加载到寄存器
            FETCH_FLOAT4(reg_a[0]) = FETCH_FLOAT4(a_shared[write_stage_idx][k][ty * THREAD_SIZE_Y]);
            FETCH_FLOAT4(reg_a[4]) = FETCH_FLOAT4(a_shared[write_stage_idx][k][ty * THREAD_SIZE_Y + 4]);
            FETCH_FLOAT4(reg_b[0]) = FETCH_FLOAT4(b_shared[write_stage_idx][k][tx * THREAD_SIZE_X]);
            FETCH_FLOAT4(reg_b[4]) = FETCH_FLOAT4(b_shared[write_stage_idx][k][tx * THREAD_SIZE_X + 4]);

            // 计算外积
#pragma unroll
            for (int i = 0; i < THREAD_SIZE_Y; i++) {
#pragma unroll
                for (int j = 0; j < THREAD_SIZE_X; j++) {
                    accum[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int k = 0; k < BLOCK_SIZE_K; k++) {
        FETCH_FLOAT4(reg_a[0]) = FETCH_FLOAT4(a_shared[write_stage_idx ^ 1][k][ty * THREAD_SIZE_Y]);
        FETCH_FLOAT4(reg_a[4]) = FETCH_FLOAT4(a_shared[write_stage_idx ^ 1][k][ty * THREAD_SIZE_Y + 4]);
        FETCH_FLOAT4(reg_b[0]) = FETCH_FLOAT4(b_shared[write_stage_idx ^ 1][k][tx * THREAD_SIZE_X]);
        FETCH_FLOAT4(reg_b[4]) = FETCH_FLOAT4(b_shared[write_stage_idx ^ 1][k][tx * THREAD_SIZE_X + 4]);

#pragma unroll
        for (int i = 0; i < THREAD_SIZE_Y; i++) {
#pragma unroll
            for (int j = 0; j < THREAD_SIZE_X; j++) {
                accum[i][j] += reg_a[i] * reg_b[j];
            }
        }
    }

    // 写回结果
    int C_row_base = by * BLOCK_SIZE_M + ty * THREAD_SIZE_Y;
    int C_col_base = bx * BLOCK_SIZE_N + tx * THREAD_SIZE_X;

#pragma unroll
    for (int i = 0; i < THREAD_SIZE_Y; i++) {
        int C_row = C_row_base + i;
        if (C_row >= M) break;

        float *C_ptr_row = C_ptr + C_row * N + C_col_base;

        if (C_col_base + THREAD_SIZE_X <= N) {
            // 可以安全写入完整的THREAD_SIZE_X个元素
            FETCH_FLOAT4(C_ptr_row[0]) = FETCH_FLOAT4(accum[i][0]);
            FETCH_FLOAT4(C_ptr_row[4]) = FETCH_FLOAT4(accum[i][4]);
        } else {
            // 需要边界检查
            for (int j = 0; j < THREAD_SIZE_X && C_col_base + j < N; j++) {
                C_ptr_row[j] = accum[i][j];
            }
        }
    }
}
__global__ void matmul_naive_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float value = 0.0f;
        for (int k = 0; k < K; k++) {
            value += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}
void solve(const float *A, const float *B, float *C, int M, int N, int K) {
    if (M % 4 != 0 || N % 4 != 0 || K % 4 != 0) {
        const int BLOCK_SIZE = 16;
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));
        matmul_naive_kernel<<<grid, block>>>(A, B, C, M, N, K);
    } else {
        const int BLOCK_SIZE_M = 128;
        const int BLOCK_SIZE_N = 128;
        const int THREAD_SIZE_Y = 8;
        const int THREAD_SIZE_X = 8;

        dim3 block(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 grid(CEIL(N, BLOCK_SIZE_N), CEIL(M, BLOCK_SIZE_M));
        matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);
    }
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
#define TORCH_BINDING_COMMON_EXTENSION(func) \
    m./ root / zyh / cuda - kernel / tekldef(STRINGFY(func), &func, STRINGFY(func));
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { TORCH_BINDING_COMMON_EXTENSION(matmul) }