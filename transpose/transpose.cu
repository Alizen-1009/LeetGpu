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

template <const int BLOCK_SIZE = 32, const int NUM_PER_THREAD = 4>
__global__ void transpose_kernel(const float* input, float* output, int M, int N) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    __shared__ float sdata[BLOCK_SIZE][BLOCK_SIZE + 1];

    int x = bx * BLOCK_SIZE + tx;
    int y = by * BLOCK_SIZE + ty;
    if (x >= N) return;

    constexpr int ROW_STRIDE = BLOCK_SIZE / NUM_PER_THREAD;
    if (y + BLOCK_SIZE <= M) {
#pragma unroll
        for (int y_off = 0; y_off < BLOCK_SIZE; y_off += ROW_STRIDE) {
            sdata[ty + y_off][tx] = input[(y + y_off) * N + x];
        }
    } else {
        for (int y_off = 0; y_off < BLOCK_SIZE; y_off += ROW_STRIDE) {
            if (ty + y_off < M) {
                sdata[ty + y_off][tx] = input[(y + y_off) * N + x];
            }
        }
    }

    __syncthreads();

    x = by * BLOCK_SIZE + tx;
    y = bx * BLOCK_SIZE + ty;
    if (y + BLOCK_SIZE <= N) {
#pragma unroll
        for (int y_off = 0; y_off < BLOCK_SIZE; y_off += ROW_STRIDE) {
            output[(y + y_off) * M + x] = sdata[tx][ty + y_off];
        }
    } else {
        for (int y_off = 0; y_off < BLOCK_SIZE; y_off += ROW_STRIDE) {
            if (ty + y_off < N) {
                output[(y + y_off) * M + x] = sdata[tx][ty + y_off];
            }
        }
    }
}
void slove(const float* input, float* output, int M, int N) {
    const int BLOCK_SIZE = 32;
    const int NUM_PER_THREAD = 4;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE / NUM_PER_THREAD);
    dim3 grid(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));
    transpose_kernel<<<grid, block>>>(input, output, M, N);
    cudaDeviceSynchronize();
}

torch::Tensor transpose(torch::Tensor input) {
    int rows = input.size(0);
    int cols = input.size(1);
    torch::Tensor output = torch::empty_like(input);
    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    slove(input_ptr, output_ptr, rows, cols);
    return output;
}

#define STRINGFY(x) #x
#define TORCH_BINDING_COMMON_EXTENSION(func) m.def(STRINGFY(func), &func, STRINGFY(func));
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { TORCH_BINDING_COMMON_EXTENSION(transpose) }
