

#include "../common.cuh"

#define TILE_DIM 32
#define MAX_KERNEL_SIZE 7
__global__ void conv_2d_kernel(const float* input, const float* kernel, float* output,
                               int input_rows, int input_cols, int kernel_rows, int kernel_cols,
                               int output_rows, int output_cols) {
    // 共享内存需要包含halo区域来处理卷积边界
    __shared__ float tile[TILE_DIM + MAX_KERNEL_SIZE - 1][TILE_DIM + MAX_KERNEL_SIZE - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int output_row = blockIdx.y * TILE_DIM + ty;
    int output_col = blockIdx.x * TILE_DIM + tx;

    // 计算在输入数据中的起始位置
    int input_start_row = blockIdx.y * TILE_DIM;
    int input_start_col = blockIdx.x * TILE_DIM;

    // 需要加载的tile大小（包括halo区域）
    int tile_size = TILE_DIM + kernel_rows - 1;

    // 协作加载共享内存数据
    for (int i = ty; i < tile_size; i += TILE_DIM) {
        for (int j = tx; j < tile_size; j += TILE_DIM) {
            int input_row = input_start_row + i;
            int input_col = input_start_col + j;

            if (input_row < input_rows && input_col < input_cols) {
                tile[i][j] = input[input_row * input_cols + input_col];
            } else {
                tile[i][j] = 0.0f;  // 零填充
            }
        }
    }

    __syncthreads();

    // 检查输出边界
    if (output_row >= output_rows || output_col >= output_cols) {
        return;
    }

    // 执行卷积计算
    float sum = 0.0f;
    for (int i = 0; i < kernel_rows; ++i) {
        for (int j = 0; j < kernel_cols; ++j) {
            int tile_row = ty + i;
            int tile_col = tx + j;
            sum += tile[tile_row][tile_col] * kernel[i * kernel_cols + j];
        }
    }

    output[output_row * output_cols + output_col] = sum;
}

void solve(const float* input, const float* kernel, float* output, int input_rows, int input_cols,
           int kernel_rows, int kernel_cols) {
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);

    // 修正的网格大小计算
    dim3 numBlocks((output_cols + TILE_DIM - 1) / TILE_DIM,
                   (output_rows + TILE_DIM - 1) / TILE_DIM);

    conv_2d_kernel<<<numBlocks, threadsPerBlock>>>(input, kernel, output, input_rows, input_cols,
                                                   kernel_rows, kernel_cols, output_rows,
                                                   output_cols);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

torch::Tensor conv_2d(torch::Tensor input, torch::Tensor kernel) {
    TORCH_CHECK(input.dim() == 2, "Input must be a 2D tensor");
    TORCH_CHECK(kernel.dim() == 2, "Kernel must be a 2D tensor");

    int input_rows = input.size(0);
    int input_cols = input.size(1);
    int kernel_rows = kernel.size(0);
    int kernel_cols = kernel.size(1);

    auto output =
        torch::zeros({input_rows - kernel_rows + 1, input_cols - kernel_cols + 1}, input.options());

    solve(input.data_ptr<float>(), kernel.data_ptr<float>(), output.data_ptr<float>(), input_rows,
          input_cols, kernel_rows, kernel_cols);

    return output;
}

#define STRINGFY(x) #x
#define TORCH_BINDING_COMMON_EXTENSION(func) m.def(STRINGFY(func), &func, STRINGFY(func));
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { TORCH_BINDING_COMMON_EXTENSION(conv_2d) }