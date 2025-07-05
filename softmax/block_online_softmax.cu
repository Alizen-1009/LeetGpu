
struct __align__(8) MD {
    float mx;
    float sum;
    __device__ __forceinline__ MD() : mx(-1e6f), sum(0.0f) {}
    __device__ __forceinline__ MD(float mx_val, float sum_val) : mx(mx_val), sum(sum_val) {}
};

template <const int WARP_SIZE = 32>
__device__ __forceinline__ MD warp_reduce_softmax(MD val) {
#pragma unroll
    for (unsigned int mask = WARP_SIZE >> 1; mask; mask >>= 1) {
        MD other;
        other.mx = __shfl_down_sync(0xffffffff, val.mx, mask);
        other.sum = __shfl_down_sync(0xffffffff, val.sum, mask);

        bool val_bigger = val.mx > other.mx;
        MD bigger_val = val_bigger ? val : other;
        MD small_val = val_bigger ? other : val;

        val.sum = bigger_val.sum + small_val.sum * __expf(small_val.mx - bigger_val.mx);
        val.mx = bigger_val.mx;
    }
    return val;
}

template <const int NUM_THREADS = 256, const int WARP_SIZE = 32>
__global__ void softmax_kernel(const float* input, float* output, int N) {
    int local_idx = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int WARP_NUM = NUM_THREADS / WARP_SIZE;
    int lane_idx = local_idx % WARP_SIZE;
    int warp_idx = local_idx / WARP_SIZE;

    MD val;
    float val_mx_tmp = global_idx < N ? input[global_idx] : -1e6;
    val.mx = val_mx_tmp;
    val.sum = global_idx < N ? 1.0 : 0.0;

    __shared__ MD shared_data[WARP_NUM];
    val = warp_reduce_softmax<WARP_SIZE>(val);
    if (lane_idx == 0) shared_data[warp_idx] = val;
    __syncthreads();

    if (warp_idx == 0) {
        val = lane_idx < WARP_NUM ? shared_data[lane_idx] : MD();
        val = warp_reduce_softmax<WARP_SIZE>(val);
        if (local_idx == 0) shared_data[0] = val;
    }
    __syncthreads();

    MD final_val = shared_data[0];
    float d_total_inverse = __fdividef(1.0f, final_val.sum);
    if (global_idx < N) {
        output[global_idx] = __expf(val_mx_tmp - final_val.mx) * d_total_inverse;
    }
}
