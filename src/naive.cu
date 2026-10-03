#include <cuda_runtime.h>
#include <gemm_kernels.cuh>


__global__ void gemm::kernel::naive(float const* __restrict__ A, float const* __restrict__ B, float* __restrict__ C,
    float const alpha, float const beta, bool const transA,
    bool const transB, int const result_height, int const result_width, int const common_dim) {

    int const row = blockIdx.y * blockDim.y + threadIdx.y;
    int const col = blockIdx.x * blockDim.x + threadIdx.x;

    float local_result = 0.0f;
    if(col < result_width && row < result_height) {
        int const c_idx = result_width*row+col;
        for (int k=0; k<common_dim;k++) {

            int const a_idx = transA ? result_height*k+row : common_dim*row + k;
            int const b_idx = transB ? common_dim*col+k : result_width*k + col;


            local_result += A[a_idx] * B[b_idx];
        }
        C[c_idx] = alpha * local_result + beta*C[c_idx];
    }
}


