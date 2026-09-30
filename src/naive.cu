#include <cuda_runtime.h>
#include <gemm_kernels.cuh>

__global__ void gemm::kernel::naive(float const *A, float const *B, float *C,
                    float const alpha, float const beta, bool const transA,
                    bool const transB, int const result_height, int const result_width, int const common_dim) {

    int const global_col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int const global_row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    float result = 0.0f;
    if(global_col_idx < result_width && global_row_idx < result_height) {
        int const c_idx = result_width*global_row_idx+global_col_idx;
        for (int i=0; i<common_dim;i++) {

            int const a_idx = transA ? result_height*i+global_row_idx : common_dim*global_row_idx + i;
            int const b_idx = transB ? common_dim*global_col_idx+i : result_width*i + global_col_idx;


            result += A[a_idx] * B[b_idx];
        }
        C[c_idx] = alpha * result + beta*C[c_idx];
    }
}
