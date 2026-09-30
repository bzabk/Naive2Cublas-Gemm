#include <gemm_kernels.cuh>
#define BLOCK_SIZE 16


__global__ void gemm::kernel::shared_tiling(float const *A, float const *B, float *C,
    float const alpha, float const beta, bool const transA, bool const transB,
    int const result_height, int const result_width, int const common_dim) {


    int const col = blockIdx.x*blockDim.x + threadIdx.x;
    int const row = blockIdx.y*blockDim.y + threadIdx.y;

    int const thread_x = threadIdx.x;
    int const thread_y = threadIdx.y;

    __shared__ float submatrix_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float submatrix_B[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    for (int i=0; i<common_dim;i+=BLOCK_SIZE) {

        if (row < result_height && (i+thread_x) < common_dim) {
            int a_idx = transA ? result_height*(i+thread_x)+row : common_dim*row+i+thread_x;
            submatrix_A[thread_y][thread_x] = A[a_idx];
        }else {
            submatrix_A[thread_y][thread_x] = 0.0f;
        }

        if (col<result_width && (i+thread_y)<common_dim) {
            int b_idx = transB ? common_dim*col+(i+thread_y) : result_width*(i+thread_y)+col;
            submatrix_B[thread_y][thread_x] = B[b_idx];
        }else {
            submatrix_B[thread_y][thread_x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int j=0;j<BLOCK_SIZE;j++) {
            sum += submatrix_A[thread_y][j]*submatrix_B[j][thread_x];
        }

        __syncthreads();

    }

    if (col < result_width && row < result_height) {
        int c_idx = result_width*row+col;
        C[c_idx] = alpha*sum+beta*C[c_idx];
    }

}
