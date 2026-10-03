#include <gemm_kernels.cuh>
constexpr int BLOCK_SIZE = 16;


__global__ void gemm::kernel::shared_tiling(float const* __restrict__ A, float const* __restrict__ B, float* __restrict__ C,
    float const alpha, float const beta, bool const transA,
    bool const transB, int const result_height, int const result_width, int const common_dim) {


    int const tx = threadIdx.x;
    int const ty = threadIdx.y;

    int const row = blockIdx.y*blockDim.y + threadIdx.y;
    int const col = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ float A_shared_block[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared_block[BLOCK_SIZE][BLOCK_SIZE];

    float local_result = 0.0f;

    for (int k=0; k<common_dim;k+=BLOCK_SIZE) {

        if (row < result_height && (k+tx) < common_dim) {
            int a_idx = transA ? result_height*(k+tx)+row : common_dim*row+k+tx;
            A_shared_block[ty][tx] = A[a_idx];
        }else {
            A_shared_block[ty][tx] = 0.0f;
        }

        if (col<result_width && (k+ty)<common_dim) {
            int b_idx = transB ? common_dim*col+(k+ty) : result_width*(k+ty)+col;
            B_shared_block[ty][tx] = B[b_idx];
        }else {
            B_shared_block[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int kk=0;kk<BLOCK_SIZE;kk++) {
            local_result += A_shared_block[ty][kk]*B_shared_block[kk][tx];
        }

        __syncthreads();

    }

    if (col < result_width && row < result_height) {
        int c_idx = result_width*row+col;
        C[c_idx] = alpha*local_result+beta*C[c_idx];
    }

}
