#include <gemm_kernels.cuh>
#define BLOCK_SIZE 16
#define THREAD_MULTIPLIER_Y 4

__global__ void gemm::kernel::thread_tiling_1d(float const *A, float const *B, float *C,
    float const alpha, float const beta, bool const transA,
    bool const transB, int const result_height, int const result_width, int const common_dim) {

    int const thread_x = threadIdx.x;
    int const thread_y = threadIdx.y;

    int const col = blockDim.x * blockIdx.x + threadIdx.x;
    int const row = blockDim.y * (blockIdx.y*THREAD_MULTIPLIER_Y) + threadIdx.y;

    __shared__ float submatrix_A[BLOCK_SIZE*THREAD_MULTIPLIER_Y][BLOCK_SIZE];
    __shared__ float submatrix_B[BLOCK_SIZE][BLOCK_SIZE];

    float registers[THREAD_MULTIPLIER_Y] = {0.0f};

    for (int i=0;i<common_dim;i+=BLOCK_SIZE) {

        for (int j=0;j<THREAD_MULTIPLIER_Y;j++) {
            if ((row+j*BLOCK_SIZE)<result_height && (i+thread_x)<common_dim) {
                int a_idx = transA ? result_height*(i+thread_x) + row+j*BLOCK_SIZE : common_dim*(row+j*BLOCK_SIZE)+i+thread_x;
                submatrix_A[thread_y+j*BLOCK_SIZE][thread_x] = A[a_idx];
            }else {
                submatrix_A[thread_y+j*BLOCK_SIZE][thread_x] = 0.0f;
            }
        }


        if ((i+thread_y)<common_dim && col < result_width) {
            int b_idx = transB ? common_dim*col+(i+thread_y) : result_width*(i+thread_y)+col;
            submatrix_B[thread_y][thread_x] = B[b_idx];
        }else {
            submatrix_B[thread_y][thread_x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int j=0;j<BLOCK_SIZE;j++) {
            float val_from_submatrix_b = submatrix_B[j][thread_x];
            #pragma unroll
            for (int k=0;k<THREAD_MULTIPLIER_Y;k++) {
                registers[k] += submatrix_A[k*BLOCK_SIZE+thread_y][j]*val_from_submatrix_b;
            }
        }

        __syncthreads();
    }


    #pragma unroll
    for (int i=0;i<THREAD_MULTIPLIER_Y;i++) {
        if (col < result_width && row+i*BLOCK_SIZE < result_height) {
            C[result_width*(row+i*BLOCK_SIZE)+col] = alpha*registers[i]+beta*C[result_width*(row+i*BLOCK_SIZE)+col];
        }
    }

}
