#include <gemm_kernels.cuh>
#define BLOCK_SIZE 16
#define THREAD_MULTIPLIER_Y 4
#define THREAD_MULTIPLIER_X 4

__global__ void gemm::kernel::thread_tiling_2d(float const* A,float const* B, float* C,
                                        float const alpha, float const beta,
                                        bool const transA, bool const transB,
                                        int const result_height, int const result_width, int const common_dim) {



    int const tileCol = threadIdx.x;
    int const tileRow = threadIdx.y;

    int const row = blockIdx.y*blockDim.y*THREAD_MULTIPLIER_X+threadIdx.y;
    int const col = blockIdx.x*blockDim.x*THREAD_MULTIPLIER_Y+threadIdx.x;

    __shared__ float A_shared_block[THREAD_MULTIPLIER_X*BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared_block[BLOCK_SIZE][BLOCK_SIZE*THREAD_MULTIPLIER_Y];

    float regA[THREAD_MULTIPLIER_X] = {0.0f};
    float regB[THREAD_MULTIPLIER_Y] = {0.0f};
    float local_results[THREAD_MULTIPLIER_X][THREAD_MULTIPLIER_Y] = {0.0f};

    for (int k=0;k<common_dim;k+=BLOCK_SIZE) {


        for (int i=0;i<THREAD_MULTIPLIER_X;i++) {
            int local_row = row + i*BLOCK_SIZE;
            if (local_row< result_height &&  k+tileCol<common_dim) {
                int a_idx = transA ? result_height*(k+tileCol)+local_row : common_dim*local_row+k+tileCol;
                A_shared_block[tileRow+i*BLOCK_SIZE][tileCol] = A[a_idx];
            }else {
                A_shared_block[tileRow+i*BLOCK_SIZE][tileCol] = 0.0f;
            }
        }

        for (int i=0;i<THREAD_MULTIPLIER_Y;i++) {
            int local_col = col+i*BLOCK_SIZE;
            if ((k+tileRow)<common_dim && local_col<result_width) {
                int b_idx = transB ? common_dim*local_col+k+tileRow : result_width*(k+tileRow)+local_col;
                B_shared_block[tileRow][tileCol+i*BLOCK_SIZE] = B[b_idx];
            }else {
                B_shared_block[tileRow][tileCol+i*BLOCK_SIZE] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk=0;kk<BLOCK_SIZE;kk++) {

            #pragma unroll
            for (int m=0;m<THREAD_MULTIPLIER_X;m++) {
                regA[m] = A_shared_block[tileRow+m*BLOCK_SIZE][kk];
            }

            #pragma unroll
            for (int n=0;n<THREAD_MULTIPLIER_Y;n++) {
                regB[n] = B_shared_block[kk][tileCol+n*BLOCK_SIZE];
            }

            #pragma unroll
            for (int m=0;m<THREAD_MULTIPLIER_X;m++) {
                #pragma unroll
                for (int n=0;n<THREAD_MULTIPLIER_Y;n++) {
                    local_results[m][n] += regA[m]*regB[n];
                }
            }
        }

        __syncthreads();

    }

    #pragma unroll
    for (int m=0;m<THREAD_MULTIPLIER_X;m++) {

        #pragma unroll
        for (int n=0;n<THREAD_MULTIPLIER_Y;n++) {
            int out_row = row+m*BLOCK_SIZE;
            int out_col = col+n*BLOCK_SIZE;
            if (out_row<result_height && out_col<result_width) {
                int c_idx = result_width*out_row+out_col;
                C[c_idx] = alpha*local_results[m][n]+beta*C[c_idx];
            }
        }
    }
}
