#include <gemm_kernels.cuh>
constexpr int BLOCK_SIZE = 16;
constexpr int THREAD_MULTIPLIER = 4;

__global__ void gemm::kernel::thread_tiling_2d(float const* __restrict__ A, float const* __restrict__ B, float* __restrict__ C,
    float const alpha, float const beta, bool const transA,
    bool const transB, int const result_height, int const result_width, int const common_dim) {



    int const tx = threadIdx.x;
    int const ty = threadIdx.y;

    int const row = blockIdx.y*blockDim.y*THREAD_MULTIPLIER+threadIdx.y;
    int const col = blockIdx.x*blockDim.x*THREAD_MULTIPLIER+threadIdx.x;

    __shared__ float A_shared_block[THREAD_MULTIPLIER*BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared_block[BLOCK_SIZE][BLOCK_SIZE*THREAD_MULTIPLIER];

    float regA[THREAD_MULTIPLIER] = {0.0f};
    float regB[THREAD_MULTIPLIER] = {0.0f};
    float local_results[THREAD_MULTIPLIER][THREAD_MULTIPLIER] = {0.0f};

    for (int k=0;k<common_dim;k+=BLOCK_SIZE) {


        for (int i=0;i<THREAD_MULTIPLIER;i++) {
            int local_row = row + i*BLOCK_SIZE;
            if (local_row< result_height &&  k+tx<common_dim) {
                int a_idx = transA ? result_height*(k+tx)+local_row : common_dim*local_row+k+tx;
                A_shared_block[ty+i*BLOCK_SIZE][tx] = A[a_idx];
            }else {
                A_shared_block[ty+i*BLOCK_SIZE][tx] = 0.0f;
            }
        }

        for (int i=0;i<THREAD_MULTIPLIER;i++) {
            int local_col = col+i*BLOCK_SIZE;
            if ((k+ty)<common_dim && local_col<result_width) {
                int b_idx = transB ? common_dim*local_col+k+ty : result_width*(k+ty)+local_col;
                B_shared_block[ty][tx+i*BLOCK_SIZE] = B[b_idx];
            }else {
                B_shared_block[ty][tx+i*BLOCK_SIZE] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk=0;kk<BLOCK_SIZE;kk++) {

            #pragma unroll
            for (int m=0;m<THREAD_MULTIPLIER;m++) {
                regA[m] = A_shared_block[ty+m*BLOCK_SIZE][kk];
            }

            #pragma unroll
            for (int n=0;n<THREAD_MULTIPLIER;n++) {
                regB[n] = B_shared_block[kk][tx+n*BLOCK_SIZE];
            }

            #pragma unroll
            for (int m=0;m<THREAD_MULTIPLIER;m++) {
                #pragma unroll
                for (int n=0;n<THREAD_MULTIPLIER;n++) {
                    local_results[m][n] += regA[m]*regB[n];
                }
            }
        }

        __syncthreads();

    }

    #pragma unroll
    for (int m=0;m<THREAD_MULTIPLIER;m++) {

        #pragma unroll
        for (int n=0;n<THREAD_MULTIPLIER;n++) {
            int out_row = row+m*BLOCK_SIZE;
            int out_col = col+n*BLOCK_SIZE;
            if (out_row<result_height && out_col<result_width) {
                int c_idx = result_width*out_row+out_col;
                C[c_idx] = alpha*local_results[m][n]+beta*C[c_idx];
            }
        }
    }
}


