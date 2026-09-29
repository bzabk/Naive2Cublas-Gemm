#include <gemm_kernels.cuh>
#define BLOCK_SIZE 16
#define THREAD_MULTIPLIER_Y 4
#define THREAD_MULTIPLIER_X 4

__global__ void gemm::kernel::thread_tiling_2d(float const *A, float const *B, float *C,
    float const alpha, float const beta, bool const transA, bool const transB,
    int const result_height, int const result_width, int const common_dim) {

    int col = blockIdx.x*blockDim.x*THREAD_MULTIPLIER_X + threadIdx.x;
    int row = blockIdx.y*blockDim.y*THREAD_MULTIPLIER_Y + threadIdx.y;

    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    __shared__ float submatrix_A[BLOCK_SIZE*THREAD_MULTIPLIER_Y][BLOCK_SIZE];
    __shared__ float submatrix_B[BLOCK_SIZE][BLOCK_SIZE*THREAD_MULTIPLIER_X];

    float accum[THREAD_MULTIPLIER_Y][THREAD_MULTIPLIER_X] = {0.0f};


    for (int i=0;i<common_dim;i+=BLOCK_SIZE) {

        for (int j=0;j<THREAD_MULTIPLIER_Y;j++) {
            if ((row+j*BLOCK_SIZE)<result_height && (i+thread_x)<common_dim) {
                int a_idx = transA ? result_height*(i+thread_x) + row+j*BLOCK_SIZE : common_dim*(row+j*BLOCK_SIZE)+i+thread_x;
                submatrix_A[thread_y+j*BLOCK_SIZE][thread_x] = A[a_idx];
            }else {
                submatrix_A[thread_y+j*BLOCK_SIZE][thread_x] = 0.0f;
            }
        }

        for (int j=0;j<THREAD_MULTIPLIER_X;j++) {
           if ((j*BLOCK_SIZE+col)<result_width && (i+thread_y)<common_dim) {
               int b_idx = transB ? common_dim*(j*BLOCK_SIZE+col)+i+thread_y : result_width*(i+thread_y)+j*BLOCK_SIZE+col;
               submatrix_B[thread_y][j*BLOCK_SIZE+thread_x] = B[b_idx];
           }else {
               submatrix_B[thread_y][j*BLOCK_SIZE+thread_x] = 0.0f;
           }
        }

        __syncthreads();
        #pragma unroll
        for (int l=0;l<BLOCK_SIZE;l++) {
            #pragma unroll
            for (int i=0;i<THREAD_MULTIPLIER_Y;i++) {
                #pragma unroll
                for (int j=0;j<THREAD_MULTIPLIER_X;j++) {
                    accum[i][j] += submatrix_A[i*BLOCK_SIZE+thread_y][l] * submatrix_B[l][j*BLOCK_SIZE+thread_x];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i=0;i<THREAD_MULTIPLIER_Y;i++) {
        #pragma unroll
        for (int j=0;j<THREAD_MULTIPLIER_X;j++) {
            if ((row+i*BLOCK_SIZE)<result_height && (col+j*BLOCK_SIZE)<result_width) {
                int c_idx = result_width*(row+i*BLOCK_SIZE)+(col+j*BLOCK_SIZE);
                C[c_idx] = alpha*accum[i][j]+beta*C[c_idx];
            }
        }
    }



}
