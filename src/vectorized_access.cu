#include <gemm_kernels.cuh>
constexpr int BLOCK_SIZE = 16;
constexpr int THREAD_MULTIPLIER = 4;

__global__ void gemm::kernel::vectorized_access(float const *A, float const *B, float *C,
    float const alpha, float const beta, bool const transA, bool const transB,
    int const result_height, int const result_width, int const common_dim) {

    int const tx = threadIdx.x;
    int const ty = threadIdx.y;

    int const id_within_thread_block = blockDim.x*ty+tx;

    int const row = blockIdx.y*blockDim.y*THREAD_MULTIPLIER+threadIdx.y;
    int const col = blockIdx.x*blockDim.x*THREAD_MULTIPLIER+threadIdx.x;

    __shared__ float A_shared_block[BLOCK_SIZE*THREAD_MULTIPLIER][BLOCK_SIZE];
    __shared__ float B_shared_block[BLOCK_SIZE][BLOCK_SIZE*THREAD_MULTIPLIER];

    float regA[THREAD_MULTIPLIER] = {0.0f};
    float regB[THREAD_MULTIPLIER] = {0.0f};
    float local_results[THREAD_MULTIPLIER][THREAD_MULTIPLIER] = {0.0f};


    for (int k=0;k<common_dim;k+=BLOCK_SIZE) {

        int A_shared_block_load_row = id_within_thread_block / 4;
        int A_shared_block_load_col = id_within_thread_block % 4;

        int a_global_row = blockIdx.y*blockDim.y*THREAD_MULTIPLIER+A_shared_block_load_row;
        int a_global_col = k+4*A_shared_block_load_col;

        if(transA) {
            #pragma unroll
            for(int j=0;j<4;j++) {
                if (a_global_col+j<common_dim && a_global_row<result_height) {
                    int a_idx = result_height*(a_global_col+j)+a_global_row;
                    A_shared_block[A_shared_block_load_row][A_shared_block_load_col*4+j] = A[a_idx];
                }else {
                    A_shared_block[A_shared_block_load_row][A_shared_block_load_col] = 0.0f;
                }
            }
        }else {
            float4* A_shared_block_vec = reinterpret_cast<float4*>(&A_shared_block[A_shared_block_load_row][A_shared_block_load_col*4]);

            if (a_global_col < common_dim && a_global_row <result_height) {
                int a_idx = a_global_row*common_dim + a_global_col;

                const float4* A_global_vec = reinterpret_cast<const float4*>(&A[a_idx]);

                *A_shared_block_vec = *A_global_vec;
            }else {
                *A_shared_block_vec = make_float4(0.0f,0.0f,0.0f,0.0f);
            }
        }



        int B_shared_block_load_row = id_within_thread_block / 16;
        int B_shared_block_load_col = id_within_thread_block % 16;

        int b_global_row = k+B_shared_block_load_row;
        int b_global_col = blockIdx.x*blockDim.x*THREAD_MULTIPLIER+B_shared_block_load_col*4;

        if(transB) {
            #pragma unroll
            for(int j=0;j<4;j++) {
                if (b_global_row < common_dim && (b_global_col+j)<result_width) {
                    int b_idx = common_dim*(b_global_col+j)+b_global_row;
                    B_shared_block[B_shared_block_load_row][B_shared_block_load_col*4+j] = B[b_idx];
                }else {
                    B_shared_block[B_shared_block_load_row][B_shared_block_load_col] = 0.0f;
                }
            }
        }else {
            float4* B_shared_block_vec = reinterpret_cast<float4*>(&B_shared_block[B_shared_block_load_row][B_shared_block_load_col*4]);

            if (b_global_row < common_dim && b_global_col < result_width) {
                int b_idx =  b_global_row*result_width + b_global_col;
                const float4* B_global_vec = reinterpret_cast<const float4*>(&B[b_idx]);

                *B_shared_block_vec = *B_global_vec;
            }else {
                *B_shared_block_vec = make_float4(0.0f,0.0f,0.0f,0.0f);
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

