#pragma once
#include <cuda_runtime.h>


namespace gemm::kernel {
    template<int const BM, int const BN, int const BK, int const TM, int const TN, bool const transA, bool const transB>
    __global__ void thread_tiling_2d(
        float const * __restrict__ A,
        float const * __restrict__ B,
        float * __restrict__ C,
        float const alpha, float const beta,
        int const M,
        int const N,
        int const K) {
        int const tx = threadIdx.x;
        int const ty = threadIdx.y;

        int const row = blockIdx.y * blockDim.y * TM + threadIdx.y;
        int const col = blockIdx.x * blockDim.x * TN + threadIdx.x;

        __shared__ float A_shared_block[BM][BK];
        __shared__ float B_shared_block[BK][BN];

        float regA[TM] = {0.0f};
        float regB[TN] = {0.0f};
        float local_results[TM][TN] = {0.0f};

        for (int k = 0; k < K; k += BK) {
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                int local_row = row + i * BK;
                if (local_row < M && k + tx < K) {
                    int a_idx = transA ? M * (k + tx) + local_row : K * local_row + k + tx;
                    A_shared_block[ty + i * BK][tx] = A[a_idx];
                } else {
                    A_shared_block[ty + i * BK][tx] = 0.0f;
                }
            }
            #pragma unroll
            for (int i = 0; i < TN; i++) {
                int local_col = col + i * BK;
                if ((k + ty) < K && local_col < N) {
                    int b_idx = transB ? K * local_col + k + ty : N * (k + ty) + local_col;
                    B_shared_block[ty][tx + i * BK] = B[b_idx];
                } else {
                    B_shared_block[ty][tx + i * BK] = 0.0f;
                }
            }

            __syncthreads();

            #pragma unroll
            for (int kk = 0; kk < BK; kk++) {
                #pragma unroll
                for (int m = 0; m < TM; m++) {
                    regA[m] = A_shared_block[ty + m * BK][kk];
                }

                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    regB[n] = B_shared_block[kk][tx + n * BK];
                }

                #pragma unroll
                for (int m = 0; m < TM; m++) {
                    #pragma unroll
                    for (int n = 0; n < TN; n++) {
                        local_results[m][n] += regA[m] * regB[n];
                    }
                }
            }

            __syncthreads();
        }

        #pragma unroll
        for (int m = 0; m < TM; m++) {
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                int out_row = row + m * BK;
                int out_col = col + n * BK;
                if (out_row < M && out_col < N) {
                    int c_idx = N * out_row + out_col;
                    C[c_idx] = alpha * local_results[m][n] + beta * C[c_idx];
                }
            }
        }
    }
}
