#pragma once
#include <cuda_runtime.h>


namespace gemm::kernel {
    template<int const BM, int const BK, int const TM, bool const transA, bool const transB>
    __global__ void thread_tiling_1d(
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
        int const col = blockIdx.x * blockDim.x + threadIdx.x;

        __shared__ float A_shared_block[BM][BK];
        __shared__ float B_shared_block[BK][BK];

        float local_results[TM] = {0.0f};

        for (int k = 0; k < K; k += BK) {
            for (int i = 0; i < TM; i++) {
                int local_row = row + i * BK;
                if (local_row < M && (k + tx) < K) {
                    int a_idx = transA ? M * (k + tx) + local_row : K * local_row + k + tx;
                    A_shared_block[ty + i * BK][tx] = A[a_idx];
                } else {
                    A_shared_block[ty + i * BK][tx] = 0.0f;
                }
            }


            if ((k + ty) < K && col < N) {
                int b_idx = transB ? K * col + (k + ty) : N * (k + ty) + col;
                B_shared_block[ty][tx] = B[b_idx];
            } else {
                B_shared_block[ty][tx] = 0.0f;
            }

            __syncthreads();

            #pragma unroll
            for (int kk = 0; kk < BK; kk++) {
                float regB = B_shared_block[kk][tx];
                #pragma unroll
                for (int m = 0; m < TM; m++) {
                    local_results[m] += A_shared_block[ty + m * BK][kk] * regB;
                }
            }

            __syncthreads();
        }


        #pragma unroll
        for (int m = 0; m < TM; m++) {
            int out_row = row + m * BK;
            if (out_row < M && col < N) {
                int c_idx = N * out_row + col;
                C[c_idx] = alpha * local_results[m] + beta * C[c_idx];
            }
        }
    }
}
