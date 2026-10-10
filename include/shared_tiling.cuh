#pragma once
#include <cuda_runtime.h>


namespace gemm::kernel {
    template<int const BK, bool const transA, bool const transB>
    __global__ void shared_tiling(
        float const * __restrict__ A,
        float const * __restrict__ B,
        float * __restrict__ C,
        float const alpha, float const beta,
        int const M,
        int const N,
        int const K) {
        int const tx = threadIdx.x;
        int const ty = threadIdx.y;

        int const row = blockIdx.y * blockDim.y + threadIdx.y;
        int const col = blockIdx.x * blockDim.x + threadIdx.x;

        __shared__ float A_shared_block[BK][BK];
        __shared__ float B_shared_block[BK][BK];

        float local_result = 0.0f;

        for (int k = 0; k < K; k += BK) {
            if (row < M && (k + tx) < K) {
                int a_idx = transA ? M * (k + tx) + row : K * row + k + tx;
                A_shared_block[ty][tx] = A[a_idx];
            } else {
                A_shared_block[ty][tx] = 0.0f;
            }

            if (col < N && (k + ty) < K) {
                int b_idx = transB ? K * col + (k + ty) : N * (k + ty) + col;
                B_shared_block[ty][tx] = B[b_idx];
            } else {
                B_shared_block[ty][tx] = 0.0f;
            }

            __syncthreads();

            #pragma unroll
            for (int kk = 0; kk < BK; kk++) {
                local_result += A_shared_block[ty][kk] * B_shared_block[kk][tx];
            }

            __syncthreads();
        }

        if (col < N && row < M) {
            int c_idx = N * row + col;
            C[c_idx] = alpha * local_result + beta * C[c_idx];
        }
    }
}
