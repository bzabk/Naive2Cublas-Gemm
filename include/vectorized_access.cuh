#pragma once
#include <cuda_runtime.h>


namespace gemm::kernel {
    template<int const BM, int const BN, int const BK, int const TM, int const TN, bool const transA, bool const transB>
    __global__ void vectorized_access(
        float const * __restrict__ A,
        float const * __restrict__ B,
        float * __restrict__ C,
        float const alpha, float const beta,
        int const M,
        int const N,
        int const K) {
        int const tx = threadIdx.x;
        int const ty = threadIdx.y;

        int const id_within_thread_block = blockDim.x * ty + tx;

        int const c_row = blockIdx.y * BM;
        int const c_col = blockIdx.x * BN;

        __shared__ float A_shared[BM][BK];
        __shared__ float B_shared[BK][BN];

        float regA[TM] = {0.0f};
        float regB[TN] = {0.0f};

        float local_results[TM][TN] = {0.0f};

        int const A_shared_float4_row = id_within_thread_block / (BK / 4);
        int const A_shared_float4_col = id_within_thread_block % (BK / 4) * 4;

        int const B_shared_float4_row = id_within_thread_block / (BN / 4);
        int const B_shared_float4_col = id_within_thread_block % (BN / 4) * 4;


        for (int k = 0; k < K; k += BK) {
            int a_matrix_global_idx = (c_row + A_shared_float4_row) * K + (k + A_shared_float4_col);

            float4 float4_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (c_row + A_shared_float4_row < M && k + A_shared_float4_col < K) {
                float4_vec = reinterpret_cast<const float4 *>(&A[a_matrix_global_idx])[0];
            }
            reinterpret_cast<float4 *>(&A_shared[A_shared_float4_row][A_shared_float4_col])[0] = float4_vec;


            int b_matrix_global_idx = (k + B_shared_float4_row) * N + (B_shared_float4_col + c_col);

            float4_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (k + B_shared_float4_row < K && c_col + B_shared_float4_col < N) {
                float4_vec = reinterpret_cast<const float4 *>(&B[b_matrix_global_idx])[0];
            }
            reinterpret_cast<float4 *>(&B_shared[B_shared_float4_row][B_shared_float4_col])[0] = float4_vec;

            __syncthreads();

            #pragma unroll
            for (int bk = 0; bk < BK; bk++) {
                #pragma unroll
                for (int i = 0; i < TM; i++) {
                    regA[i] = A_shared[ty * TM + i][bk];
                }

                #pragma unroll
                for (int i = 0; i < TN; i++) {
                    regB[i] = B_shared[bk][TN * tx + i];
                }

                #pragma unroll
                for (int i = 0; i < TM; i++) {
                    #pragma unroll
                    for (int j = 0; j < TN; j++) {
                        local_results[i][j] += regA[i] * regB[j];
                    }
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                if (c_row + ty * TM + i < M && c_col + tx * TN + j < N) {
                    C[N * (c_row + ty * TM + i) + (c_col + tx * TN + j)] =
                            alpha * local_results[i][j] + beta * C[
                                N * (c_row + ty * TM + i) + (c_col + tx * TN + j)];
                }
            }
        }
    }
}
