#include <cuda_runtime.h>

namespace gemm::kernel {
    template<bool const transA, bool const transB>
    __global__ void naive(
        float const * __restrict__ A,
        float const * __restrict__ B,
        float * __restrict__ C,
        float const alpha, float const beta,
        int const M,
        int const N,
        int const K) {
        int const row = blockIdx.y * blockDim.y + threadIdx.y;
        int const col = blockIdx.x * blockDim.x + threadIdx.x;

        float local_result = 0.0f;
        if (col < N && row < M) {
            int const c_idx = N * row + col;
            for (int k = 0; k < K; k++) {
                int const a_idx = transA ? M * k + row : K * row + k;
                int const b_idx = transB ? K * col + k : N * k + col;


                local_result += A[a_idx] * B[b_idx];
            }
            C[c_idx] = alpha * local_result + beta * C[c_idx];
        }
    }
}
