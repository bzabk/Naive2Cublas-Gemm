#include <gemm_kernels.cuh>

__global__ void gemm::kernel::thread_tiling_1d(float const *A, float const *B, float *C,
    float const alpha, float const beta, bool const transA,
    bool const transB, int const result_height, int const result_width, int const common_dim) {

}
