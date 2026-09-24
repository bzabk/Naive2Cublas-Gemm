#include <gemm_kernels.cuh>

__global__ void gemm::kernel::vectorized_mem_access(float const *A, float const *B, float *C,
    float const alfa, float const beta, bool const transA, bool const transB,
    int const result_height, int const result_width, int const common_dim) {

}
