#include <cuda_runtime.h>
#include <functional>
#include <gemm_kernels.cuh>
#include <array>
#include <cublas_api.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
constexpr int BLOCK_SIZE = 16;

constexpr std::array<int, 7> MATRIX_SIZES = {64, 128, 256, 512, 1024, 2048, 4096};

constexpr int WARMUP_SIZE = 5;

constexpr float ALPHA = 1.0f;
constexpr float BETA = 0.0f;
constexpr bool TRANSA = false;
constexpr bool TRANSB = false;

struct kernel_implementation {
    const char *kernel_name;
    std::function<void(float const *A, float const *B, float *C,
                       float alpha, float beta,
                       bool transA, bool transB,
                       int result_height, int result_width, int common_dim)> launch;
};


int main() {
    cublasHandle_t handle = nullptr;

    cublasCreate(&handle);

    std::array<kernel_implementation, 6> kernels2profile = {
        {
            {
                "naive",
                [](const float *d_a, const float *d_b, float *d_c, float alpha, float beta, bool transA, bool transB,
                   int M,
                   int N, int K) -> void {
                    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
                    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

                    gemm::kernel::naive<<<grid,threads_per_block>>
                            >(d_a, d_b, d_c, alpha, beta, transA, transB, M, N, K);
                }
            },
            {
                "shared_tilling",
                [](const float *d_a, const float *d_b, float *d_c, float alpha, float beta, bool transA, bool transB,
                   int M,
                   int N, int K) -> void {
                    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
                    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

                    gemm::kernel::shared_tiling<<<grid,threads_per_block>>>(
                        d_a, d_b, d_c, alpha, beta, transA, transB, M, N, K);
                }
            },
            {
                "thread_tiling_1d",
                [](const float *d_a, const float *d_b, float *d_c, float alpha, float beta, bool transA, bool transB,
                   int M,
                   int N, int K) -> void {
                    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
                    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

                    gemm::kernel::thread_tiling_1d<<<grid,threads_per_block>>>(
                        d_a, d_b, d_c, alpha, beta, transA, transB, M, N, K);
                }
            },
            {
                "thread_tiling_2d",
                [](const float *d_a, const float *d_b, float *d_c, float alpha, float beta, bool transA, bool transB,
                   int M,
                   int N, int K) -> void {
                    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
                    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

                    gemm::kernel::thread_tiling_2d<<<grid,threads_per_block>>>(
                        d_a, d_b, d_c, alpha, beta, transA, transB, M, N, K);
                }
            },
            {
                "vectorized_mem_access",
                [](const float *d_a, const float *d_b, float *d_c, float alpha, float beta, bool transA, bool transB,
                   int M,
                   int N, int K) -> void {
                    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
                    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

                    gemm::kernel::vectorized_access<<<grid,threads_per_block>>>(
                        d_a, d_b, d_c, alpha, beta, transA, transB, M, N, K);
                }
            },
            {

                "cublas",
                [&](const float *d_a, const float *d_b, float *d_c, float alpha, float beta, bool transA,
                    bool transB,
                    int M,
                    int N, int K) -> void {
                    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
                    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;

                    int ldb = transB ? K : N;
                    int lda = transA ? M : K;

                    cublasSgemm(handle, opB, opA,
                                N, M, K,
                                &alpha,
                                d_b, ldb,
                                d_a, lda,
                                &beta,
                                d_c, N);
                }
            }


        }

    };


    for (const auto &matrix_size: MATRIX_SIZES) {
        int M = matrix_size;
        int N = matrix_size;
        int K = matrix_size;

        size_t const no_elements = M * N;
        size_t const size_in_bytes = no_elements * sizeof(float);

        std::vector<float> host_a(no_elements, 1.0f);
        std::vector<float> host_b(no_elements, 2.0f);
        std::vector<float> host_c(no_elements, 0.0f);

        float *dev_a = nullptr;
        float *dev_b = nullptr;
        float *dev_c = nullptr;

        cudaMalloc(&dev_a, size_in_bytes);
        cudaMalloc(&dev_b, size_in_bytes);
        cudaMalloc(&dev_c, size_in_bytes);

        cudaMemcpy(dev_a, host_a.data(), size_in_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, host_b.data(), size_in_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_c, host_c.data(), size_in_bytes, cudaMemcpyHostToDevice);


        for (const auto &kernel: kernels2profile) {
            auto kernel_run = [&]() { kernel.launch(dev_a, dev_b, dev_c, ALPHA, BETA, TRANSA, TRANSB, M, N, K); };

            for (int i = 0; i < WARMUP_SIZE; i++) {
                kernel_run();
            }

            cudaDeviceSynchronize();

            cudaProfilerStart();

            kernel_run();
            cudaDeviceSynchronize();

            cudaProfilerStop();
        }
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);

    }
    cublasDestroy(handle);


    return 0;
}
