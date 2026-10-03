#pragma once
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cublas_v2.h>

// for critical section
#define CUDA_ASSERT(call) do { \
    cudaError_t err = call; \
        ASSERT_EQ(err, cudaSuccess) \
            << "CRITICAL CUDA Error at " << __FILE__ << ":" << __LINE__ << "\n" \
        << "Call: " << #call << "\n" \
    << "Error: " << cudaGetErrorString(err); \
} while(0)

// for tests
#define CUDA_EXPECT(call) do { \
    cudaError_t err = call; \
        EXPECT_EQ(err, cudaSuccess) \
            << "CUDA Error at " << __FILE__ << ":" << __LINE__ << "\n" \
        << "Call: " << #call << "\n" \
    << "Error: " << cudaGetErrorString(err); \
} while(0)


struct Scenario {
    int M;
    int N;
    int K;
    float alpha;
    float beta;
    bool transA;
    bool transB;
};

class GemmTest : public::testing::TestWithParam<Scenario> {
protected:
    cublasHandle_t handle = nullptr;

    float* host_a = nullptr;
    float* host_b = nullptr;
    float* host_c = nullptr;
    float* host_cublas_c = nullptr;

    float* dev_a = nullptr;
    float* dev_b = nullptr;
    float* dev_c = nullptr;
    float* dev_cublas_c = nullptr;

    void SetUp() override {

        Scenario scenario = GetParam();
        
        cublasCreate(&handle);

        int common_dim = scenario.K;

        int count_a = scenario.M*common_dim;
        int count_b = common_dim*scenario.N;
        int count_c = scenario.M*scenario.N;

        size_t size_a = count_a*sizeof(float);
        size_t size_b = count_b*sizeof(float);
        size_t size_c = count_c*sizeof(float);

        host_a = new float[count_a];
        host_b = new float[count_b];
        host_c = new float[count_c];

        host_cublas_c = new float[count_c];

        std::generate(host_a, host_a + count_a, []() {
            return static_cast<float>(rand()) / RAND_MAX;
        });
        std::generate(host_b, host_b + count_b, []() {
            return static_cast<float>(rand()) / RAND_MAX;
        });
        std::fill(host_c, host_c + count_c, 0.0f);
        std::fill(host_cublas_c, host_cublas_c + count_c, 0.0f);

        CUDA_ASSERT(cudaMalloc(&dev_a, size_a));
        CUDA_ASSERT(cudaMalloc(&dev_b, size_b));
        CUDA_ASSERT(cudaMalloc(&dev_c, size_c));
        CUDA_ASSERT(cudaMalloc(&dev_cublas_c, size_c));

        CUDA_ASSERT(cudaMemcpy(dev_a,host_a,size_a,cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(dev_b,host_b,size_b,cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(dev_c,host_c,size_c,cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(dev_cublas_c,host_cublas_c,size_c,cudaMemcpyHostToDevice));

        cublasOperation_t opB = scenario.transB ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t opA = scenario.transA ? CUBLAS_OP_T : CUBLAS_OP_N;

        int ldb = scenario.transB ? scenario.K : scenario.N;
        int lda = scenario.transA ? scenario.M : scenario.K;

        cublasSgemm(handle, opB, opA,
                    scenario.N, scenario.M, scenario.K,
                    &scenario.alpha,
                    dev_b, ldb,
                    dev_a, lda,
                    &scenario.beta,
                    dev_cublas_c, scenario.N);

        CUDA_ASSERT(cudaMemcpy(host_cublas_c, dev_cublas_c, size_c, cudaMemcpyDeviceToHost));


    }

    void TearDown() override {

        delete[] host_a;
        delete[] host_b;
        delete[] host_c;
        delete[] host_cublas_c;


        cublasDestroy(handle);

        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        cudaFree(dev_cublas_c);
    }

};




