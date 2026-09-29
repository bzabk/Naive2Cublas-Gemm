#pragma once
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cublas_v2.h>

struct Scenario {
    int M;
    int N;
    int K;
    float alpha;
    float beta;
    bool transA;
    bool transB;

    float* host_a;
    float* host_b;
    float* host_c;
    float* host_cublas_c;

    float* dev_a;
    float* dev_b;
    float* dev_c;
    float* dev_cublas_c;
};



class GemmTest : public::testing::Test {
protected:

    inline static Scenario scenarios[7] = {
        {1024, 1024, 1024, 1.0f, 0.0f, false, false},
        {1024, 1024, 1024, 1.0f, 0.0f, true,  false},
        {1024, 1024, 1024, 1.0f, 0.0f, false, true },
        {1024, 1024, 1024, 1.0f, 0.0f, true,  true },
        {1,    1024, 1024, 1.0f, 0.0f, false, false},
        {1024, 1,    1024, 1.0f, 0.0f, false, false},
        {1024, 1024, 1024, 2.5f, 1.5f, false, false}
    };



    static void SetUpTestSuite() {

        cublasHandle_t handle;
        cublasCreate(&handle);


        for (int i = 0; i < 7; i++) {

            auto& scenario = scenarios[i];

            int common_dim = scenario.K;

            int count_a = scenario.M*common_dim;
            int count_b = common_dim*scenario.N;
            int count_c = scenario.M*scenario.N;

            size_t size_a = count_a*sizeof(float);
            size_t size_b = count_b*sizeof(float);
            size_t size_c = count_c*sizeof(float);

            scenario.host_a = new float[count_a];
            scenario.host_b = new float[count_b];
            scenario.host_c = new float[count_c];

            scenario.host_cublas_c = new float[count_c];

            std::generate(scenario.host_a, scenario.host_a + count_a, []() {
                return static_cast<float>(rand()) / RAND_MAX;
            });
            std::generate(scenario.host_b, scenario.host_b + count_b, []() {
                return static_cast<float>(rand()) / RAND_MAX;
            });
            std::fill(scenario.host_c, scenario.host_c + count_c, 0.0f);
            std::fill(scenario.host_cublas_c, scenario.host_cublas_c + count_c, 0.0f);

            cudaMalloc(&scenario.dev_a, size_a);
            cudaMalloc(&scenario.dev_b, size_b);
            cudaMalloc(&scenario.dev_c, size_c);
            cudaMalloc(&scenario.dev_cublas_c, size_c);

            cudaMemcpy(scenario.dev_a,scenario.host_a,size_a,cudaMemcpyHostToDevice);
            cudaMemcpy(scenario.dev_b,scenario.host_b,size_b,cudaMemcpyHostToDevice);
            cudaMemcpy(scenario.dev_c,scenario.host_c,size_c,cudaMemcpyHostToDevice);
            cudaMemcpy(scenario.dev_cublas_c,scenario.host_cublas_c,size_c,cudaMemcpyHostToDevice);

            cublasOperation_t opB = scenario.transB ? CUBLAS_OP_T : CUBLAS_OP_N;
            cublasOperation_t opA = scenario.transA ? CUBLAS_OP_T : CUBLAS_OP_N;

            int ldb = scenario.transB ? scenario.K : scenario.N;
            int lda = scenario.transA ? scenario.M : scenario.K;

            cublasSgemm(handle, opB, opA,
                        scenario.N, scenario.M, scenario.K,
                        &scenario.alpha,
                        scenario.dev_b, ldb,
                        scenario.dev_a, lda,
                        &scenario.beta,
                        scenario.dev_cublas_c, scenario.N);

            cudaMemcpy(scenario.host_cublas_c, scenario.dev_cublas_c, size_c, cudaMemcpyDeviceToHost);


        }

        cublasDestroy(handle);

    }

    static void TearDownTestSuite() {
        for (int i=0;i<7;i++) {
            delete[] scenarios[i].host_a;
            delete[] scenarios[i].host_b;
            delete[] scenarios[i].host_c;
            delete[] scenarios[i].host_cublas_c;

            cudaFree(scenarios[i].dev_a);
            cudaFree(scenarios[i].dev_b);
            cudaFree(scenarios[i].dev_c);
            cudaFree(scenarios[i].dev_cublas_c);
        }
    }

};
