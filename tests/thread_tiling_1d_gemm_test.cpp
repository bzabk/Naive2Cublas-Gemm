#include <gemm_kernels.cuh>
#include <gtest/gtest.h>
#include <test_utils.h>

#define BLOCK_SIZE 16
#define THREAD_MULTIPLIER_Y 4

TEST_F(GemmTest, ThreadTiling1D_Square_NN) {

    int test_idx = 0;
    Scenario& scenario = scenarios[test_idx];


    dim3 thread_per_block(BLOCK_SIZE,BLOCK_SIZE);
    dim3 blocks((scenario.N+BLOCK_SIZE-1)/(BLOCK_SIZE),
        (scenario.M+BLOCK_SIZE*THREAD_MULTIPLIER_Y-1)/(BLOCK_SIZE*THREAD_MULTIPLIER_Y));



    gemm::kernel::thread_tiling_1d<<<blocks,thread_per_block>>>(scenario.dev_a,
                scenario.dev_b, scenario.dev_c,
                scenario.alpha, scenario.beta,
                scenario.transA, scenario.transB, scenario.M, scenario.N, scenario.K);

    cudaDeviceSynchronize();

    size_t size_c = scenario.M*scenario.N*sizeof(float);
    cudaMemcpy(scenario.host_c,scenario.dev_c,size_c,cudaMemcpyDeviceToHost);

    for (int i=0;i<scenario.M*scenario.N; i++) {
        EXPECT_NEAR(scenario.host_c[i],scenario.host_cublas_c[i], 1e-3);
    }

}

TEST_F(GemmTest, ThreadTiling1D_Square_TN) {

    int test_idx = 1;
    Scenario& scenario = scenarios[test_idx];


    dim3 thread_per_block(BLOCK_SIZE,BLOCK_SIZE);
    dim3 blocks((scenario.N+BLOCK_SIZE-1)/(BLOCK_SIZE),
        (scenario.M+BLOCK_SIZE*THREAD_MULTIPLIER_Y-1)/(BLOCK_SIZE*THREAD_MULTIPLIER_Y));



    gemm::kernel::thread_tiling_1d<<<blocks,thread_per_block>>>(scenario.dev_a,
                scenario.dev_b, scenario.dev_c,
                scenario.alpha, scenario.beta,
                scenario.transA, scenario.transB, scenario.M, scenario.N, scenario.K);

    cudaDeviceSynchronize();

    size_t size_c = scenario.M*scenario.N*sizeof(float);
    cudaMemcpy(scenario.host_c,scenario.dev_c,size_c,cudaMemcpyDeviceToHost);

    for (int i=0;i<scenario.M*scenario.N; i++) {
        EXPECT_NEAR(scenario.host_c[i],scenario.host_cublas_c[i], 1e-3);
    }

}

TEST_F(GemmTest, ThreadTiling1D_Square_NT) {

    int test_idx = 2;
    Scenario& scenario = scenarios[test_idx];


    dim3 thread_per_block(BLOCK_SIZE,BLOCK_SIZE);
    dim3 blocks((scenario.N+BLOCK_SIZE-1)/(BLOCK_SIZE),
        (scenario.M+BLOCK_SIZE*THREAD_MULTIPLIER_Y-1)/(BLOCK_SIZE*THREAD_MULTIPLIER_Y));



    gemm::kernel::thread_tiling_1d<<<blocks,thread_per_block>>>(scenario.dev_a,
                scenario.dev_b, scenario.dev_c,
                scenario.alpha, scenario.beta,
                scenario.transA, scenario.transB, scenario.M, scenario.N, scenario.K);

    cudaDeviceSynchronize();

    size_t size_c = scenario.M*scenario.N*sizeof(float);
    cudaMemcpy(scenario.host_c,scenario.dev_c,size_c,cudaMemcpyDeviceToHost);

    for (int i=0;i<scenario.M*scenario.N; i++) {
        EXPECT_NEAR(scenario.host_c[i],scenario.host_cublas_c[i], 1e-3);
    }

}

TEST_F(GemmTest, ThreadTiling1D_Square_TT) {

    int test_idx = 3;
    Scenario& scenario = scenarios[test_idx];


    dim3 thread_per_block(BLOCK_SIZE,BLOCK_SIZE);
    dim3 blocks((scenario.N+BLOCK_SIZE-1)/(BLOCK_SIZE),
        (scenario.M+BLOCK_SIZE*THREAD_MULTIPLIER_Y-1)/(BLOCK_SIZE*THREAD_MULTIPLIER_Y));



    gemm::kernel::thread_tiling_1d<<<blocks,thread_per_block>>>(scenario.dev_a,
                scenario.dev_b, scenario.dev_c,
                scenario.alpha, scenario.beta,
                scenario.transA, scenario.transB, scenario.M, scenario.N, scenario.K);

    cudaDeviceSynchronize();

    size_t size_c = scenario.M*scenario.N*sizeof(float);
    cudaMemcpy(scenario.host_c,scenario.dev_c,size_c,cudaMemcpyDeviceToHost);

    for (int i=0;i<scenario.M*scenario.N; i++) {
        EXPECT_NEAR(scenario.host_c[i],scenario.host_cublas_c[i], 1e-3);
    }

}

TEST_F(GemmTest, ThreadTiling1D_SingleRow_M1) {

    int test_idx = 4;
    Scenario& scenario = scenarios[test_idx];


    dim3 thread_per_block(BLOCK_SIZE,BLOCK_SIZE);
    dim3 blocks((scenario.N+BLOCK_SIZE-1)/(BLOCK_SIZE),
        (scenario.M+BLOCK_SIZE*THREAD_MULTIPLIER_Y-1)/(BLOCK_SIZE*THREAD_MULTIPLIER_Y));



    gemm::kernel::thread_tiling_1d<<<blocks,thread_per_block>>>(scenario.dev_a,
                scenario.dev_b, scenario.dev_c,
                scenario.alpha, scenario.beta,
                scenario.transA, scenario.transB, scenario.M, scenario.N, scenario.K);

    cudaDeviceSynchronize();

    size_t size_c = scenario.M*scenario.N*sizeof(float);
    cudaMemcpy(scenario.host_c,scenario.dev_c,size_c,cudaMemcpyDeviceToHost);

    for (int i=0;i<scenario.M*scenario.N; i++) {
        EXPECT_NEAR(scenario.host_c[i],scenario.host_cublas_c[i], 1e-3);
    }

}

TEST_F(GemmTest, ThreadTiling1D_SingleColumn_N1) {

    int test_idx = 5;
    Scenario& scenario = scenarios[test_idx];


    dim3 thread_per_block(BLOCK_SIZE,BLOCK_SIZE);
    dim3 blocks((scenario.N+BLOCK_SIZE-1)/(BLOCK_SIZE),
        (scenario.M+BLOCK_SIZE*THREAD_MULTIPLIER_Y-1)/(BLOCK_SIZE*THREAD_MULTIPLIER_Y));



    gemm::kernel::thread_tiling_1d<<<blocks,thread_per_block>>>(scenario.dev_a,
                scenario.dev_b, scenario.dev_c,
                scenario.alpha, scenario.beta,
                scenario.transA, scenario.transB, scenario.M, scenario.N, scenario.K);

    cudaDeviceSynchronize();

    size_t size_c = scenario.M*scenario.N*sizeof(float);
    cudaMemcpy(scenario.host_c,scenario.dev_c,size_c,cudaMemcpyDeviceToHost);

    for (int i=0;i<scenario.M*scenario.N; i++) {
        EXPECT_NEAR(scenario.host_c[i],scenario.host_cublas_c[i], 1e-3);
    }

}

TEST_F(GemmTest, ThreadTiling1D_AlphaBetaScaling) {


    int test_idx = 6;
    Scenario& scenario = scenarios[test_idx];


    dim3 thread_per_block(BLOCK_SIZE,BLOCK_SIZE);
    dim3 blocks((scenario.N+BLOCK_SIZE-1)/(BLOCK_SIZE),
        (scenario.M+BLOCK_SIZE*THREAD_MULTIPLIER_Y-1)/(BLOCK_SIZE*THREAD_MULTIPLIER_Y));



    gemm::kernel::thread_tiling_1d<<<blocks,thread_per_block>>>(scenario.dev_a,
                scenario.dev_b, scenario.dev_c,
                scenario.alpha, scenario.beta,
                scenario.transA, scenario.transB, scenario.M, scenario.N, scenario.K);

    cudaDeviceSynchronize();

    size_t size_c = scenario.M*scenario.N*sizeof(float);
    cudaMemcpy(scenario.host_c,scenario.dev_c,size_c,cudaMemcpyDeviceToHost);

    for (int i=0;i<scenario.M*scenario.N; i++) {
        EXPECT_NEAR(scenario.host_c[i],scenario.host_cublas_c[i], 1e-3);
    }

}
