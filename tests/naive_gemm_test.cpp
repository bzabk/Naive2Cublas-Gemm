#include <gemm_kernels.cuh>
#include <gtest/gtest.h>
#include <test_utils.h>
#define BLOCK_SIZE 16


TEST_P(GemmTest,Naive) {

    Scenario scenario = GetParam();

    dim3 thread_per_block(BLOCK_SIZE,BLOCK_SIZE);
    dim3 blocks((scenario.N+BLOCK_SIZE-1)/BLOCK_SIZE,(scenario.M+BLOCK_SIZE-1)/BLOCK_SIZE);



    gemm::kernel::naive<<<blocks,thread_per_block>>>(dev_a,
                dev_b, dev_c,
                scenario.alpha, scenario.beta,
                scenario.transA, scenario.transB, scenario.M, scenario.N, scenario.K);

    CUDA_EXPECT(cudaGetLastError());
    CUDA_EXPECT(cudaDeviceSynchronize());

    size_t size_c = scenario.M*scenario.N*sizeof(float);
    CUDA_EXPECT(cudaMemcpy(host_c,dev_c,size_c,cudaMemcpyDeviceToHost));

    for (int i=0;i<scenario.M*scenario.N; i++) {
        EXPECT_NEAR(host_c[i],host_cublas_c[i], 1e-3);
    }
}


INSTANTIATE_TEST_SUITE_P(
    GemmTestScenario,
    GemmTest,
    ::testing::Values(
    // M N K alpha beta transA transB
    Scenario{1024, 1024, 1024, 1.0f, 0.0f, false, false},
    Scenario{1024, 1024, 1024, 1.0f, 0.0f, true,  false},
    Scenario{1024, 1024, 1024, 1.0f, 0.0f, false, true},
    Scenario{1024, 1024, 1024, 1.0f, 0.0f, true,  true},
    Scenario{1,    1024, 1024, 1.0f, 0.0f, false, false},
    Scenario{1024, 1,    1024, 1.0f, 0.0f, false, false},
    Scenario{1024, 1024, 1024, 2.5f, 1.5f, false, false}
    )
);