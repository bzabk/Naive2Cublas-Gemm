#pragma once

namespace gemm::kernel {

    __global__ void naive(float const* A,float const* B, float* C,
                                    float alfa, float beta,
                                    bool transA, bool transB,
                                    int result_height,int result_width,int common_dim);


    __global__ void gmem_coalescing(float const* A,float const* B, float* C,
                                        float alfa, float beta,
                                        bool transA, bool transB,
                                        int result_height,int result_width,int common_dim);

    __global__ void smem_caching(float const* A,float const* B, float* C,
                                        float alfa, float beta,
                                        bool transA, bool transB,
                                        int result_height,int result_width,int common_dim);


    __global__ void blocking_1d(float const* A,float const* B, float* C,
                                        float alfa, float beta,
                                        bool transA, bool transB,
                                        int result_height,int result_width,int common_dim);

    __global__ void blocking_2d(float const* A,float const* B, float* C,
                                        float alfa, float beta,
                                        bool transA, bool transB,
                                        int result_height,int result_width,int common_dim);

    __global__ void vectorized_mem_access(float const* A,float const* B, float* C,
                                        float alfa, float beta,
                                        bool transA, bool transB,
                                        int result_height,int result_width,int common_dim);

}


