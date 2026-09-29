#pragma once
#include <cuda_runtime.h>

namespace gemm::kernel {

    __global__ void naive(float const* A,float const* B, float* C,
                                    float alpha, float beta,
                                    bool transA, bool transB,
                                    int result_height,int result_width,int common_dim);

    __global__ void shared_tiling(float const* A,float const* B, float* C,
                                        float alpha, float beta,
                                        bool transA, bool transB,
                                        int result_height,int result_width,int common_dim);


    __global__ void thread_tiling_1d(float const* A,float const* B, float* C,
                                        float alpha, float beta,
                                        bool transA, bool transB,
                                        int result_height,int result_width,int common_dim);

    __global__ void thread_tiling_2d(float const* A,float const* B, float* C,
                                        float alpha, float beta,
                                        bool transA, bool transB,
                                        int result_height,int result_width,int common_dim);

    __global__ void vectorized_access(float const* A,float const* B, float* C,
                                        float alpha, float beta,
                                        bool transA, bool transB,
                                        int result_height,int result_width,int common_dim);

}


