cmake_minimum_required(VERSION 3.30)

project(naive2cublas-gemm LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

add_library(naive2cublas-gemm STATIC
        src/naive.cu
        src/gmem_coalescing.cu
        src/smem_caching.cu
        src/blocking_1d.cu
        src/blocking_2d.cu
        src/vectorize_mem_access.cu)

target_include_directories(naive2cublas-gemm PUBLIC
        include)

set_target_properties(naive2cublas-gemm PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)


add_executable(gemm_test
        tests/naive_gemm_test.cpp
        tests/gmem_coalescing_gemm_test.cpp
        tests/smem_caching_gemm_test.cpp
        tests/blocking_1d_gemm_test.cpp
        tests/blocking_2d_gemm_test.cpp
        tests/vectorized_mem_access_gemm_test.cpp)

target_link_libraries(gemm_test PRIVATE naive2cublas-gemm gtest_main)

include(GoogleTest)
gtest_discover_tests(gemm_test)





