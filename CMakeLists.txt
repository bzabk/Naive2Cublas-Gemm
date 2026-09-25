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



add_library(test_utils STATIC tests/test_utils.cpp)
target_include_directories(test_utils PUBLIC tests)

set(TEST_TARGETS
    naive_gemm_test
    gmem_coalescing_gemm_test
    smem_caching_gemm_test
    blocking_1d_gemm_test
    blocking_2d_gemm_test
    vectorized_mem_access_gemm_test)

include(GoogleTest)
foreach(TEST_NAME ${TEST_TARGETS})
    add_executable(${TEST_NAME} tests/${TEST_NAME}.cpp)
    target_link_libraries(${TEST_NAME} PRIVATE naive2cublas-gemm test_utils gtest_main)
    gtest_discover_tests(${TEST_NAME})
endforeach()






