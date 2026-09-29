cmake_minimum_required(VERSION 3.30)

project(naive2cublas-gemm LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

add_library(naive2cublas-gemm STATIC
        src/naive.cu
        src/shared_tiling.cu
        src/thread_tiling_1d.cu
        src/thread_tiling_2d.cu
        src/vectorized_access.cu)

target_include_directories(naive2cublas-gemm PUBLIC
        include)

set_target_properties(naive2cublas-gemm PROPERTIES
                      CUDA_SEPARABLE_COMPILATION OFF)



add_library(test_utils INTERFACE)
target_link_libraries(test_utils INTERFACE CUDA::cudart CUDA::cublas)
target_include_directories(test_utils INTERFACE tests)


set(TEST_TARGETS
    naive_gemm_test
    shared_tiling_gemm_test
    thread_tiling_1d_gemm_test
    thread_tiling_2d_gemm_test
    vectorized_access_gemm_test)

file(GLOB CUDA_RUNTIME_DLLS
     "${CUDAToolkit_BIN_DIR}/cudart64_*.dll"
     "${CUDAToolkit_BIN_DIR}/cublas64_*.dll"
     "${CUDAToolkit_BIN_DIR}/cublasLt64_*.dll"
     "${CUDAToolkit_BIN_DIR}/x64/cudart64_*.dll"
     "${CUDAToolkit_BIN_DIR}/x64/cublas64_*.dll"
     "${CUDAToolkit_BIN_DIR}/x64/cublasLt64_*.dll")

include(GoogleTest)
foreach(TEST_NAME ${TEST_TARGETS})
    set_source_files_properties(tests/${TEST_NAME}.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(${TEST_NAME} tests/${TEST_NAME}.cpp)
    target_link_libraries(${TEST_NAME} PRIVATE naive2cublas-gemm test_utils gtest_main)

    if(CUDA_RUNTIME_DLLS)
        add_custom_command(TARGET ${TEST_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${CUDA_RUNTIME_DLLS}
                    $<TARGET_FILE_DIR:${TEST_NAME}>)
    endif()

    gtest_discover_tests(${TEST_NAME})
endforeach()






