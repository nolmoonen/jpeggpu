cmake_minimum_required(VERSION 3.18)
project(jpeggpu LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# TODO make all lambdas functors
set(CMAKE_CUDA_FLAGS "$CMAKE_CUDA_FLAGS --extended-lambda")

message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
add_library(jpeggpu STATIC
    "${SOURCE_DIR}/jpeggpu.cpp"
    "${SOURCE_DIR}/reader.cpp"
    "${SOURCE_DIR}/idct.cu"
    "${SOURCE_DIR}/decode_gpu.cu"
    "${SOURCE_DIR}/destuff.cu"
    "${SOURCE_DIR}/convert.cu"
    "${SOURCE_DIR}/decoder.cpp")
target_include_directories(jpeggpu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(jpeggpu PUBLIC CUDA::cudart_static)

add_executable(example_tool example_tool.c)
target_link_libraries(example_tool PRIVATE jpeggpu)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE jpeggpu CUDA::nvjpeg)
