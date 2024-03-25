cmake_minimum_required(VERSION 3.18)
project(jpeggpu LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
add_library(jpeggpu STATIC
    "${SOURCE_DIR}/convert.cu"
    "${SOURCE_DIR}/jpeggpu.cpp"
    "${SOURCE_DIR}/decode_dc.cu"
    "${SOURCE_DIR}/decode_destuff.cu"
    "${SOURCE_DIR}/decode_huffman.cu"
    "${SOURCE_DIR}/decode_transpose.cu"
    "${SOURCE_DIR}/decoder.cpp"
    "${SOURCE_DIR}/defs.cpp"
    "${SOURCE_DIR}/idct.cu"
    "${SOURCE_DIR}/logger.cpp"
    "${SOURCE_DIR}/reader.cpp")
target_include_directories(jpeggpu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(jpeggpu PUBLIC CUDA::cudart_static)

add_executable(example_tool example_tool.c)
target_link_libraries(example_tool PRIVATE jpeggpu)

add_subdirectory(benchmark)
