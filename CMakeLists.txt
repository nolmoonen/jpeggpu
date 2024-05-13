# Copyright (c) 2023-2024 Nol Moonen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
    "${SOURCE_DIR}/jpeggpu.cpp"
    "${SOURCE_DIR}/decode_dc.cu"
    "${SOURCE_DIR}/decode_destuff.cu"
    "${SOURCE_DIR}/decode_huffman.cu"
    "${SOURCE_DIR}/decode_transpose.cu"
    "${SOURCE_DIR}/decoder.cpp"
    "${SOURCE_DIR}/defs.cpp"
    "${SOURCE_DIR}/idct.cu"
    "${SOURCE_DIR}/reader.cpp")
target_include_directories(jpeggpu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(jpeggpu PUBLIC CUDA::cudart_static)

add_executable(jpeggpu_example example_tool.c)
target_link_libraries(jpeggpu_example PRIVATE jpeggpu)

add_subdirectory(benchmark)
add_subdirectory(test)
