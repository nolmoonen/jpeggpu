# Copyright (c) 2023-2024 Nol Moonen
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https:#www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.18) # 3.18 CMAKE_CUDA_ARCHITECTURES
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

add_subdirectory(benchmark)
add_subdirectory(example)
add_subdirectory(test)
