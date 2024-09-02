// Copyright (c) 2023-2024 Nol Moonen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef JPEGGPU_DEFS_HPP_
#define JPEGGPU_DEFS_HPP_

#include "logger.hpp"

#include <jpeggpu/jpeggpu.h>

#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <iostream>
#include <stdint.h>

// Accelerating JPEG Decompression on GPUs
// https://arxiv.org/abs/2111.09219

// JPEG Specification
// https://www.w3.org/Graphics/JPEG/itu-t81.pdf

namespace jpeggpu {

#define JPEGGPU_CHECK_CUDA(call)                                                               \
    do {                                                                                       \
        cudaError_t err = call;                                                                \
        if (err != cudaSuccess) {                                                              \
            logger.log(                                                                        \
                "CUDA error \"%s\" at: " __FILE__ ":%d\n", cudaGetErrorString(err), __LINE__); \
            return JPEGGPU_INTERNAL_ERROR;                                                     \
        }                                                                                      \
    } while (0)

#define JPEGGPU_CHECK_STAT(call)           \
    do {                                   \
        jpeggpu_status stat = call;        \
        if (stat != JPEGGPU_SUCCESS) {     \
            return JPEGGPU_INTERNAL_ERROR; \
        }                                  \
    } while (0)

/// \brief The number of rows or columns in a data unit, eight.
///
/// TODO "data unit" terminology is not correct, should be "block",
///   however "block" is confusing with the thread block concept
constexpr int data_unit_vector_size = 8;
/// \brief The number of pixels in a data units, 64.
constexpr int data_unit_size = 64;

/// \brief Maximum number of Huffman tables that can be defined per scan.
///
/// A Huffman table can be defined as one of four possible indices, tables
///   may be redefined between scans.
constexpr int max_huffman_count_per_scan = 4;
/// as defined by jpeg spec
constexpr int max_comp_count = JPEGGPU_MAX_COMP;

constexpr int max_qtable_count_per_scan = 4;

/// huffman types
enum huff { HUFF_DC = 0, HUFF_AC = 1, HUFF_COUNT = 2 };

using qtable = uint8_t[64];

// clang-format off
/// \brief Convert zig-zag index to raster index,
///   `i`th value is the raster index of the zig-zag index `i`.
__device__ constexpr int order_natural[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63};
// clang-format on

/// \brief Enable additional, synchronous debugging. Intentionally not a compile-time constant.
extern bool is_debug;

inline bool operator==(const jpeggpu_subsampling& lhs, const jpeggpu_subsampling& rhs)
{
    for (int c = 0; c < jpeggpu::max_comp_count; ++c) {
        if (lhs.x[c] != rhs.x[c] || lhs.y[c] != rhs.y[c]) {
            return false;
        }
    }
    return true;
}

inline bool operator!=(const jpeggpu_subsampling& lhs, const jpeggpu_subsampling& rhs)
{
    return !(lhs == rhs);
}

} // namespace jpeggpu

#endif // JPEGGPU_DEFS_HPP_
