#ifndef JPEGGPU_DEFS_HPP_
#define JPEGGPU_DEFS_HPP_

#include "logger.hpp"

#include <jpeggpu/jpeggpu.h>

#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <iostream>
#include <stdint.h>

// https://arxiv.org/abs/2111.09219
// https://www.w3.org/Graphics/JPEG/itu-t81.pdf
// https://u.cs.biu.ac.il/~wisemay/computer2003.pdf

namespace jpeggpu {

#define JPEGGPU_CHECK_CUDA(call)                                                                   \
    do {                                                                                           \
        cudaError_t err = call;                                                                    \
        if (err != cudaSuccess) {                                                                  \
            log("CUDA error \"%s\" at: " __FILE__ ":%d\n", cudaGetErrorString(err), __LINE__);     \
            return JPEGGPU_INTERNAL_ERROR;                                                         \
        }                                                                                          \
    } while (0)

#define JPEGGPU_CHECK_STAT(call)                                                                   \
    do {                                                                                           \
        jpeggpu_status stat = call;                                                                \
        if (stat != JPEGGPU_SUCCESS) {                                                             \
            return JPEGGPU_INTERNAL_ERROR;                                                         \
        }                                                                                          \
    } while (0)

/// TODO more descriptive name?
constexpr int block_size     = 8;
constexpr int data_unit_size = 64;
/// as defined by jpeg spec
constexpr int max_comp_count = JPEGGPU_MAX_COMP;
/// huffman types
enum huff { HUFF_DC = 0, HUFF_AC = 1, HUFF_COUNT = 2 };

using qtable = uint8_t[64];

// clang-format off
__device__ constexpr int order_natural[64 + 16] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
    63, 63, 63, 63, 63, 63, 63, 63, // Extra entries for safety in decoder TODO is this needed?
    63, 63, 63, 63, 63, 63, 63, 63};
// clang-format on

constexpr bool is_debug = false;

} // namespace jpeggpu

#endif // JPEGGPU_DEFS_HPP_
