#ifndef JPEGGPU_DEFS_HPP_
#define JPEGGPU_DEFS_HPP_

#include <jpeggpu/jpeggpu.h>

#include <array>
#include <cstdio>
#include <stdint.h>

namespace jpeggpu {

// https://arxiv.org/abs/2111.09219
// https://www.w3.org/Graphics/JPEG/itu-t81.pdf

#define DBG_PRINT(...) printf(__VA_ARGS__);

constexpr int block_size     = 8;
/// components
constexpr int max_comp_count = JPEGGPU_MAX_COMP;
/// huffman types
enum huff { HUFF_DC = 0, HUFF_AC = 1, HUFF_COUNT = 2 };

using qtable = uint8_t[64];

// clang-format off
constexpr std::array<int, 64 + 16> order_natural = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
    63, 63, 63, 63, 63, 63, 63, 63, // Extra entries for safety in decoder
    63, 63, 63, 63, 63, 63, 63, 63};
// clang-format on

} // namespace jpeggpu

#endif // JPEGGPU_DEFS_HPP_
