// Copyright (c) 2024 Nol Moonen
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

#ifndef JPEGGPU_DECODE_HUFFMAN_COMMON_HPP_
#define JPEGGPU_DECODE_HUFFMAN_COMMON_HPP_

#include "reader.hpp"

namespace jpeggpu {

template <int block_size>
__device__ void load_huffman_table(const huffman_table& table_global, huffman_table& table_shared)
{
    // assert that loading a word at a time is valid
    static_assert(sizeof(huffman_table) % 4 == 0 && sizeof(huffman_table::entry) % 4 == 0);
    constexpr int num_words            = sizeof(huffman_table) / 4;
    constexpr int num_words_per_thread = // TODO not efficient since num_words < block_size
        ceiling_div(num_words, static_cast<unsigned int>(block_size));
    for (int i = 0; i < num_words_per_thread; ++i) {
        const int idx = block_size * i + threadIdx.x;
        if (idx < num_words) {
            reinterpret_cast<uint32_t*>(&table_shared)[idx] =
                reinterpret_cast<const uint32_t*>(&table_global)[idx];
        }
    }
}

/// \brief Returns the oldest (most significant) `num_bits` from `data`.
inline __device__ uint32_t u32_select_bits(uint32_t data, int num_bits)
{
    assert(num_bits <= 32);
    return data >> (32 - num_bits);
}

inline __device__ int get_value(int num_bits, int code)
{
    // TODO leftshift negative value is UB
    return code < ((1 << num_bits) >> 1) ? (code + ((-1) << num_bits) + 1) : code;
}

} // namespace jpeggpu

#endif // JPEGGPU_DECODE_HUFFMAN_COMMON_HPP_
