// Copyright (c) 2026 Nol Moonen
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#ifndef JPEGGPU_HUFFMAN_CUH_
#define JPEGGPU_HUFFMAN_CUH_

#include "reader.hpp"

namespace jpeggpu {

__device__ inline int huff_extend(int x, int s)
{
    // Table F.1 and Table F.2
    assert(s >= 1);
    int half = 1 << (s - 1);
    if (x >= half) {
        assert(x < (1 << s));
        return x;
    } else {
        return x - (1 << s) + 1;
    }
}

/// \brief Returns the oldest (most significant) `num_bits` from `data`.
__device__ inline uint32_t u32_select_bits(uint32_t data, int num_bits)
{
    assert(num_bits <= 32);
    return data >> (32 - num_bits);
}

/// \brief Get the Huffman category from stream. Reads at most 16 bits.
///
/// \param[in] data Holds at least 16 data bits in the most significant positions.
/// \param[out] length Number of bits read.
/// \param[in] table
__device__ inline uint8_t get_category(uint32_t data, int& length, const huffman_table& table)
{
    const int id = u32_select_bits(data, huffman_table::lookup_len);

    const typename huffman_table::lut_entry row = table.lut[id];
    if (row.nbits != 0) {
        length = row.nbits;
        return row.val;
    }

    int i;
    int32_t code;
    huffman_table::entry entry;
    for (i = huffman_table::lookup_len; i < 16; ++i) {
        code                    = u32_select_bits(data, i + 1);
        const bool is_last_iter = i == 15;
        entry                   = table.entries[i];
        if (code <= entry.maxcode || is_last_iter) {
            break;
        }
    }
    assert(1 <= i + 1 && i + 1 <= 16);
    // termination condition: 1 <= i + 1 <= 16, i + 1 is number of bits
    length        = i + 1;
    const int idx = entry.valptr_sub_mincode + code;
    // Cast to [0, 256) to produce a valid index in the array in the event of invalid input.
    return table.huffval[static_cast<uint8_t>(idx)];
}

/// \brief Typedef for the maximum amount of Huffman tables for a scan.
using huffman_tables = huffman_table[max_baseline_huff_per_scan];

/// \brief Load Huffman tables from global memory into shared.
template <int block_size>
__device__ void load_huffman_tables(
    const huffman_table* huffman_tables_global, huffman_tables& tables_shared)
{
    // assert that loading a word at a time is valid
    static_assert(sizeof(huffman_table) % 4 == 0 && sizeof(huffman_table::entry) % 4 == 0);
    constexpr int num_words = sizeof(huffman_tables) / 4;
    constexpr int num_words_per_thread =
        ceiling_div(num_words, static_cast<unsigned int>(block_size));
    for (int i = 0; i < num_words_per_thread; ++i) {
        const int idx = block_size * i + threadIdx.x;
        if (idx < num_words) {
            reinterpret_cast<uint32_t*>(&tables_shared)[idx] =
                reinterpret_cast<const uint32_t*>(huffman_tables_global)[idx];
        }
    }
}

} // namespace jpeggpu

#endif // JPEGGPU_HUFFMAN_CUH_
