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

#ifndef JPEGGPU_DECODE_BLOCK_HPP_
#define JPEGGPU_DECODE_BLOCK_HPP_

#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <stddef.h>
#include <stdint.h>

namespace jpeggpu {

template <bool do_it>
jpeggpu_status decode_block(
    const jpeg_stream& info,
    const uint8_t* d_scan_destuffed,
    int16_t* d_dcs,
    int* d_block_bit_offsets,
    const struct jpeggpu::scan& scan,
    huffman_table* d_huff_tables,
    uint8_t* (&d_image)[max_comp_count],
    int (&pitch)[max_comp_count],
    qtable* (&d_qtable)[max_comp_count],
    stack_allocator& allocator,
    cudaStream_t stream,
    logger& logger);

} // namespace jpeggpu

#endif // JPEGGPU_DECODE_BLOCK_HPP_
