// Copyright (c) 2024-2026 Nol Moonen
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

#ifndef JPEGGPU_DECODE_HUFFMAN_HPP_
#define JPEGGPU_DECODE_HUFFMAN_HPP_

#include "decode_destuff.hpp"
#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <stddef.h>
#include <stdint.h>

namespace jpeggpu {

/// \brief Decodes the Huffman-encoded JPEG data.
///
/// \param[in] info
/// \param[in] d_scan_destuffed Device memory, holding destuffed scan data. In other words,
///   all the bytes after a SOS marker until the next non-restart marker. In this data, all restart markers
///   (0xffd0-0xffd7) are removed and all stuffed bytes (0xff00) are replaced by 0xff.
///   All segments are rounded up to be multiples of subsequence size.
///   Should be aligned with cudaMalloc alignment.
/// \param[in] d_segments Device memory, segment info.
/// \param[in] d_segment_indices Device memory, for every subsequence its segment index.
/// \param[in] scan Scan info.
/// \param[in] d_huff_tables Device memory, Huffman tables, in the order following their IDs in the JPEG header.
/// \param[inout] allocator
/// \param[inout] stream
/// \param[inout] logger
template <bool do_it>
jpeggpu_status decode_sync(
    const jpeg_stream& info,
    const uint8_t* d_scan_destuffed,
    const segment* d_segments,
    const int* d_segment_indices,
    int16_t* d_dcs,
    int* d_block_bit_offsets,
    const struct jpeggpu::scan& scan,
    huffman_table* d_huff_tables,
    stack_allocator& allocator,
    cudaStream_t stream,
    logger& logger);

} // namespace jpeggpu

#endif // JPEGGPU_DECODE_HUFFMAN_HPP_
