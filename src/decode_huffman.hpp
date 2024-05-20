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
/// \param[out] d_out Pointer to device memory where quantized-cosine-transformed pixel data should be stored.
///   If the scan is interleaved, it should be big enough to hold all components. If the scan is not interleaved,
///   this function should be called multiple times for each scan. The data is stored as how it appears in the stream:
///   one data unit at a time, components possibly interleaved.
/// \param[in] scan Scan info.
/// \param[in] d_huff_tables Device memory, Huffman tables, in the order following their IDs in the JPEG header.
/// \param[inout] allocator
/// \param[inout] stream
/// \param[inout] logger
template <bool do_it>
jpeggpu_status decode_scan(
    const jpeg_stream& info,
    const uint8_t* d_scan_destuffed,
    const segment* d_segments,
    const int* d_segment_indices,
    int16_t* d_out,
    const struct jpeggpu::scan& scan,
    huffman_table* d_huff_tables,
    stack_allocator& allocator,
    cudaStream_t stream,
    logger& logger);

} // namespace jpeggpu

#endif // JPEGGPU_DECODE_HUFFMAN_HPP_
