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

#ifndef JPEGGPU_DESTUFF_HPP_
#define JPEGGPU_DESTUFF_HPP_

#include "reader.hpp"

namespace jpeggpu {

/// \brief Prepares the scan data for decoding.
///   Remove stuffed bytes (0x00 is inserted before encoded 0xFF), restart markers (0xFFD0-0xFFD7),
///   and pads segments to be integer multiples of subsequence size.
///
/// \param[in] info Info about the JPEG stream.
/// \param[in] d_image_data Device memory, JPEG file data.
/// \param[out] d_scan_destuffed Device memory, destuffed scan.
/// \param[in] d_segments Device memory, segment info.
/// \param[out] d_segment_indices Device memory, for every subsequence its segment index.
/// \param[in] scan Scan info.
/// \param[inout] allocator
/// \param[inout] stream
/// \param[inout] logger
template <bool do_it>
jpeggpu_status destuff_scan(
    const jpeg_stream& info,
    const uint8_t* d_image_data,
    uint8_t* d_scan_destuffed,
    const segment* d_segments,
    int* d_segment_indices,
    const scan& scan,
    stack_allocator& allocator,
    cudaStream_t stream,
    logger& logger);

} // namespace jpeggpu

#endif // JPEGGPU_DESTUFF_HPP_
