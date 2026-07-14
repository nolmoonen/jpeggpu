// Copyright (c) 2024 Nol Moonen
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
