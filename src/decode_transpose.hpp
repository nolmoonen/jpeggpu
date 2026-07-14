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

#ifndef JPEGGPU_DECODE_TRANSPOSE_HPP_
#define JPEGGPU_DECODE_TRANSPOSE_HPP_

#include "logger.hpp"
#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <cuda_runtime.h>

#include <stdint.h>

namespace jpeggpu {

/// \brief Converts the image data from order as it appears in Huffman encoding to raster order
///   spread over multiple components, undoing interleaving.
///
/// \param[in] info
/// \param[in] d_out Device memory, holds output of Huffman decoding.
/// \param[in] scan
/// \param[out] d_image_qdct Device memory, for each component, where image data should be stored.
/// \param[inout] stream
/// \param[inout] logger
jpeggpu_status decode_transpose(
    const jpeg_stream& info,
    const int16_t* d_out,
    const scan& scan,
    int16_t* (&d_image_qdct)[jpeggpu::max_comp_count],
    cudaStream_t stream,
    logger& logger);

} // namespace jpeggpu

#endif // JPEGGPU_DECODE_TRANSPOSE_HPP_
