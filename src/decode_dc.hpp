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

#ifndef JPEGGPU_DECODE_DC_HPP_
#define JPEGGPU_DECODE_DC_HPP_

#include "logger.hpp"
#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <cuda_runtime.h>

#include <stdint.h>

namespace jpeggpu {

/// \brief Undo DC difference encoding: performs the component-wise inplace sum of the
///   DC values by segment index.
///
/// \param[in] info
/// \param[in] scan
/// \param[inout] d_out Device pointer to quantized and cosine-transformed data how it appears in the encoded JPEG stream.
/// \param[inout] allocator
/// \param[inout] stream
/// \param[inout] logger
template <bool do_it>
jpeggpu_status decode_dc(
    const jpeg_stream& info,
    const scan& scan,
    int16_t* d_out,
    stack_allocator& allocator,
    cudaStream_t stream,
    logger& logger);

} // namespace jpeggpu

#endif // JPEGGPU_DECODE_DC_HPP_
