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
