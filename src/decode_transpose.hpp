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

#ifndef JPEGGPU_DECODE_TRANSPOSE_HPP_
#define JPEGGPU_DECODE_TRANSPOSE_HPP_

#include "logger.hpp"
#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <cuda_runtime.h>

#include <stdint.h>

namespace jpeggpu {

jpeggpu_status decode_transpose_component(
    const int16_t* d_out,
    int16_t* d_image_qdct,
    ivec2 data_size,
    int data_size_mcu_x,
    cudaStream_t stream,
    logger& logger);

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
