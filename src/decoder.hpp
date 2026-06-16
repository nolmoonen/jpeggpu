// Copyright (c) 2023-2024 Nol Moonen
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

#ifndef JPEGGPU_DECODER_HPP_
#define JPEGGPU_DECODER_HPP_

#include "logger.hpp"
#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <cstdarg>
#include <stddef.h>
#include <stdint.h>
#include <vector>

namespace jpeggpu {

struct decoder {

    jpeggpu_status init();

    void cleanup();

    jpeggpu_status parse_header(jpeggpu_img_info& img_info, const uint8_t* data, size_t size);

    jpeggpu_status transfer(void* d_tmp, size_t tmp_size, cudaStream_t stream);

    template <bool do_it>
    jpeggpu_status decode_impl(cudaStream_t stream);

    template <bool do_it>
    jpeggpu_status decode_idct_impl(jpeggpu_img* img, cudaStream_t stream);

    jpeggpu_status decode_get_size(size_t& tmp_size);

    jpeggpu_status decode_common(void* d_tmp, size_t tmp_size, cudaStream_t stream);

    jpeggpu_status decode(jpeggpu_img* img, void* d_tmp, size_t tmp_size, cudaStream_t stream);

    jpeggpu_status decode_no_idct(void* d_tmp, size_t tmp_size, cudaStream_t stream);

    struct reader reader;

    /// \brief Keeps track of allocations for the current decoded image.
    stack_allocator allocator;

    struct logger logger;

    qtable* d_qtables[max_comp_count];

    // Output of decoding, quantized and cosine-transformed image data.
    int16_t* d_image_qdct[max_comp_count];
};

} // namespace jpeggpu

struct jpeggpu_decoder {
    jpeggpu::decoder decoder;
};

#endif // JPEGGPU_DECODER_HPP_
