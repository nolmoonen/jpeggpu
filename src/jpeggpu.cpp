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

#include "decoder.hpp"
#include "logger.hpp"
#include "marker.hpp"

#include <jpeggpu/jpeggpu.h>

#include <array>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <type_traits>
#include <vector>

extern "C" {

const char* jpeggpu_get_status_string(enum jpeggpu_status stat)
{
    switch (stat) {
    case JPEGGPU_SUCCESS:
        return "success";
    case JPEGGPU_INVALID_ARGUMENT:
        return "invalid argument";
    case JPEGGPU_INVALID_JPEG:
        return "invalid jpeg";
    case JPEGGPU_INTERNAL_ERROR:
        return "internal jpeggpu error";
    case JPEGGPU_NOT_SUPPORTED:
        return "jpeg is not supported";
    case JPEGGPU_OUT_OF_HOST_MEMORY:
        return "out of host memory";
    }
    return "unknown status";
}

enum jpeggpu_status jpeggpu_decoder_startup(jpeggpu_decoder_t* decoder)
{
    if (!decoder) {
        return JPEGGPU_INVALID_ARGUMENT;
    }

    *decoder = reinterpret_cast<jpeggpu_decoder*>(malloc(sizeof(jpeggpu_decoder)));

    return (*decoder)->decoder.init();
}

enum jpeggpu_status jpeggpu_set_logging(jpeggpu_decoder_t decoder, int do_logging)
{
    if (!decoder) {
        return JPEGGPU_INVALID_ARGUMENT;
    }

    decoder->decoder.logger.set_logging(do_logging);

    return JPEGGPU_SUCCESS;
}

int is_css_444(struct jpeggpu_subsampling css, int num_components)
{
    switch (num_components) {
    case 1:
        return css.x[0] == 1 && css.y[0] == 1;
    case 2:
        return css.x[0] == 1 && css.y[0] == 1 && css.x[1] == 1 && css.y[1] == 1;
    case 3:
        return css.x[0] == 1 && css.y[0] == 1 && css.x[1] == 1 && css.y[1] == 1 && css.x[2] == 1 &&
               css.y[2] == 1;
    case 4:
        return css.x[0] == 1 && css.y[0] == 1 && css.x[1] == 1 && css.y[1] == 1 && css.x[2] == 1 &&
               css.y[2] == 1 && css.x[3] == 1 && css.y[3] == 1;
    }
    return 0;
}

enum jpeggpu_status jpeggpu_decoder_parse_header(
    jpeggpu_decoder_t decoder, struct jpeggpu_img_info* img_info, const uint8_t* data, size_t size)
{
    if (!decoder) {
        return JPEGGPU_INVALID_ARGUMENT;
    }

    return decoder->decoder.parse_header(*img_info, data, size);
}

enum jpeggpu_status jpeggpu_decoder_get_buffer_size(jpeggpu_decoder_t decoder, size_t* tmp_size)
{
    if (!decoder || !tmp_size) {
        return JPEGGPU_INVALID_ARGUMENT;
    }

    return decoder->decoder.decode_get_size(*tmp_size);
}

enum jpeggpu_status jpeggpu_decoder_transfer(
    jpeggpu_decoder_t decoder, void* d_tmp, size_t tmp_size, cudaStream_t stream)
{
    if (!decoder) {
        return JPEGGPU_INVALID_ARGUMENT;
    }

    return decoder->decoder.transfer(d_tmp, tmp_size, stream);
}

enum jpeggpu_status jpeggpu_decoder_decode(
    jpeggpu_decoder_t decoder,
    struct jpeggpu_img* img,
    void* d_tmp,
    size_t tmp_size,
    cudaStream_t stream)
{
    if (!decoder || !img) {
        return JPEGGPU_INVALID_ARGUMENT;
    }

    return decoder->decoder.decode(img, d_tmp, tmp_size, stream);
}

enum jpeggpu_status jpeggpu_decoder_cleanup(jpeggpu_decoder_t decoder)
{
    if (!decoder) {
        return JPEGGPU_INVALID_ARGUMENT;
    }

    decoder->decoder.cleanup();
    free(decoder);

    return JPEGGPU_SUCCESS;
}
}
