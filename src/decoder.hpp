#ifndef JPEGGPU_DECODER_HPP_
#define JPEGGPU_DECODER_HPP_

#include "logger.hpp"
#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <cstdarg>
#include <stddef.h>
#include <stdint.h>

namespace jpeggpu {

struct decoder {

    jpeggpu_status init();

    void cleanup();

    jpeggpu_status parse_header(jpeggpu_img_info& img_info, const uint8_t* data, size_t size);

    template <bool do_it>
    jpeggpu_status decode_impl(
        jpeggpu_img& img,
        jpeggpu_color_format color_fmt,
        jpeggpu_pixel_format pixel_fmt,
        jpeggpu_subsampling subsampling,
        cudaStream_t stream);

    jpeggpu_status decode(
        jpeggpu_img& img,
        jpeggpu_color_format color_fmt,
        jpeggpu_pixel_format pixel_fmt,
        jpeggpu_subsampling subsampling,
        void* d_tmp,
        size_t& tmp_size,
        cudaStream_t stream);

    struct logger logger;

    struct reader reader;

    uint8_t* d_qtables[max_comp_count];

    // output of decoding, quantized and cosine-transformed image data
    int16_t* d_image_qdct[max_comp_count];

    // output of dequantize and idct, the output image
    uint8_t* d_image[max_comp_count];

    stack_allocator allocator;
};

} // namespace jpeggpu

struct jpeggpu_decoder {
    jpeggpu::decoder decoder;
};

#endif // JPEGGPU_DECODER_HPP_
