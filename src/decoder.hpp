#ifndef JPEGGPU_DECODER_HPP_
#define JPEGGPU_DECODER_HPP_

#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <stddef.h>
#include <stdint.h>

namespace jpeggpu {

struct decoder {

    jpeggpu_status parse_header(jpeggpu_img_info& img_info, const uint8_t* data, size_t size);

    jpeggpu_status decode(
        jpeggpu_img& img,
        jpeggpu_color_format color_fmt,
        jpeggpu_pixel_format pixel_fmt,
        jpeggpu_subsampling subsampling,
        void* d_tmp,
        size_t& tmp_size,
        cudaStream_t stream);

    struct reader reader;

    // output of decoding, quantized and cosine-transformed image data
    int16_t* d_image_qdct[max_comp_count];

    // output of dequantize and idct, the output image
    uint8_t* d_image[max_comp_count];
};

} // namespace jpeggpu

struct jpeggpu_decoder {
    jpeggpu::decoder decoder;
};

#endif // JPEGGPU_DECODER_HPP_
