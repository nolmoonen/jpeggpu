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

    jpeggpu_status transfer(void* d_tmp, size_t tmp_size, cudaStream_t stream);

    template <bool do_it>
    jpeggpu_status decode_impl(cudaStream_t stream);

    jpeggpu_status decode_get_size(size_t& tmp_size);

    jpeggpu_status decode(
        uint8_t* (&image)[max_comp_count],
        int (&pitch)[max_comp_count],
        jpeggpu_color_format_out color_fmt,
        jpeggpu_subsampling subsampling,
        bool out_is_interleaved,
        void* d_tmp,
        size_t tmp_size,
        cudaStream_t stream);

    jpeggpu_status decode(jpeggpu_img* img, void* d_tmp, size_t tmp_size, cudaStream_t stream);

    jpeggpu_status decode(
        jpeggpu_img_interleaved* img, void* d_tmp, size_t tmp_size, cudaStream_t stream);

    struct reader reader;

    uint8_t* d_qtables[max_comp_count];
    huffman_table* d_huff_tables[max_huffman_count][HUFF_COUNT];

    // output of decoding, quantized and cosine-transformed image data
    int16_t* d_image_qdct[max_comp_count];

    // output of dequantize and idct, the output image
    uint8_t* d_image[max_comp_count];

    stack_allocator allocator;
    struct logger logger;
};

} // namespace jpeggpu

struct jpeggpu_decoder {
    jpeggpu::decoder decoder;
};

#endif // JPEGGPU_DECODER_HPP_
