#ifndef JPEGGPU_DECODER_HPP_
#define JPEGGPU_DECODER_HPP_

#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <stddef.h>
#include <stdint.h>

namespace jpeggpu {

struct decoder {

    jpeggpu_status parse_header(jpeggpu_img_info* img_info, const uint8_t* data, size_t size);

    jpeggpu_status decode(jpeggpu_img* img);

    struct reader reader;
};

} // namespace jpeggpu

struct jpeggpu_decoder {
    jpeggpu::decoder decoder;
};

#endif // JPEGGPU_DECODER_HPP_