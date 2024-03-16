#ifndef JPEGGPU_DECODE_TRANSPOSE_HPP_
#define JPEGGPU_DECODE_TRANSPOSE_HPP_

#include "logger.hpp"
#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <cuda_runtime.h>

#include <stdint.h>

namespace jpeggpu {

jpeggpu_status decode_transpose(
    const jpeg_stream& info,
    int16_t* d_out,
    int16_t* (&d_image_qdct)[jpeggpu::max_comp_count],
    cudaStream_t stream);

}

#endif // JPEGGPU_DECODE_TRANSPOSE_HPP_
