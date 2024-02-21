#ifndef JPEGGPU_DECODE_GPU_HPP_
#define JPEGGPU_DECODE_GPU_HPP_

#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <stddef.h>
#include <stdint.h>

namespace jpeggpu {

size_t calculate_gpu_decode_memory(const jpeggpu::reader& reader);

/// \brief Unregarded of whether the scan is interleaved or non-interleaved,
///   this function will output the quantized-cosine-transformed pixel data
///   in planar format.
jpeggpu_status decode(
    logger& logger,
    reader& reader,
    const uint8_t* d_image_data,
    uint8_t* d_image_data_destuffed,
    int16_t* (&d_image_qdct)[jpeggpu::max_comp_count],
    void*& d_tmp,
    size_t& tmp_size,
    cudaStream_t stream);

} // namespace jpeggpu

#endif // JPEGGPU_DECODE_GPU_HPP_
