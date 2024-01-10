#ifndef JPEGGPU_DECODE_GPU_HPP_
#define JPEGGPU_DECODE_GPU_HPP_

#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

bool is_gpu_decode_possible(jpeggpu::reader& reader);

jpeggpu_status process_scan(
    jpeggpu::reader& reader,
    int16_t* (&d_image_qdct)[jpeggpu::max_comp_count],
    cudaStream_t stream);

#endif // JPEGGPU_DECODE_GPU_HPP_
