#ifndef JPEGGPU_DECODE_GPU_HPP_
#define JPEGGPU_DECODE_GPU_HPP_

#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

namespace jpeggpu {

bool is_gpu_decode_possible(const jpeggpu::reader& reader);

size_t calculate_gpu_decode_memory(const jpeggpu::reader& reader);

/// \brief Unregarding of whether the scan is interleaved or non-interleaved,
///   this function will output the quantized-cosine-transformed pixel data
///   in planar format. 
jpeggpu_status process_scan(
    jpeggpu::reader& reader,
    int16_t* (&d_image_qdct)[jpeggpu::max_comp_count],
    void*& d_tmp,
    size_t& tmp_size,
    cudaStream_t stream);

} // namespace jpeggpu

#endif // JPEGGPU_DECODE_GPU_HPP_
