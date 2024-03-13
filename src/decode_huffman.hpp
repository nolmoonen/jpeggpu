#ifndef JPEGGPU_DECODE_HUFFMAN_HPP_
#define JPEGGPU_DECODE_HUFFMAN_HPP_

#include "decode_destuff.hpp"
#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <stddef.h>
#include <stdint.h>

namespace jpeggpu {

size_t calculate_gpu_decode_memory(const jpeggpu::reader& reader);

/// \brief Unregarded of whether the scan is interleaved or non-interleaved,
///   this function will output the quantized-cosine-transformed pixel data
///   in planar format.
jpeggpu_status decode_scan(
    logger& logger,
    reader& reader,
    uint8_t* d_scan_destuffed,
    segment_info* d_segment_infos,
    int* d_segment_indices,
    int16_t* d_out,
    void*& d_tmp,
    size_t& tmp_size,
    const struct jpeggpu::scan& scan,
    cudaStream_t stream);

} // namespace jpeggpu

#endif // JPEGGPU_DECODE_HUFFMAN_HPP_
