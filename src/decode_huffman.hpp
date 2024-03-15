#ifndef JPEGGPU_DECODE_HUFFMAN_HPP_
#define JPEGGPU_DECODE_HUFFMAN_HPP_

#include "decode_destuff.hpp"
#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <stddef.h>
#include <stdint.h>

namespace jpeggpu {

/// \brief Unregarded of whether the scan is interleaved or non-interleaved,
///   this function will output the quantized-cosine-transformed pixel data
///   in planar format.
template<bool do_it>
jpeggpu_status decode_scan(
    logger& logger,
    reader& reader,
    uint8_t* d_scan_destuffed,
    segment_info* d_segment_infos,
    int* d_segment_indices,
    int16_t* d_out,
    const struct jpeggpu::scan& scan,
    stack_allocator& allocator,
    cudaStream_t stream);

} // namespace jpeggpu

#endif // JPEGGPU_DECODE_HUFFMAN_HPP_
