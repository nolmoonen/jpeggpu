#ifndef JPEGGPU_DESTUFF_HPP_
#define JPEGGPU_DESTUFF_HPP_

#include "reader.hpp"

namespace jpeggpu {

/// \brief Remove stuffed bytes, restart markers, and determine segments.
template <bool do_it>
jpeggpu_status destuff_scan(
    const jpeg_stream& info,
    const uint8_t* d_image_data,
    uint8_t* d_scan_destuffed,
    const segment* d_segments,
    int* d_segment_indices, // for every subsequence, its segment
    const scan& scan,
    stack_allocator& allocator,
    cudaStream_t stream);

} // namespace jpeggpu

#endif // JPEGGPU_DESTUFF_HPP_
