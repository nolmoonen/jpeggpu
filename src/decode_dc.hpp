#ifndef JPEGGPU_DECODE_DC_HPP_
#define JPEGGPU_DECODE_DC_HPP_

#include "logger.hpp"
#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <cuda_runtime.h>

#include <stdint.h>

namespace jpeggpu {

/// \brief Undo DC difference encoding
///
/// \param[in] info
/// \param[inout] d_out Device pointer to quantized and cosine-transformed data how it appears in the encoded JPEG stream.
/// \param[in] allocator
/// \param[in] stream
template <bool do_it>
jpeggpu_status decode_dc(
    const jpeg_stream& info,
    int16_t* d_out,
    stack_allocator& allocator,
    cudaStream_t stream,
    logger& logger);

} // namespace jpeggpu

#endif // JPEGGPU_DECODE_DC_HPP_
