#ifndef JPEGGPU_DECODE_TRANSPOSE_HPP_
#define JPEGGPU_DECODE_TRANSPOSE_HPP_

#include "logger.hpp"
#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <cuda_runtime.h>

#include <stdint.h>

namespace jpeggpu {

/// \brief Converts the image data from order as it appears in Huffman encoding to raster order
///   spread over multiple components, undoing interleaving.
///
/// \param[in] info
/// \param[in] d_out Pointer to device memory holding output of Huffman decoding.
/// \param[out] d_image_qdct For each component, a pointer to device memory where image data should be stored.
/// \param[in] stream
jpeggpu_status decode_transpose(
    const jpeg_stream& info,
    const int16_t* d_out,
    int16_t* (&d_image_qdct)[jpeggpu::max_comp_count],
    cudaStream_t stream,
    logger& logger);

} // namespace jpeggpu

#endif // JPEGGPU_DECODE_TRANSPOSE_HPP_
