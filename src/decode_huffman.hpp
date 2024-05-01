#ifndef JPEGGPU_DECODE_HUFFMAN_HPP_
#define JPEGGPU_DECODE_HUFFMAN_HPP_

#include "decode_destuff.hpp"
#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <stddef.h>
#include <stdint.h>

namespace jpeggpu {

/// \brief Decodes the Huffman-encoded JPEG data.
///
/// \param[in] info
/// \param[in] d_scan_destuffed Pointer to device memory holding destuffed scan data. In other words,
///   all the bytes after a SOS marker until the next non-restart marker. In this data, all restart markers
///   (0xffd0-0xffd7) are removed and all stuffed bytes (0xff00) are replaced by 0xff.
/// \param[in] d_segment_infos Pointer to device memory holding information about segments (sections in data formed
///   by restart markers).
/// \param[out] d_out Pointer to device memory where quantized-cosine-transformed pixel data should be stored.
///   If the scan is interleaved, it should be big enough to hold all components. If the scan is not interleaved,
///   this function should be called multiple times for each scan. The data is stored as how it appears in the stream:
///   one data unit at a time, components possibly interleaved.
/// \param[in] scan
/// \param[in] d_huff_tables Pointer to device memory holding Huffman tables, in the order following their IDs in the JPEG header.
/// \param[in] allocator
/// \param[in] stream
template <bool do_it>
jpeggpu_status decode_scan(
    const jpeg_stream& info,
    const uint8_t* d_scan_destuffed,
    const segment* d_segments,
    const int* d_segment_indices,
    int16_t* d_out,
    const struct jpeggpu::scan& scan,
    huffman_table* d_huff_tables,
    stack_allocator& allocator,
    cudaStream_t stream,
    logger& logger);

} // namespace jpeggpu

#endif // JPEGGPU_DECODE_HUFFMAN_HPP_
