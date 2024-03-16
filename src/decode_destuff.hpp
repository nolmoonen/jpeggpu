#ifndef JPEGGPU_DESTUFF_HPP_
#define JPEGGPU_DESTUFF_HPP_

#include "reader.hpp"

namespace jpeggpu {

/// \brief Holds info describing a segment. If a restart marker should be placed
///   but the number of encoded bits is not a multiple of eight, the encoded stream
///   is padded with zeroes. To be able to deal with this in the decoding process,
///   these segments must be decoded independently.
struct segment_info {
    /// \brief Global destuffed index of first byte in this segment.
    int begin;
    /// \brief Global destuffed index of first byte beyond this segment.
    int end;
    /// \brief Number of subsequences that come before this segment.
    int subseq_offset;
};

/// \brief Remove stuffed bytes, restart markers, and determine segments.
///
/// \param[in] reader
/// \param[out] d_scan Device pointer to destuffed scan bytes.
/// \param[out] scan_size
/// \param[out] num_subsequences The maximum amount of subsequences.
/// \param[out] d_segment_infos For each segment, a range in `d_scan`.
/// \param[out] d_segment_indices For each data byte, the segment index.
template<bool do_it>
jpeggpu_status destuff_scan(
    const jpeg_stream& info,
    segment_info* d_segment_infos,
    int* d_segment_indices,
    const uint8_t* d_image_data,
    uint8_t* d_image_data_destuffed,
    const scan& scan,
    stack_allocator& allocator,
    cudaStream_t stream);

} // namespace jpeggpu

#endif // JPEGGPU_DESTUFF_HPP_
