#ifndef JPEGGPU_DECODER_DEFS_HPP_
#define JPEGGPU_DECODER_DEFS_HPP_

namespace jpeggpu {
/// \brief "s", subsequence size in 32 bits. Paper uses 4 or 32 depending on the quality of the
///   encoded image.
constexpr int chunk_size             = 16; ///< "s", in 32 bits. configurable
constexpr int subsequence_size_bytes = chunk_size * 4;
static_assert(subsequence_size_bytes % 4 == 0);
constexpr int subsequence_size = chunk_size * 32; ///< size in bits
static_assert(subsequence_size % 32 == 0); // is always multiple of 32 bits
} // namespace jpeggpu

#endif // JPEGGPU_DECODER_DEFS_HPP_
