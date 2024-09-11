// Copyright (c) 2024 Nol Moonen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef JPEGGPU_DECODE_HUFFMAN_READER_HPP_
#define JPEGGPU_DECODE_HUFFMAN_READER_HPP_

#include "decoder_defs.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cstring>
#include <stdint.h>

namespace jpeggpu {

/// \brief Rearrange bytes after loading an unsigned 32-bits integer.
__device__ uint32_t swap_endian(uint32_t x)
{
    return __byte_perm(x, uint32_t{0}, uint32_t{0x0123});
}

/// \brief Reader state where each thread as a private buffer in shared memory.
template <int block_size>
struct reader_state_thread_cache {
    /// \brief Pointer to first data byte, will always be aligned to cudaMalloc alignment.
    const uint8_t* data;
    /// \brief Pointer to byte following the last data byte.
    const uint8_t* data_end;
    /// \brief Cache of two sectors, which are each 128 bits (4x32-bits words).
    ///   The oldest bits are in the most significant positions.
    ///
    /// TODO get clarity on "sector" definition
    ulonglong4* cache;
    /// \brief Index of the next bit in the cache to read.
    int cache_idx;

    /// \brief Typedef for one cache per thread.
    using smem_type = ulonglong4[block_size];
};

/// \brief Load the next sector (128 bits) into shared memory cache.
template <int block_size>
__device__ void global_load_uint4(reader_state_thread_cache<block_size>& rstate, uint4* shared)
{
    for (int i = 0; i < sizeof(uint4) / sizeof(uint32_t); ++i) {
        const uint32_t val                     = reinterpret_cast<const uint32_t*>(rstate.data)[i];
        reinterpret_cast<uint32_t*>(shared)[i] = swap_endian(val);
    }
    rstate.data += sizeof(uint4);
}

/// \brief Return the next 32 bits from cache, padded with zero bits if not enough available.
template <int block_size>
__device__ uint32_t load_32_bits(reader_state_thread_cache<block_size>& rstate)
{
    // check if more bits are needed
    if (256 <= rstate.cache_idx + 32) {
        // makes computation easier, means data is always multiple of the loaded size
        static_assert(subsequence_size_bytes % sizeof(uint4) == 0);

        // shift over to make place for 128 bits in z and w
        rstate.cache->x = rstate.cache->z;
        rstate.cache->y = rstate.cache->w;

        // assert that the lo cache is fully used up
        assert(128 <= rstate.cache_idx);
        rstate.cache_idx -= 128;

        if (sizeof(uint4) <= rstate.data_end - rstate.data) {
            for (int i = 0; i < sizeof(uint4) / sizeof(uint32_t); ++i) {
                const uint32_t val = reinterpret_cast<const uint32_t*>(rstate.data)[i];
                reinterpret_cast<uint32_t*>(rstate.cache)[4 + i] = swap_endian(val);
            }
            rstate.data += sizeof(uint4);
        } else {
            // just set it to zero, but is not required
            std::memset(&(reinterpret_cast<uint4*>(rstate.cache)[1]), 0, sizeof(uint4));
        }
    }
    // enough bits are present to pull 32 bits
    assert(rstate.cache_idx + 32 <= 256);

    int word_32_idx = rstate.cache_idx / 32;
    assert(word_32_idx < 8);
    // load the 32-bits word and the next, i.e. load a 64-bits word at the position of the 32-bits word
    word_32_idx  = min(word_32_idx, 6);
    uint64_t ret = (uint64_t{reinterpret_cast<uint32_t*>(rstate.cache)[word_32_idx]} << 32) |
                   reinterpret_cast<uint32_t*>(rstate.cache)[word_32_idx + 1];

    // shift to get the relevant 32 from 64 bits
    int off = word_32_idx * 32;
    assert(off <= rstate.cache_idx);
    int off_rel = rstate.cache_idx - off;
    assert(off_rel <= 32 && off_rel + 32 <= 64);
    int shift = 64 - off_rel - 32;

    return static_cast<uint32_t>(ret >> shift);
}

/// \brief Discards `bit_count` bits from the reader state.
template <int block_size>
__device__ void discard_bits(reader_state_thread_cache<block_size>& rstate, int bit_count)
{
    rstate.cache_idx += bit_count;
}

/// \brief Constructs the reader state from the start of a subsequence (fills the cache).
///
/// \param[in] scan
/// \param[in] segment_info Segment info for this subsequence.
/// \param[in] rstate_memory Pointer to shared memory cache.
/// \param[in] subseq_idx_rel Subsequence index relative to segment.
template <int block_size>
__device__ reader_state_thread_cache<block_size> rstate_from_subseq_start(
    const uint8_t* scan,
    const segment& segment_info,
    typename reader_state_thread_cache<block_size>::smem_type& rstate_memory,
    int subseq_idx_rel)
{
    reader_state_thread_cache<block_size> rstate;
    rstate.data = scan + (segment_info.subseq_offset + subseq_idx_rel) * subsequence_size_bytes;
    rstate.data_end =
        scan + (segment_info.subseq_offset + segment_info.subseq_count) * subsequence_size_bytes;
    rstate.cache     = &(rstate_memory[threadIdx.x]);
    rstate.cache_idx = 0;

    // assert aligned load is valid
    assert((rstate.data - scan) % sizeof(uint4) == 0);
    // there should be at least one sector
    assert(sizeof(uint4) <= rstate.data_end - rstate.data);
    global_load_uint4(rstate, &(reinterpret_cast<uint4*>(rstate.cache)[0]));

    if (sizeof(uint4) <= rstate.data_end - rstate.data) {
        global_load_uint4(rstate, &(reinterpret_cast<uint4*>(rstate.cache)[1]));
    }

    return rstate;
}

/// \brief Constructs the reader state from overflow, which need not be
///   the start of a subsequence (fills the cache).
///
/// \param[in] scan
/// \param[in] segment_info Segment info for this subsequence.
/// \param[in] rstate_memory Pointer to shared memory cache.
/// \param[in] p Bit offset in segment.
template <int block_size>
__device__ reader_state_thread_cache<block_size> rstate_from_subseq_overflow(
    const uint8_t* scan,
    const segment& segment_info,
    typename reader_state_thread_cache<block_size>::smem_type& rstate_memory,
    int p)
{
    reader_state_thread_cache<block_size> rstate;
    // loaded in chunks of 128, each chunk has 16 bytes
    const int byte_offset = (p / 128) * (sizeof(uint4));
    rstate.data = scan + segment_info.subseq_offset * subsequence_size_bytes + byte_offset;
    rstate.data_end =
        scan + (segment_info.subseq_offset + segment_info.subseq_count) * subsequence_size_bytes;
    rstate.cache     = &(rstate_memory[threadIdx.x]);
    rstate.cache_idx = p - byte_offset * 8;

    // assert aligned load is valid
    assert((rstate.data - scan) % sizeof(uint4) == 0);
    // TODO why is there not always one sector?
    //   there should be at least one sector
    if (sizeof(uint4) <= rstate.data_end - rstate.data) {
        global_load_uint4(rstate, &(reinterpret_cast<uint4*>(rstate.cache)[0]));
    }
    if (sizeof(uint4) <= rstate.data_end - rstate.data) {
        global_load_uint4(rstate, &(reinterpret_cast<uint4*>(rstate.cache)[1]));
    }

    return rstate;
}

/// \brief Reader state where all subsequences for all threads in the block are
///   read into shared memory.
template <int block_size>
struct reader_state_all_subsequences {
    uint64_t cache;
    int bits_in_cache;
    int next_word;
    uint32_t* smem;

    /// \brief Typedef for one cache per thread.
    using smem_type = uint32_t[block_size * chunk_size];
};

template <int block_size>
__device__ void load_all_subsequences(
    const uint8_t* scan,
    int num_subsequences,
    typename reader_state_all_subsequences<block_size>::smem_type& rstate_memory)
{
    constexpr int word_count_block = block_size * chunk_size;
    const int block_word_off       = blockIdx.x * word_count_block;
    for (int i = threadIdx.x; i < word_count_block; i += block_size) {
        // TODO only do if statement in last block?
        if (block_word_off + i < num_subsequences * chunk_size) {
            const uint32_t val = reinterpret_cast<const uint32_t*>(scan)[block_word_off + i];
            rstate_memory[i]   = swap_endian(val);
        }
    }
}

template <int block_size>
__device__ uint32_t load_32_bits(reader_state_all_subsequences<block_size>& rstate)
{
    if (rstate.bits_in_cache < 32) {
        if (rstate.next_word < block_size * chunk_size) {
            rstate.cache |= uint64_t{rstate.smem[rstate.next_word++]}
                            << (32 - rstate.bits_in_cache);
        }
        rstate.bits_in_cache += 32;
    }

    return rstate.cache >> 32;
}

template <int block_size>
__device__ void discard_bits(reader_state_all_subsequences<block_size>& rstate, int bit_count)
{
    assert(bit_count <= rstate.bits_in_cache);
    rstate.cache <<= bit_count;
    rstate.bits_in_cache -= bit_count;
}

template <int block_size>
__device__ reader_state_all_subsequences<block_size> rstate_from_subseq_start(
    const segment& segment_info,
    typename reader_state_all_subsequences<block_size>::smem_type& rstate_memory,
    int subseq_idx_rel)
{
    constexpr int word_count_block = block_size * chunk_size;
    const int block_word_off       = blockIdx.x * word_count_block;
    const int thread_word_off      = (segment_info.subseq_offset + subseq_idx_rel) * chunk_size;
    assert(block_word_off <= thread_word_off);

    reader_state_all_subsequences<block_size> rstate;
    rstate.next_word     = thread_word_off - block_word_off;
    rstate.bits_in_cache = 0;
    rstate.cache         = 0;
    rstate.smem          = rstate_memory;

    return rstate;
}

template <int block_size>
__device__ reader_state_all_subsequences<block_size> rstate_from_subseq_overflow(
    const segment& segment_info,
    typename reader_state_all_subsequences<block_size>::smem_type& rstate_memory,
    int subseq_idx_rel,
    int p,
    uint32_t cache)
{
    constexpr int word_count_block = block_size * chunk_size;
    const int block_word_off       = blockIdx.x * word_count_block;
    const int thread_word_off      = (segment_info.subseq_offset + subseq_idx_rel) * chunk_size;
    assert(block_word_off <= thread_word_off);

    reader_state_all_subsequences<block_size> rstate;
    rstate.next_word = thread_word_off - block_word_off;

    // The decoding of this subsequence starts from the results of having decoded the previous
    //   subsequence. Because decoding will not read beyond the subsequence boundary, we may need
    //   to start reading in bits that "belong" to the previous subsequence. For convenience, these
    //   bits are stored in subsequence_info::cache.

    // number of bits that are from the previous subsequence
    const int tmp = (subsequence_size - (p % subsequence_size)) % subsequence_size;
    assert(tmp < 32);
    rstate.bits_in_cache = tmp;
    // select the upper tmp bits of the 32 bits word
    uint32_t mask = ((1 << tmp) - 1) << (32 - tmp);
    // place them in the upper bits of the 64 bits word
    rstate.cache = uint64_t{cache & mask} << 32;
    rstate.smem  = rstate_memory;

    return rstate;
}

} // namespace jpeggpu

#endif // JPEGGPU_DECODE_HUFFMAN_READER_HPP_
