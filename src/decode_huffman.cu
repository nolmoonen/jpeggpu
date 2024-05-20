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

#include "decode_dc.hpp"
#include "decode_destuff.hpp"
#include "decode_huffman.hpp"
#include "decode_transpose.hpp"
#include "decoder_defs.hpp"
#include "defs.hpp"
#include "marker.hpp"
#include "reader.hpp"
#include "util.hpp"

#include <jpeggpu/jpeggpu.h>

#include <cub/block/block_scan.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/thread/thread_operators.cuh>

#include <cuda_runtime.h>

#include <cassert>
#include <type_traits>
#include <vector>

using namespace jpeggpu;

namespace {

/// \brief Contains all required information about the last synchronization point for the
///   subsequence. All information is relative to the segment.
struct subsequence_info {
    /// \brief Bit(!) position in scan. "Location of the last detected codeword."
    ///   TODO size_t?
    int p;
    /// \brief The number of decoded symbols.
    int n;
    /// \brief The data unit index in the MCU. Combined with the sampling factors, the color component
    ///   can be inferred. The paper calls this field "the current color component",
    ///   but merely checking the color component does not suffice.
    int c;
    /// \brief Zig-zag index.
    int z;
};

/// \brief Reader state.
struct reader_state {
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
};

/// \brief Helper struct to pass constant inputs to the kernels.
struct const_state {
    /// \brief Pointer to first data byte, will always be aligned to cudaMalloc alignment.
    const uint8_t* scan;
    /// \brief Pointer to byte following the last data byte.
    const uint8_t* scan_end;
    /// \brief Segment info.
    const segment* segments;
    /// \brief For each subsequence, its segment index.
    const int* segment_indices;
    /// \brief Number of segments in the scan.
    ///   TODO zero for a scan without restart markers, could optimize?
    const int num_segments;
    /// \brief Huffman tables.
    const huffman_table* huffman_tables;
    int dc_0; /// DC Huffman table index for component 0.
    int ac_0; /// AC Huffman table index for component 0.
    int dc_1; /// DC Huffman table index for component 1.
    int ac_1; /// AC Huffman table index for component 1.
    int dc_2; /// DC Huffman table index for component 2.
    int ac_2; /// AC Huffman table index for component 2.
    int dc_3; /// DC Huffman table index for component 3.
    int ac_3; /// AC Huffman table index for component 3.
    int c0_inc_prefix; // Inclusive prefix of number of JPEG blocks in MCU for component 0.
    int c1_inc_prefix; // Inclusive prefix of number of JPEG blocks in MCU for component 1.
    int c2_inc_prefix; // Inclusive prefix of number of JPEG blocks in MCU for component 2.
    int c3_inc_prefix; // Inclusive prefix of number of JPEG blocks in MCU for component 3.
    int num_data_units_in_mcu;
    int num_components;
    int num_data_units;
    int num_mcus_in_segment;
};
// check that struct can be copied to device
// TODO not sufficient, C-style array is also trivially copyable not not supported as kernel argument
static_assert(std::is_trivially_copyable_v<const_state>);

/// \brief Typedef for the maximum amount of Huffman tables for a scan.
using huffman_tables = huffman_table[max_huffman_count * HUFF_COUNT];

/// \brief Typedef for one cache per thread.
template <int block_size>
using cache = ulonglong4[block_size];

/// \brief Load Huffman tables from global memory into shared.
///
/// TODO this always loads the maximum possible amount of tables.
///   in the common case (one for luminance and one for chrominance),
///   this results in twice as many loads as needed
template <int block_size>
__device__ void load_huffman_tables(
    const huffman_table* tables_global, huffman_tables& tables_shared)
{
    // assert that loading a word at a time is valid
    static_assert(sizeof(huffman_table) % 4 == 0 && sizeof(huffman_table::entry) % 4 == 0);
    constexpr int num_words = sizeof(tables_shared) / 4;
    constexpr int num_words_per_thread =
        ceiling_div(num_words, static_cast<unsigned int>(block_size));
    for (int i = 0; i < num_words_per_thread; ++i) {
        const int idx = block_size * i + threadIdx.x;
        if (idx < num_words) {
            reinterpret_cast<uint32_t*>(tables_shared)[idx] =
                reinterpret_cast<const uint32_t*>(tables_global)[idx];
        }
    }
    __syncthreads();
}

/// \brief Load the next sector (128 bits) into shared memory cache.
__device__ void global_load_uint4(reader_state& rstate, uint4* shared)
{
    for (int i = 0; i < sizeof(uint4) / sizeof(uint32_t); ++i) {
        uint32_t val                           = reinterpret_cast<const uint32_t*>(rstate.data)[i];
        val                                    = __byte_perm(val, uint32_t{0}, uint32_t{0x0123});
        reinterpret_cast<uint32_t*>(shared)[i] = val;
    }
    rstate.data += sizeof(uint4);
}

/// \brief Return the next 32 bits from cache, padded with zero bits if not enough available.
__device__ uint32_t load_32_bits(reader_state& rstate)
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
                uint32_t val = reinterpret_cast<const uint32_t*>(rstate.data)[i];
                val          = __byte_perm(val, uint32_t{0}, uint32_t{0x0123});
                reinterpret_cast<uint32_t*>(rstate.cache)[4 + i] = val;
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

/// \brief Constructs the reader state from the start of a subsequence (fills the cache).
///
/// \param[in] cstate
/// \param[in] segment_info Segment info for this subsequence.
/// \param[in] cache Pointer to shared memory cache.
/// \param[in] subseq_idx_rel Subsequence index relative to segment.
__device__ reader_state rstate_from_subseq_start(
    const const_state& cstate, const segment& segment_info, ulonglong4* cache, int subseq_idx_rel)
{
    reader_state rstate;
    rstate.data =
        cstate.scan + (segment_info.subseq_offset + subseq_idx_rel) * subsequence_size_bytes;
    rstate.data_end = cstate.scan + (segment_info.subseq_offset + segment_info.subseq_count) *
                                        subsequence_size_bytes;
    rstate.cache     = cache;
    rstate.cache_idx = 0;

    // assert aligned load is valid
    assert((rstate.data - cstate.scan) % sizeof(uint4) == 0);
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
/// \param[in] cstate
/// \param[in] segment_info Segment info for this subsequence.
/// \param[in] cache Pointer to shared memory cache.
/// \param[in] p Bit offset in segment.
__device__ reader_state rstate_from_subseq_overflow(
    const const_state& cstate, const segment& segment_info, ulonglong4* cache, int p)
{
    reader_state rstate;
    // loaded in chunks of 128, each chunk has 16 bytes
    const int byte_offset = (p / 128) * (sizeof(uint4));
    rstate.data = cstate.scan + segment_info.subseq_offset * subsequence_size_bytes + byte_offset;
    rstate.data_end = cstate.scan + (segment_info.subseq_offset + segment_info.subseq_count) *
                                        subsequence_size_bytes;
    rstate.cache     = cache;
    rstate.cache_idx = p - byte_offset * 8;

    // assert aligned load is valid
    assert((rstate.data - cstate.scan) % sizeof(uint4) == 0);
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

/// \brief Returns the oldest (most significant) `num_bits` from `data`.
__device__ uint32_t u32_select_bits(uint32_t data, int num_bits)
{
    assert(num_bits <= 32);
    return data >> (32 - num_bits);
}

/// \brief Discards the oldest (most significant) `num_bits` from `data`.
__device__ uint32_t u32_discard_bits(uint32_t data, int num_bits)
{
    assert(num_bits <= 32);
    return data << num_bits;
}

/// \brief Get the Huffman category from stream. Reads at most 16 bits.
///
/// \param[in] data Holds at least 16 data bits in the most significant positions.
/// \param[out] length Number of bits read.
/// \param[in] table
uint8_t __device__ get_category(uint32_t data, int& length, const huffman_table& table)
{
    int i;
    int32_t code;
    huffman_table::entry entry;
    for (i = 0; i < 16; ++i) {
        code                    = u32_select_bits(data, i + 1);
        const bool is_last_iter = i == 15;
        entry                   = table.entries[i];
        if (code <= entry.maxcode || is_last_iter) {
            break;
        }
    }
    assert(1 <= i + 1 && i + 1 <= 16);
    // termination condition: 1 <= i + 1 <= 16, i + 1 is number of bits
    length        = i + 1;
    const int idx = entry.valptr + (code - entry.mincode);
    if (idx < 0 || 256 <= idx) {
        // found a value that does not make sense. this can happen if the wrong huffman
        //   table is used. return arbitrary value
        return 0;
    }
    return table.huffval[idx];
}

__device__ int get_value(int num_bits, int code)
{
    // TODO leftshift negative value is UB
    return code < ((1 << num_bits) >> 1) ? (code + ((-1) << num_bits) + 1) : code;
}

template <bool do_write>
__device__ void decode_next_symbol_dc(
    int& length, int& symbol, int& run_length, uint32_t data, const huffman_table& table)
{
    int category_length    = 0;
    const uint8_t category = get_category(data, category_length, table);

    if (category == 0) {
        // coeff is zero
        length     = category_length;
        symbol     = 0;
        run_length = 0;
        return;
    }

    data = u32_discard_bits(data, category_length);

    length = category_length + category;
    if (do_write) {
        assert(0 < category && category <= 16);
        const int offset = u32_select_bits(data, category);
        const int value  = get_value(category, offset);
        symbol           = value;
    }
    run_length = 0;
};

template <bool do_write>
__device__ void decode_next_symbol_ac(
    int& length, int& symbol, int& run_length, uint32_t data, const huffman_table& table, int z)
{
    int category_length = 0;
    const uint8_t s     = get_category(data, category_length, table);
    const int run       = s >> 4;
    const int category  = s & 0xf;

    if (category == 0) {
        // coeff is zero
        length = category_length;
        symbol = 0;
        if (run == 15) run_length = 15; // ZRL
        else run_length = 63 - z; // EOB
        return;
    }

    data = u32_discard_bits(data, category_length);

    length = category_length + category;
    if (do_write) {
        assert(0 < category && category <= 16);
        const int offset = u32_select_bits(data, category);
        const int value  = get_value(category, offset);
        symbol           = value;
    }
    run_length = run;
};

/// \brief Extracts coefficients from the bitstream while switching between DC and AC Huffman tables.
///
/// \param[out] length The number of processed bits. Will be non-zero.
/// \param[out] symbol The decoded coefficient.
/// \param[out] run_length The run-length of zeroes that preceeds the coefficient.
/// \param[in] data The next 32 bits of encoded data, possibly appended with zeroes if OOB.
/// \param[in] table_dc DC Huffman table.
/// \param[in] table_ac AC Huffman table.
/// \param[in] z Current index in the zig-zag sequence.
template <bool do_write>
__device__ void decode_next_symbol(
    int& length,
    int& symbol,
    int& run_length,
    uint32_t data,
    const huffman_table& table_dc,
    const huffman_table& table_ac,
    int z)
{
    if (z == 0) {
        decode_next_symbol_dc<do_write>(length, symbol, run_length, data, table_dc);
    } else {
        decode_next_symbol_ac<do_write>(length, symbol, run_length, data, table_ac, z);
    }
    assert(length > 0);
}

/// \brief Decodes a subsequence (related to paper alg-2).
///
/// \tparam is_overflow Whether decoding (and decoder state) is continued from a previous subsequence.
/// \tparam do_write Whether to write the coefficients to the output buffer.
///
/// \param[in] subseq_idx_rel Subsequence index relative to segment.
/// \param[out] out Output memory for coefficients, may be `nullptr` if `do_write` is false.
/// \param[in] s_info Array of all subsequence infos.
/// \param[in] cstate
/// \param[in] segment_info The segment info for this subsequence.
/// \param[in] segment_idx The segment index for this subsequence.
/// \param[inout] rstate
/// \param[in] tables
/// \param[in] position_in_output Offset in `out` where the decoded coefficients of this subsequence
///   will be written. Only relevant if `do_write` is true.
template <bool is_overflow, bool do_write>
__device__ subsequence_info decode_subsequence(
    int subseq_idx_rel,
    int16_t* __restrict__ out,
    const subsequence_info* __restrict__ s_info,
    const const_state& cstate,
    const segment& segment_info,
    int segment_idx,
    reader_state& rstate,
    const huffman_tables& tables,
    int position_in_output = 0)
{
    subsequence_info info;
    if constexpr (is_overflow) {
        const int subseq_idx = segment_info.subseq_offset + subseq_idx_rel;
        info.p               = s_info[subseq_idx - 1].p;
        // do not load `n` here, to achieve that `s_info.n` is the number of decoded symbols
        //   only for each subsequence (and not an aggregate)
        info.n = 0;
        info.c = s_info[subseq_idx - 1].c;
        info.z = s_info[subseq_idx - 1].z;

    } else {
        // start of i-th subsequence
        info.p = subseq_idx_rel * subsequence_size;
        info.n = 0;
        info.c = 0; // start from the first data unit of the Y component
        info.z = 0;
    }

    const int end_subseq = (subseq_idx_rel + 1) * subsequence_size; // first bit in next subsequence
    while (true) {
        // check if we have all blocks. this is needed since the scan is padded to a 8-bit multiple
        //   (so info.p cannot reliably be used to determine if the loop should break)
        //   this problem is excerbated by restart intevals, where padding occurs more frequently
        if (do_write && position_in_output >= (segment_idx + 1) * cstate.num_mcus_in_segment *
                                                  cstate.num_data_units_in_mcu * data_unit_size) {
            break;
        }

        int length     = 0;
        int symbol     = 0;
        int run_length = 0;
        int dc;
        int ac;
        if (info.c < cstate.c0_inc_prefix) {
            dc = cstate.dc_0;
            ac = cstate.ac_0;
        } else if (info.c < cstate.c1_inc_prefix) {
            dc = cstate.dc_1;
            ac = cstate.ac_1;
        } else if (info.c < cstate.c2_inc_prefix) {
            dc = cstate.dc_2;
            ac = cstate.ac_2;
        } else {
            dc = cstate.dc_3;
            ac = cstate.ac_3;
        }
        const huffman_table& table_dc = tables[dc * HUFF_COUNT + HUFF_DC];
        const huffman_table& table_ac = tables[ac * HUFF_COUNT + HUFF_AC];

        // decode_next_symbol uses at most 32 bits. if fewer than 32 bits are available, append with zeroes
        uint32_t data = load_32_bits(rstate);

        // always returns length > 0 if there are bits in `rstate` to ensure progress
        decode_next_symbol<do_write>(length, symbol, run_length, data, table_dc, table_ac, info.z);

        if (info.p + length > end_subseq) {
            // reading the next symbol will be outside the subsequence
            break;
        }

        // commit
        rstate.cache_idx += length;

        if (do_write) {
            // TODO could make a separate kernel for this
            position_in_output += run_length;
            const int data_unit_idx    = position_in_output / data_unit_size;
            const int idx_in_data_unit = position_in_output % data_unit_size;
            out[data_unit_idx * data_unit_size + order_natural[idx_in_data_unit]] = symbol;
            ++position_in_output;
        }

        info.p += length;
        info.n += run_length + 1;
        info.z += run_length + 1;

        if (info.z >= 64) {
            // the data unit is complete
            info.z = 0;
            ++info.c;

            if (info.c >= cstate.num_data_units_in_mcu) {
                // mcu is complete
                info.c = 0;
            }
        }
    }

    return info;
}

struct logical_and {
    __device__ bool operator()(const bool& lhs, const bool& rhs) { return lhs && rhs; }
};

/// \brief Synchronize all subsequences in a subsequence (relates to paper alg-3:05-23).
///   Thread i decodes subsequence i, decoding subsequent subsequences until the state
///   is equal to the state of a thread with a higher index. If the states are the same
///   synchronization is achieved.
///
/// \tparam block_size The constant "b", the number of adjacent subsequences that form a sequence,
///   equal to the block size of this kernel.
template <int block_size>
__global__ void sync_intra_sequence(
    subsequence_info* __restrict__ s_info, int num_subsequences, const_state cstate)
{
    __shared__ huffman_tables tables;
    load_huffman_tables<block_size>(cstate.huffman_tables, tables);

    __shared__ cache<block_size> block_cache;
    ulonglong4* thread_cache = &(block_cache[threadIdx.x]);

    assert(blockDim.x == block_size);
    const int subseq_idx_begin = blockDim.x * blockIdx.x + threadIdx.x;

    // first subsequence index owned/read by the next thread block
    const int subseq_idx_from_next_block = (blockIdx.x + 1) * block_size;

    // last index dictated by block and JPEG stream
    int end = min(subseq_idx_from_next_block, num_subsequences);

    // since all threads must partipate in thread sync, some threads will be assigned to
    //  a subsequence out of bounds
    int segment_idx; // segment index of subseq_idx_begin
    segment seg_info; // segment info of subseq_idx_begin
    reader_state rstate;
    if (subseq_idx_begin < num_subsequences) {
        segment_idx = cstate.segment_indices[subseq_idx_begin];
        assert(segment_idx < cstate.num_segments);
        seg_info = cstate.segments[segment_idx];

        const int segment_subseq_end = seg_info.subseq_offset + seg_info.subseq_count;
        assert(subseq_idx_begin < segment_subseq_end);
        end = min(end, segment_subseq_end);

        const int subeq_idx_begin_rel = subseq_idx_begin - seg_info.subseq_offset;

        rstate = rstate_from_subseq_start(cstate, seg_info, thread_cache, subeq_idx_begin_rel);

        subsequence_info info = decode_subsequence<false, false>(
            subeq_idx_begin_rel, nullptr, s_info, cstate, seg_info, segment_idx, rstate, tables);
        // paper text does not mention `n` should be stored here, but if not storing `n`
        //   the first subsequence info's `n` will not be initialized. for simplicity, store all
        s_info[subseq_idx_begin] = info;
    }
    __syncthreads();

    __shared__ bool is_block_done;
    using block_reduce = cub::BlockReduce<bool, block_size>;
    __shared__ typename block_reduce::TempStorage temp_storage;

    // TODO could read in entire sequence into shared memory first

    bool is_synced = false;
    bool do_write  = true;
    for (int i = 0; i < block_size; ++i) {
        // index of subsequence we are flowing into
        const int subseq_idx = subseq_idx_begin + 1 + i;
        subsequence_info info;
        if (subseq_idx < end && !is_synced) {
            assert(seg_info.subseq_offset <= subseq_idx);
            const int subseq_idx_rel = subseq_idx - seg_info.subseq_offset;
            info                     = decode_subsequence<true, false>(
                subseq_idx_rel, nullptr, s_info, cstate, seg_info, segment_idx, rstate, tables);
            const subsequence_info& stored_info = s_info[subseq_idx];
            if (info.p == stored_info.p && info.c == stored_info.c && info.z == stored_info.z) {
                // synchronization is achieved: the decoding process of this thread has found
                //   the same "outcome" for the `subseq_idx`th subsequence as the stored result
                //   of a decoding process that started from an earlier point in the JPEG stream,
                //   meaning that this outcome is correct.
                is_synced = true;
            }
        } else {
            do_write = false;
        }
        bool is_thread_done      = is_synced || subseq_idx >= end;
        bool is_block_done_local = block_reduce(temp_storage).Reduce(is_thread_done, logical_and{});
        __syncthreads(); // await s_info reads
        if (threadIdx.x == 0) is_block_done = is_block_done_local;
        if (do_write) s_info[subseq_idx] = info;
        __syncthreads(); // await s_info writes and is_block_done write
        if (is_block_done) return;
    }
}

/// \brief Synchronizes subsequences in multiple contiguous arrays of subsequences,
///   (relates to paper alg-3:25-41). Each thread handles such a contiguous array.
///
///   W.r.t. the paper, `size_in_subsequences` of 1 and `block_size` equal to "b" will be
///     equivalent to "intersequence synchronization".
///
/// \tparam size_in_subsequences Amount of contigous subsequences are synchronized.
/// \tparam block_size Block size of the kernel.
template <int size_in_subsequences, int block_size>
__global__ void sync_subsequences(
    subsequence_info* __restrict__ s_info, int num_subsequences, const_state cstate)
{
    __shared__ huffman_tables tables;
    load_huffman_tables<block_size>(cstate.huffman_tables, tables);

    __shared__ cache<block_size> block_cache;
    ulonglong4* thread_cache = &(block_cache[threadIdx.x]);

    assert(blockDim.x == block_size);
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // subsequence index this thread is overflowing from, this will be the first s_info read
    const int subseq_idx_from = (tid + 1) * size_in_subsequences - 1;
    // overflowing to
    const int subseq_idx_to = subseq_idx_from + 1;

    // first subsequence index owned/read by the next thread block, follow from `subseq_idx_from`
    //   calculation, substituting `blockDim.x` by `blockDim.x + 1` and `threadIdx.x` by `0`
    const int subseq_idx_from_next_block =
        ((blockIdx.x + 1) * block_size + 1) * size_in_subsequences - 1;

    // last index dictated by block and JPEG stream
    int end = min(subseq_idx_from_next_block, num_subsequences);

    // since all threads must partipate, some threads will be assigned to
    //  a subsequence out of bounds
    int segment_idx;
    segment seg_info;
    reader_state rstate;
    if (subseq_idx_from < num_subsequences) {
        // segment index from the subsequence we are flowing from
        segment_idx = cstate.segment_indices[subseq_idx_from];
        assert(segment_idx < cstate.num_segments);
        seg_info = cstate.segments[segment_idx];
        // index of the final subsequence for this segment
        const int subseq_last_idx_segment = seg_info.subseq_offset + seg_info.subseq_count;
        assert(subseq_idx_from < subseq_last_idx_segment);
        end = min(end, subseq_last_idx_segment);

        rstate =
            rstate_from_subseq_overflow(cstate, seg_info, thread_cache, s_info[subseq_idx_from].p);
    }

    __shared__ bool is_block_done;
    using block_reduce = cub::BlockReduce<bool, block_size>;
    __shared__ typename block_reduce::TempStorage temp_storage;

    bool is_synced = false;
    bool do_write  = true;
    for (int i = 0; i < block_size * size_in_subsequences; ++i) {
        const int subseq_idx = subseq_idx_to + i;
        subsequence_info info;
        if (subseq_idx < end && !is_synced) {
            assert(seg_info.subseq_offset <= subseq_idx);
            const int subseq_idx_rel = subseq_idx - seg_info.subseq_offset;
            info                     = decode_subsequence<true, false>(
                subseq_idx_rel, nullptr, s_info, cstate, seg_info, segment_idx, rstate, tables);
            const subsequence_info& stored_info = s_info[subseq_idx];
            if (info.p == stored_info.p && info.c == stored_info.c && info.z == stored_info.z) {
                // synchronization is achieved: the decoding process of this thread has found
                //   the same "outcome" for the `subseq_idx`th subsequence as the stored result
                //   of a decoding process that started from an earlier point in the JPEG stream,
                //   meaning that this outcome is correct.
                is_synced = true;
            }
        } else {
            do_write = false;
        }
        bool is_thread_done      = is_synced || subseq_idx >= end;
        bool is_block_done_local = block_reduce(temp_storage).Reduce(is_thread_done, logical_and{});
        __syncthreads(); // await s_info reads
        if (threadIdx.x == 0) is_block_done = is_block_done_local;
        if (do_write) s_info[subseq_idx] = info;
        __syncthreads(); // await s_info writes and is_block_done write
        if (is_block_done) return;
    }
}

/// \brief Redo the decoding of each subsequence once, this time writing the coefficients
///   (relates to paper alg-1:9-14).
///
/// \tparam block_size The constant "b". TODO is that required?
template <int block_size>
__global__ void decode_write(
    int16_t* __restrict__ out,
    const subsequence_info* __restrict__ s_info,
    int num_subsequences,
    const_state cstate)
{
    __shared__ huffman_tables tables;
    load_huffman_tables<block_size>(cstate.huffman_tables, tables);

    __shared__ cache<block_size> block_cache;
    ulonglong4* thread_cache = &(block_cache[threadIdx.x]);

    const int subseq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (subseq_idx >= num_subsequences) {
        return;
    }

    const int segment_idx  = cstate.segment_indices[subseq_idx];
    const segment seg_info = cstate.segments[segment_idx];
    assert(seg_info.subseq_offset <= subseq_idx);
    const int subseq_idx_rel = subseq_idx - seg_info.subseq_offset;

    // offset in pixels
    const int segment_offset =
        segment_idx * cstate.num_mcus_in_segment * cstate.num_data_units_in_mcu * data_unit_size;
    // subsequence_info.n is relative to segment
    const int position_in_output = segment_offset + s_info[subseq_idx].n;

    // only first thread does not do overflow
    constexpr bool do_write = true;
    if (subseq_idx_rel == 0) {
        reader_state rstate = rstate_from_subseq_start(cstate, seg_info, thread_cache, 0);
        decode_subsequence<false, do_write>(
            subseq_idx_rel,
            out,
            s_info,
            cstate,
            seg_info,
            segment_idx,
            rstate,
            tables,
            position_in_output);
    } else {
        reader_state rstate =
            rstate_from_subseq_overflow(cstate, seg_info, thread_cache, s_info[subseq_idx - 1].p);
        decode_subsequence<true, do_write>(
            subseq_idx_rel,
            out,
            s_info,
            cstate,
            seg_info,
            segment_idx,
            rstate,
            tables,
            position_in_output);
    }
}

struct sum_subsequence_info {
    __device__ __forceinline__ subsequence_info
    operator()(const subsequence_info& a, const subsequence_info& b) const
    {
        // asserts in the comparison function are not great since CUB may execute the comparator on
        // garbage data if the block or warp is not completely full
        return {0, a.n + b.n, 0, 0};
    }
};

/// \brief Copy `subsequence_info::n` from `src` to `dst`.
__global__ void assign_sinfo_n(
    int num_subsequences, subsequence_info* dst, const subsequence_info* src)
{
    const int lid = blockDim.x * blockIdx.x + threadIdx.x;
    if (lid >= num_subsequences) {
        return;
    }

    assert(src[lid].n >= 0);
    dst[lid].n = src[lid].n;
}

} // namespace

template <bool do_it>
jpeggpu_status jpeggpu::decode_scan(
    const jpeg_stream& info,
    const uint8_t* d_scan_destuffed,
    const segment* d_segments,
    const int* d_segment_indices,
    int16_t* d_out,
    const struct scan& scan,
    huffman_table* d_huff_tables,
    stack_allocator& allocator,
    cudaStream_t stream,
    logger& logger)
{
    // "N": number of subsequences, determined by JPEG stream
    const int num_subsequences = scan.num_subsequences;

    // alg-1:01
    int num_data_units = 0;
    for (int c = 0; c < info.num_components; ++c) {
        num_data_units += (info.components[c].data_size_x / jpeggpu::data_unit_vector_size) *
                          (info.components[c].data_size_y / jpeggpu::data_unit_vector_size);
    }

    // alg-1:05
    subsequence_info* d_s_info;
    JPEGGPU_CHECK_STAT(
        allocator.reserve<do_it>(&d_s_info, num_subsequences * sizeof(subsequence_info)));

    // block count in MCU
    const int c0_count = info.components[0].ss_x * info.components[0].ss_y;
    const int c1_count = info.components[1].ss_x * info.components[1].ss_y;
    const int c2_count = info.components[2].ss_x * info.components[2].ss_y;
    const int c3_count = info.components[3].ss_x * info.components[3].ss_y;

    const const_state cstate = {
        d_scan_destuffed,
        // this is not the end of the destuffed data, but the end of the stuffed allocation.
        //   the final subsequence may read garbage bits (but not bytes).
        //   this can introduce additional (non-existent) symbols,
        //   but a check is in place to prevent writing more symbols than needed
        d_scan_destuffed + (scan.end - scan.begin),
        d_segments,
        d_segment_indices,
        scan.num_segments,
        d_huff_tables,
        info.components[0].dc_idx,
        info.components[0].ac_idx,
        info.components[1].dc_idx,
        info.components[1].ac_idx,
        info.components[2].dc_idx,
        info.components[2].ac_idx,
        info.components[3].dc_idx,
        info.components[3].ac_idx,
        c0_count,
        c0_count + c1_count,
        c0_count + c1_count + c2_count,
        c0_count + c1_count + c2_count + c3_count,
        info.num_data_units_in_mcu,
        info.num_components,
        num_data_units,
        info.restart_interval != 0 ? info.restart_interval : info.num_mcus_x * info.num_mcus_y};

    // decode all subsequences
    // "b", sequence size in number of subsequences, configurable
    constexpr int num_subsequences_in_sequence = 256;
    // "B", number of sequences
    const int num_sequences =
        ceiling_div(num_subsequences, static_cast<unsigned int>(num_subsequences_in_sequence));

    // synchronize intra sequence/inter subsequence
    if (do_it && num_subsequences > 1) {
        logger.log(
            "intra sync of %d blocks of %d subsequences\n",
            num_sequences,
            num_subsequences_in_sequence);
        sync_intra_sequence<num_subsequences_in_sequence>
            <<<num_sequences, num_subsequences_in_sequence, 0, stream>>>(
                d_s_info, num_subsequences, cstate);
        JPEGGPU_CHECK_CUDA(cudaGetLastError());
    }

    // synchronize intra supersequence/inter sequence
    constexpr int num_sequences_in_supersequence = 512; // configurable
    const int num_supersequences =
        ceiling_div(num_sequences, static_cast<unsigned int>(num_sequences_in_supersequence));
    if (do_it && num_sequences > 1) {
        logger.log(
            "intra sync of %d blocks of %d subsequences\n",
            num_supersequences,
            num_sequences_in_supersequence * num_subsequences_in_sequence);
        sync_subsequences<num_subsequences_in_sequence, num_sequences_in_supersequence>
            <<<num_supersequences, num_sequences_in_supersequence, 0, stream>>>(
                d_s_info, num_subsequences, cstate);
        JPEGGPU_CHECK_CUDA(cudaGetLastError());
    }

    if (num_supersequences > num_sequences_in_supersequence) {
        constexpr int max_byte_size =
            subsequence_size_bytes * num_subsequences_in_sequence * num_sequences_in_supersequence;
        logger.log("byte stream is larger than max supported (%d bytes)\n", max_byte_size);
        return JPEGGPU_NOT_SUPPORTED;
    }

    // TODO consider SoA or do in-place
    // alg-1:07-08
    subsequence_info* d_reduce_out;
    JPEGGPU_CHECK_STAT(
        allocator.reserve<do_it>(&d_reduce_out, num_subsequences * sizeof(subsequence_info)));
    if (do_it) {
        // TODO debug to satisfy initcheck
        JPEGGPU_CHECK_CUDA(
            cudaMemsetAsync(d_reduce_out, 0, num_subsequences * sizeof(subsequence_info), stream));
    }

    const subsequence_info init_value{0, 0, 0, 0};
    void* d_tmp_storage      = nullptr;
    size_t tmp_storage_bytes = 0;
    JPEGGPU_CHECK_CUDA(cub::DeviceScan::ExclusiveScanByKey(
        d_tmp_storage,
        tmp_storage_bytes,
        d_segment_indices, // d_keys_in
        d_s_info,
        d_reduce_out,
        sum_subsequence_info{},
        init_value,
        num_subsequences,
        cub::Equality{},
        stream));

    JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(&d_tmp_storage, tmp_storage_bytes));
    if (do_it) {
        // TODO debug to satisfy initcheck
        JPEGGPU_CHECK_CUDA(cudaMemsetAsync(d_tmp_storage, 0, tmp_storage_bytes, stream));
    }

    if (do_it) {
        JPEGGPU_CHECK_CUDA(cub::DeviceScan::ExclusiveScanByKey(
            d_tmp_storage,
            tmp_storage_bytes,
            d_segment_indices, // d_keys_in
            d_s_info,
            d_reduce_out,
            sum_subsequence_info{},
            init_value,
            num_subsequences,
            cub::Equality{},
            stream));
    }

    constexpr int block_size_assign = 256;
    const int grid_dim =
        ceiling_div(num_subsequences, static_cast<unsigned int>(block_size_assign));
    if (do_it) {
        assign_sinfo_n<<<grid_dim, block_size_assign, 0, stream>>>(
            num_subsequences, d_s_info, d_reduce_out);
        JPEGGPU_CHECK_CUDA(cudaGetLastError());
    }

    if (do_it) {
        // alg-1:09-15
        decode_write<num_subsequences_in_sequence>
            <<<num_sequences, num_subsequences_in_sequence, 0, stream>>>(
                d_out, d_s_info, num_subsequences, cstate);
        JPEGGPU_CHECK_CUDA(cudaGetLastError());
    }

    return JPEGGPU_SUCCESS;
}

template jpeggpu_status jpeggpu::decode_scan<false>(
    const jpeg_stream&,
    const uint8_t*,
    const segment*,
    const int*,
    int16_t*,
    const struct scan&,
    huffman_table*,
    stack_allocator&,
    cudaStream_t,
    logger&);

template jpeggpu_status jpeggpu::decode_scan<true>(
    const jpeg_stream&,
    const uint8_t*,
    const segment*,
    const int*,
    int16_t*,
    const struct scan&,
    huffman_table*,
    stack_allocator&,
    cudaStream_t,
    logger&);
