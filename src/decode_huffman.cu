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

// Current scheme:
// - Hierarchical, intra sync and inter sync just like the paper.
//   Biggest difference w.r.t. the paper is that no multiple inter launches
//   are needed since the syncing is performed in lockstep since there is
//   only one block (and block syncs are introduced). Having one thread block
//   for inter-sequence synchronization seems sufficient for even the largest images.
// Alternative schemes:
// - Within intra sync kernel, wait for the next block to become available.
//   This can be done by sleeping and checking a global flag for each block.
//   Waiting only for the next block is simple, the caveat is that in theory
//   a block can overflow into all subsequent blocks, causing the implementation
//   to not be very simple. Waiting only for the next block seemed to give slightly
//   beter performance than the standard scheme.
// - Do a fixed number of iterations for every block, e.g. four, and set some
//   flag for each subsequence whether it's synced. For the sequences that are
//   not synced, look back until a synced sequence is found and decode from there.
// - Have some "halo" for each intra-sequence thread block of threads that only
//   decode subsequences for the purpose of synchronization.

#include "decode_dc.hpp"
#include "decode_destuff.hpp"
#include "decode_huffman.hpp"
#include "decode_huffman_reader.hpp"
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
    /// \brief Bit position in scan after decoding subsequence.
    ///   If `p` is a multiple of the subsequence size in bits, it will be inside the subsequence
    ///   and the distance between `p` and the subsequence boundary is less than 32 bits (because an
    ///   encoded symbol is at most 32 bits).
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
    /// \brief In the case that `p` is not a multiple of the subsequence size in bits, contains the data bits
    ///   between `p` and the boundary in the most significant positions.
    uint32_t cache;
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
    int spectral_start;
    int spectral_end;
};
// check that struct can be copied to device
// TODO not sufficient, C-style array is also trivially copyable not not supported as kernel argument
static_assert(std::is_trivially_copyable_v<const_state>);

/// \brief Typedef for the maximum amount of Huffman tables for a scan.
using huffman_tables = huffman_table[max_comp_count * HUFF_COUNT];

template <int block_size>
__device__ void load_huffman_table(const huffman_table& table_global, huffman_table& table_shared)
{
    // assert that loading a word at a time is valid
    static_assert(sizeof(huffman_table) % 4 == 0 && sizeof(huffman_table::entry) % 4 == 0);
    constexpr int num_words            = sizeof(huffman_table) / 4;
    constexpr int num_words_per_thread = // TODO not efficient since num_words < block_size
        ceiling_div(num_words, static_cast<unsigned int>(block_size));
    for (int i = 0; i < num_words_per_thread; ++i) {
        const int idx = block_size * i + threadIdx.x;
        if (idx < num_words) {
            reinterpret_cast<uint32_t*>(&table_shared)[idx] =
                reinterpret_cast<const uint32_t*>(&table_global)[idx];
        }
    }
}

/// \brief Load Huffman tables from global memory into shared.
///
/// TODO this always loads the maximum possible amount of tables.
///   in the common case (one for luminance and one for chrominance),
///   this results in twice as many loads as needed
template <int bs>
__device__ void load_huffman_tables(const const_state& cstate, huffman_tables& tables_shared)
{
    assert(cstate.num_components > 0);
    const bool load_dc = cstate.spectral_start == 0;
    const bool load_ac = cstate.spectral_end >= 1;
    if (load_dc) load_huffman_table<bs>(cstate.huffman_tables[cstate.dc_0], tables_shared[0]);
    if (load_ac) load_huffman_table<bs>(cstate.huffman_tables[cstate.ac_0], tables_shared[1]);
    if (cstate.num_components <= 1) return;
    if (load_dc) load_huffman_table<bs>(cstate.huffman_tables[cstate.dc_1], tables_shared[2]);
    if (load_ac) load_huffman_table<bs>(cstate.huffman_tables[cstate.ac_1], tables_shared[3]);
    if (cstate.num_components <= 2) return;
    if (load_dc) load_huffman_table<bs>(cstate.huffman_tables[cstate.dc_2], tables_shared[4]);
    if (load_ac) load_huffman_table<bs>(cstate.huffman_tables[cstate.ac_2], tables_shared[5]);
    if (cstate.num_components <= 3) return;
    if (load_dc) load_huffman_table<bs>(cstate.huffman_tables[cstate.dc_3], tables_shared[6]);
    if (load_ac) load_huffman_table<bs>(cstate.huffman_tables[cstate.ac_3], tables_shared[7]);
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
/// \param[in] cstate
/// \param[in] segment_idx The segment index for this subsequence.
/// \param[inout] rstate
/// \param[in] tables
/// \param[in] info
/// \param[in] position_in_output Offset in `out` where the decoded coefficients of this subsequence
///   will be written. Only relevant if `do_write` is true.
template <bool do_write, typename reader_state>
__device__ subsequence_info decode_subsequence(
    int subseq_idx_rel,
    int16_t* __restrict__ out,
    const const_state& cstate,
    int segment_idx,
    reader_state& rstate,
    const huffman_tables& tables,
    subsequence_info info,
    int position_in_output = 0)
{
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
        int table_idx;
        if (info.c < cstate.c0_inc_prefix) {
            table_idx = 0;
        } else if (info.c < cstate.c1_inc_prefix) {
            table_idx = 1;
        } else if (info.c < cstate.c2_inc_prefix) {
            table_idx = 2;
        } else {
            table_idx = 3;
        }
        const huffman_table& table_dc = tables[table_idx * HUFF_COUNT + HUFF_DC];
        const huffman_table& table_ac = tables[table_idx * HUFF_COUNT + HUFF_AC];

        // decode_next_symbol uses at most 32 bits. if fewer than 32 bits are available, append with zeroes
        uint32_t data = load_32_bits(rstate);

        // always returns length > 0 if there are bits in `rstate` to ensure progress
        decode_next_symbol<do_write>(length, symbol, run_length, data, table_dc, table_ac, info.z);

        if (info.p + length > end_subseq) {
            // Reading the next symbol will be outside the subsequence, so stop here.
            //   Save the remaining data for uniform data loading in the write kernel.
            info.cache = data;
            break;
        }

        // commit
        discard_bits(rstate, length);

        if (do_write) {
            // TODO could make a separate kernel for this
            position_in_output += run_length;
            const int data_unit_idx    = position_in_output / data_unit_size;
            const int idx_in_data_unit = position_in_output % data_unit_size;
            out[data_unit_idx * data_unit_size + order_natural[idx_in_data_unit]] = symbol;
            ++position_in_output;
        }

        info.p += length;
        if (!do_write) info.n += run_length + 1;
        info.z += run_length + 1;

        if (info.z >= cstate.spectral_end + 1) {
            // the data unit is complete
            info.z = cstate.spectral_start;
            ++info.c;
            // spectral_end is inclusive
            const int skip = 64 - (cstate.spectral_end + 1 - cstate.spectral_start);
            info.n += skip;
            position_in_output += skip;

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
    __shared__ subsequence_info s_info_shared[block_size];

    __shared__ huffman_tables tables;
    load_huffman_tables<block_size>(cstate, tables);
    __syncthreads();

    // Don't load all subsequences of the block into memory at the beginning as it
    //   hurts occupancy to have so much shared memory.
    using reader_state = reader_state_thread_cache<block_size>;
    __shared__ typename reader_state::smem_type rstate_memory;

    assert(blockDim.x == block_size);
    const int block_off        = blockDim.x * blockIdx.x;
    const int subseq_idx_begin = block_off + threadIdx.x;

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

        rstate = rstate_from_subseq_start<block_size>(
            cstate.scan, seg_info, rstate_memory, subeq_idx_begin_rel);

        subsequence_info info;
        // start of i-th subsequence
        info.p = subeq_idx_begin_rel * subsequence_size;
        info.n = cstate.spectral_start;
        info.c = 0;
        info.z = cstate.spectral_start;

        // paper text does not mention `n` should be stored here, but if not storing `n`
        //   the first subsequence info's `n` will not be initialized. for simplicity, store all
        s_info_shared[threadIdx.x] = decode_subsequence<false>(
            subeq_idx_begin_rel, nullptr, cstate, segment_idx, rstate, tables, info);
    }
    __syncthreads();

    __shared__ bool is_block_done;
    using block_reduce = cub::BlockReduce<bool, block_size>;
    __shared__ typename block_reduce::TempStorage temp_storage;

    bool is_synced = false;
    bool do_write  = true;
    for (int i = 0; i < block_size; ++i) {
        // index of subsequence we are flowing into
        const int subseq_idx = subseq_idx_begin + 1 + i;
        subsequence_info info;
        if (subseq_idx < end && !is_synced) {
            assert(seg_info.subseq_offset <= subseq_idx);
            const int subseq_idx_rel = subseq_idx - seg_info.subseq_offset;

            assert(block_off <= subseq_idx - 1 && subseq_idx - 1 - block_off < block_size);
            subsequence_info old_info;
            old_info.p = s_info_shared[subseq_idx - 1 - block_off].p;
            // do not load `n` here, to achieve that `s_info.n` is the number of decoded symbols
            //   only for each subsequence (and not an aggregate)
            old_info.n = cstate.spectral_start;
            old_info.c = s_info_shared[subseq_idx - 1 - block_off].c;
            old_info.z = s_info_shared[subseq_idx - 1 - block_off].z;

            info = decode_subsequence<false>(
                subseq_idx_rel, nullptr, cstate, segment_idx, rstate, tables, old_info);
            assert(block_off <= subseq_idx && subseq_idx - block_off < block_size);
            const subsequence_info& stored_info = s_info_shared[subseq_idx - block_off];
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
        if (do_write) {
            assert(block_off <= subseq_idx && subseq_idx - block_off < block_size);
            s_info_shared[subseq_idx - block_off] = info;
        }
        __syncthreads(); // await s_info writes and is_block_done write
        if (is_block_done) break;
    }
    if (subseq_idx_begin < num_subsequences) {
        s_info[subseq_idx_begin] = s_info_shared[threadIdx.x];
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
    load_huffman_tables<block_size>(cstate, tables);
    __syncthreads();

    using reader_state = reader_state_thread_cache<block_size>;
    __shared__ typename reader_state::smem_type rstate_memory;

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

        rstate = rstate_from_subseq_overflow<block_size>(
            cstate.scan, seg_info, rstate_memory, s_info[subseq_idx_from].p);
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

            subsequence_info old_info;
            old_info.p = s_info[subseq_idx - 1].p;
            // do not load `n` here, to achieve that `s_info.n` is the number of decoded symbols
            //   only for each subsequence (and not an aggregate)
            old_info.n = cstate.spectral_start;
            old_info.c = s_info[subseq_idx - 1].c;
            old_info.z = s_info[subseq_idx - 1].z;

            info = decode_subsequence<false>(
                subseq_idx_rel, nullptr, cstate, segment_idx, rstate, tables, old_info);
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
    load_huffman_tables<block_size>(cstate, tables);

    using reader_state = reader_state_all_subsequences<block_size>;
    __shared__ typename reader_state::smem_type rstate_memory;
    load_all_subsequences<block_size>(cstate.scan, num_subsequences, rstate_memory);

    __syncthreads();

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
        reader_state rstate =
            rstate_from_subseq_start<block_size>(seg_info, rstate_memory, subseq_idx_rel);

        subsequence_info info;
        info.p = 0;
        // n is not used in writing
        info.c = 0;
        info.z = cstate.spectral_start;

        decode_subsequence<do_write>(
            subseq_idx_rel, out, cstate, segment_idx, rstate, tables, info, position_in_output);
    } else {
        subsequence_info info = s_info[subseq_idx - 1];

        reader_state rstate = rstate_from_subseq_overflow<block_size>(
            seg_info, rstate_memory, subseq_idx_rel, info.p, info.cache);

        decode_subsequence<do_write>(
            subseq_idx_rel, out, cstate, segment_idx, rstate, tables, info, position_in_output);
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
    for (int c = 0; c < scan.num_components; ++c) {
        const scan_component& scan_comp = scan.scan_components[c];
        num_data_units += (scan_comp.data_size.x / jpeggpu::data_unit_vector_size) *
                          (scan_comp.data_size.y / jpeggpu::data_unit_vector_size);
    }

    // alg-1:05
    subsequence_info* d_s_info;
    JPEGGPU_CHECK_STAT(
        allocator.reserve<do_it>(&d_s_info, num_subsequences * sizeof(subsequence_info)));

    // block count in MCU
    int c_offsets[max_comp_count] = {};
    int c_offset                  = 0;
    for (int c = 0; c < scan.num_components; ++c) {
        const component& comp = info.components[scan.scan_components[c].component_idx];
        c_offset += comp.ss.x * comp.ss.y;
        c_offsets[c] = c_offset;
    }

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
        scan.scan_components[0].dc_idx,
        scan.scan_components[0].ac_idx,
        scan.scan_components[1].dc_idx,
        scan.scan_components[1].ac_idx,
        scan.scan_components[2].dc_idx,
        scan.scan_components[2].ac_idx,
        scan.scan_components[3].dc_idx,
        scan.scan_components[3].ac_idx,
        c_offsets[0],
        c_offsets[1],
        c_offsets[2],
        c_offsets[3],
        scan.num_data_units_in_mcu,
        scan.num_components,
        num_data_units,
        info.restart_interval != 0 ? info.restart_interval : scan.num_mcus.x * scan.num_mcus.y,
        scan.spectral_start,
        scan.spectral_end};

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
