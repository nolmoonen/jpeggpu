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

struct reader_state {
    const uint8_t* data;
    const uint8_t* data_end;
    int32_t cache; // new bits are at the least significant positions
    int cache_num_bits;
};

/// \brief Loads the next eight bits.
__device__ void load_byte(reader_state& rstate)
{
    assert(rstate.data < rstate.data_end);
    assert(rstate.cache_num_bits + 8 < 32);

    // byte stuffing and restart markers are removed beforehand, padding in front
    //   of restart markers is not
    const uint8_t next_byte = *(rstate.data++);
    rstate.cache            = (rstate.cache << 8) | next_byte;
    rstate.cache_num_bits += 8;
}

/// \brief If there are enough bits in the input stream, loads `num_bits` into cache.
__device__ void load_bits(reader_state& rstate, int num_bits)
{
    while (rstate.cache_num_bits < num_bits) {
        if (rstate.data >= rstate.data_end) {
            return; // no more data to load
        }

        load_byte(rstate);
    }
}

/// \brief Peeks `num_bits` from cache, does not remove them.
///   Assumes enough bits are present.
__device__ int select_bits(reader_state& rstate, int num_bits)
{
    assert(num_bits < 31);
    assert(rstate.cache_num_bits >= num_bits);

    // upper bits are zero
    return rstate.cache >> (rstate.cache_num_bits - num_bits);
}

/// \brief Removes `num_bits` from cache.
__device__ void discard_bits(reader_state& rstate, int num_bits)
{
    assert(rstate.cache_num_bits >= num_bits);
    // set discarded bits to zero (upper bits in the cache)
    rstate.cache = rstate.cache & ((1 << (rstate.cache_num_bits - num_bits)) - 1);
    rstate.cache_num_bits -= num_bits;
}

/// \brief Get the Huffman category from stream.
///
/// \tparam do_discard Whether to discard the bits that were read in the process.
/// \param[out] length Number of bits read.
template <bool do_discard = true>
uint8_t __device__ get_category(reader_state& rstate, int& length, const huffman_table& table)
{
    load_bits(rstate, 16);

    // due to possibly guessing the huffman table wrong, there may not be enough bits left
    const int max_bits = min(rstate.cache_num_bits, 16);
    if (max_bits == 0) {
        // exit if there are no bits
        length = 0;
        return 0;
    }
    int i;
    int32_t code;
    for (i = 0; i < max_bits; ++i) {
        code                    = select_bits(rstate, i + 1);
        const bool is_last_iter = i == (max_bits - 1);
        if (code <= table.maxcode[i] || is_last_iter) {
            break;
        }
    }
    assert(1 <= i + 1 && i + 1 <= 16);
    // termination condition: 1 <= i + 1 <= 16, i + 1 is number of bits
    if constexpr (do_discard) {
        discard_bits(rstate, i + 1);
    }
    length        = i + 1;
    const int idx = table.valptr[i] + (code - table.mincode[i]);
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

__device__ void decode_next_symbol_dc(
    reader_state& rstate, int& length, int& symbol, int& run_length, const huffman_table& table)
{
    int category_length    = 0;
    const uint8_t category = get_category(rstate, category_length, table);

    if (category == 0) {
        // coeff is zero
        length     = category_length;
        symbol     = 0;
        run_length = 0;
        return;
    }

    assert(0 < category && category <= 16);
    load_bits(rstate, category);
    // there might not be `category` bits left
    if (rstate.cache_num_bits < category) {
        // eat all remaining so the `decode_subsequence` loop does not get stuck
        length = category_length + rstate.cache_num_bits;
        discard_bits(rstate, rstate.cache_num_bits);
        symbol     = 0; // arbitrary symbol
        run_length = 0;
        return;
    }
    const int offset = select_bits(rstate, category);
    discard_bits(rstate, category);
    const int value = get_value(category, offset);

    length     = category_length + category;
    symbol     = value;
    run_length = 0;
};

__device__ void decode_next_symbol_ac(
    reader_state& rstate,
    int& length,
    int& symbol,
    int& run_length,
    const huffman_table& table,
    int z)
{
    int category_length = 0;
    const uint8_t s     = get_category(rstate, category_length, table);
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

    assert(0 < category && category <= 16);
    load_bits(rstate, category);
    // there might not be `category` bits left
    if (rstate.cache_num_bits < category) {
        // eat all remaining so the `decode_subsequence` loop does not get stuck
        length = category_length + rstate.cache_num_bits;
        discard_bits(rstate, rstate.cache_num_bits);
        symbol     = 0; // arbitrary symbol
        run_length = 0; // arbitrary length
        return;
    }
    const int offset = select_bits(rstate, category);
    discard_bits(rstate, category);
    const int value = get_value(category, offset);

    length     = category_length + category;
    symbol     = value;
    run_length = run;
};

/// \brief Extracts coefficients from the bitstream while switching between DC and AC Huffman tables.
///
/// \param[inout] rstate Reader state.
/// \param[out] length The number of processed bits. Will be non-zero.
/// \param[out] symbol The decoded coefficient.
/// \param[out] run_length The run-length of zeroes that preceeds the coefficient.
/// \param[in] table_dc DC Huffman table.
/// \param[in] table_ac AC Huffman table.
/// \param[in] z Current index in the zig-zag sequence.
__device__ void decode_next_symbol(
    reader_state& rstate,
    int& length,
    int& symbol,
    int& run_length,
    const huffman_table& table_dc,
    const huffman_table& table_ac,
    int z)
{
    if (z == 0) {
        decode_next_symbol_dc(rstate, length, symbol, run_length, table_dc);
    } else {
        decode_next_symbol_ac(rstate, length, symbol, run_length, table_ac, z);
    }
    assert(length > 0);
}

struct const_state {
    const uint8_t* scan;
    const uint8_t* scan_end;
    const segment* segments;
    const int* segment_indices;
    const int num_segments;
    const huffman_table* dc_0;
    const huffman_table* ac_0;
    const huffman_table* dc_1;
    const huffman_table* ac_1;
    const huffman_table* dc_2;
    const huffman_table* ac_2;
    const huffman_table* dc_3;
    const huffman_table* ac_3;
    int c0_inc_prefix; // inclusive prefix of number of JPEG blocks in MCU
    int c1_inc_prefix;
    int c2_inc_prefix;
    int c3_inc_prefix;
    int num_data_units_in_mcu;
    int num_components;
    int num_data_units;
    int num_mcus_in_segment;
};
static_assert(std::is_trivially_copyable_v<const_state>);

/// \brief Algorithm 2.
///
/// \tparam is_overflow Whether `i` was decoded by another thread already. TODO word this better.
/// \tparam do_write Whether to write the coefficients to the output buffer.
/// \param i Global subsequence index.
/// \param segment_info Segment info for the segment that subsequence `i` is in.
/// \param segment_idx The index of the segment subsequence `i` is in.
template <bool is_overflow, bool do_write>
__device__ subsequence_info decode_subsequence(
    int i,
    int16_t* out,
    subsequence_info* s_info,
    const const_state& cstate,
    const segment& segment_info,
    int segment_idx)
{
    assert(i >= segment_info.subseq_offset);
    const int i_rel = i - segment_info.subseq_offset;

    subsequence_info info;
    // start of i-th subsequence
    info.p = i_rel * subsequence_size;
    info.n = 0;
    info.c = 0; // start from the first data unit of the Y component
    info.z = 0;

    reader_state rstate;
    // subsequence_size is multiple of eight, info.p is multiple of eight
    rstate.data = cstate.scan + segment_info.subseq_offset * subsequence_size_bytes + (info.p / 8);
    rstate.data_end = cstate.scan + (segment_info.subseq_offset + segment_info.subseq_count) *
                                        subsequence_size_bytes;
    rstate.cache          = 0;
    rstate.cache_num_bits = 0;

    int position_in_output = 0;
    if constexpr (do_write) {
        // offset in pixels
        int segment_offset = segment_idx * cstate.num_mcus_in_segment *
                             cstate.num_data_units_in_mcu * data_unit_size;
        // subsequence_info.n is relative to segment
        position_in_output = segment_offset + s_info[i].n;
    }
    if constexpr (is_overflow) {
        info.p = s_info[i - 1].p;
        // do not load `n` here, to achieve that `s_info.n` is the number of decoded symbols
        //   only for each subsequence (and not an aggregate)
        info.c = s_info[i - 1].c;
        info.z = s_info[i - 1].z;

        // overflowing from saved state, restore the reader state
        rstate.data =
            cstate.scan + segment_info.subseq_offset * subsequence_size_bytes + (info.p / 8);
        const int in_cache = (8 - (info.p % 8)) % 8;
        if (in_cache > 0) {
            rstate.cache          = *(rstate.data++);
            rstate.cache_num_bits = 8;
            discard_bits(rstate, 8 - in_cache);
        }
    }

    const int end_subseq  = (i_rel + 1) * subsequence_size; // first bit in next subsequence
    const int end_segment = segment_info.subseq_count * subsequence_size; // bit count in segment
    subsequence_info last_symbol; // the last detected codeword
    while (info.p < min(end_subseq, end_segment)) {
        // check if we have all blocks. this is needed since the scan is padded to a 8-bit multiple
        //   (so info.p cannot reliably be used to determine if the loop should break)
        //   this problem is excerbated by restart intevals, where padding occurs more frequently
        if (do_write && position_in_output >= (segment_idx + 1) * cstate.num_mcus_in_segment *
                                                  cstate.num_data_units_in_mcu * data_unit_size) {
            break;
        }

        last_symbol = info;

        int length     = 0;
        int symbol     = 0;
        int run_length = 0;
        const huffman_table* dc;
        const huffman_table* ac;
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

        // always returns length > 0 if there are bits in `rstate` to ensure progress
        decode_next_symbol(rstate, length, symbol, run_length, *dc, *ac, info.z);
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

    return last_symbol;
}

/// \brief Decode all subsequences once without synchronizing.
__global__ void decode_subsequences(
    subsequence_info* s_info, int num_subsequences, const_state cstate)
{
    const int subseq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (subseq_idx >= num_subsequences) {
        return;
    }

    // obtain the segment info for this subsequence
    const int segment_idx  = cstate.segment_indices[subseq_idx];
    const segment seg_info = cstate.segments[segment_idx];

    subsequence_info info = decode_subsequence<false, false>(
        subseq_idx, nullptr, s_info, cstate, seg_info, segment_idx);
    s_info[subseq_idx].p = info.p;
    // paper text does not mention `n` should be stored here, but if not storing `n`
    //   the first subsequence info's `n` will not be initialized. for simplicity, store all
    s_info[subseq_idx].n = info.n;
    s_info[subseq_idx].c = info.c;
    s_info[subseq_idx].z = info.z;
}

struct logical_and {
    __device__ bool operator()(const bool& lhs, const bool& rhs) { return lhs && rhs; }
};

/// \brief Synchronize between sequences. Each thread handles one sequence,
///   the last sequence requires no handling. Meaning is changed w.r.t. paper!

/// \brief Intra sequence synchronization (alg-3:05-23).
///   Each thread handles one subsequence at a time. Starting from each unique subsequence,
///   decode one subsequence at a time until the result is equal to the result of a different
///   thread having decoded that subsequence. If that is the case, the result is correct and
///   this thread is done.
///
/// \tparam block_size "b", the number of adjacent subsequences that form a sequence,
///   equal to the block size of this kernel.

/// \brief Synchronizes subsequences in multiple contiguous arrays of subsequences.
///   Each thread handles such a contiguous array.
///
///   W.r.t. the paper, `size_in_subsequences` of 1 and `block_size` equal to "b" will be
///     equivalent to "intersequence synchronization".
///
///
///
/// \tparam size_in_subsequences Amount of contigous subsequences are synchronized.
/// \tparam block_size Block size of the kernel.
template <int size_in_subsequences, int block_size>
__global__ void sync_subsequences(
    subsequence_info* s_info, int num_subsequences, const_state cstate)
{
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
    if (subseq_idx_from < num_subsequences) {
        // segment index from the subsequence we are flowing from
        segment_idx = cstate.segment_indices[subseq_idx_from];
        assert(segment_idx < cstate.num_segments);
        seg_info = cstate.segments[segment_idx];
        // index of the final subsequence for this segment
        const int subseq_last_idx_segment = seg_info.subseq_offset + seg_info.subseq_count;
        assert(subseq_idx_from <= subseq_last_idx_segment);
        end = min(end, subseq_last_idx_segment);
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
            info = decode_subsequence<true, false>(
                subseq_idx, nullptr, s_info, cstate, seg_info, segment_idx);
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

__global__ void decode_write(
    int16_t* out, subsequence_info* s_info, int num_subsequences, const_state cstate)
{
    const int si = blockIdx.x * blockDim.x + threadIdx.x;
    if (si >= num_subsequences) {
        return;
    }

    const int segment_idx  = cstate.segment_indices[si];
    const segment seg_info = cstate.segments[segment_idx];

    // only first thread does not do overflow
    constexpr bool do_write = true;
    if (si == seg_info.subseq_offset) {
        decode_subsequence<false, do_write>(si, out, s_info, cstate, seg_info, segment_idx);
    } else {
        decode_subsequence<true, do_write>(si, out, s_info, cstate, seg_info, segment_idx);
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
    huffman_table* (&d_huff_tables)[max_huffman_count][HUFF_COUNT],
    stack_allocator& allocator,
    cudaStream_t stream)
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
        d_huff_tables[info.components[0].dc_idx][HUFF_DC],
        d_huff_tables[info.components[0].ac_idx][HUFF_AC],
        d_huff_tables[info.components[1].dc_idx][HUFF_DC],
        d_huff_tables[info.components[1].ac_idx][HUFF_AC],
        d_huff_tables[info.components[2].dc_idx][HUFF_DC],
        d_huff_tables[info.components[2].ac_idx][HUFF_AC],
        d_huff_tables[info.components[3].dc_idx][HUFF_DC],
        d_huff_tables[info.components[3].ac_idx][HUFF_AC],
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
    if (do_it) {
        log("decoding %d subsequences\n", num_subsequences);
        decode_subsequences<<<num_sequences, num_subsequences_in_sequence, 0, stream>>>(
            d_s_info, num_subsequences, cstate);
        JPEGGPU_CHECK_CUDA(cudaGetLastError());
    }

    // synchronize intra sequence/inter subsequence
    if (do_it && num_subsequences > 1) {
        log("intra sync of %d blocks of %d subsequences\n",
            num_sequences,
            num_subsequences_in_sequence);
        sync_subsequences<1, num_subsequences_in_sequence>
            <<<num_sequences, num_subsequences_in_sequence, 0, stream>>>(
                d_s_info, num_subsequences, cstate);
        JPEGGPU_CHECK_CUDA(cudaGetLastError());
    }

    // synchronize intra supersequence/inter sequence
    constexpr int num_sequences_in_supersequence = 512; // configurable
    const int num_supersequences =
        ceiling_div(num_sequences, static_cast<unsigned int>(num_sequences_in_supersequence));
    if (do_it && num_sequences > 1) {
        log("intra sync of %d blocks of %d subsequences\n",
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
        log("byte stream is larger than max supported (%d bytes)\n", max_byte_size);
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
        decode_write<<<num_sequences, num_subsequences_in_sequence, 0, stream>>>(
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
    huffman_table* (&)[max_huffman_count][HUFF_COUNT],
    stack_allocator&,
    cudaStream_t);

template jpeggpu_status jpeggpu::decode_scan<true>(
    const jpeg_stream&,
    const uint8_t*,
    const segment*,
    const int*,
    int16_t*,
    const struct scan&,
    huffman_table* (&)[max_huffman_count][HUFF_COUNT],
    stack_allocator&,
    cudaStream_t);
