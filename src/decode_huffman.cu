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
    /// \brief The data unit index in the MCU. With the sampling factors, the color component
    ///   can be inferred. The paper calls this field "the current color component",
    ///   but merely checking the color component will not suffice.
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
uint8_t __device__ get_category(
    reader_state& rstate, int& length, const jpeggpu::huffman_table& table)
{
    load_bits(rstate, 16);

    // due to possibly guessing the huffman table wrong, there may not be enough bits left
    const int max_bits = min(rstate.cache_num_bits, 16);
    if (max_bits == 0) {
        // exit if there are no bits
        length = 0;
        return 0;
    }
    int i, code;
    for (i = 0; i < max_bits; ++i) {
        code                    = select_bits(rstate, i + 1);
        const bool is_last_iter = i == (max_bits - 1);
        if (code <= table.maxcode[i + 1] || is_last_iter) {
            break;
        }
    }
    assert(1 <= i + 1 && i + 1 <= 16);
    // termination condition: 1 <= i + 1 <= 16, i + 1 is number of bits
    if constexpr (do_discard) {
        discard_bits(rstate, i + 1);
    }
    length        = i + 1;
    const int idx = table.valptr[i + 1] + (code - table.mincode[i + 1]);
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
    reader_state& rstate,
    int& length,
    int& symbol,
    int& run_length,
    const jpeggpu::huffman_table& table_dc,
    const jpeggpu::huffman_table& table_ac,
    int z)
{
    int category_length    = 0;
    const uint8_t category = get_category(rstate, category_length, table_dc);

    if (category != 0) {
        assert(0 < category && category < 17);
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

        length = category_length + category;
        symbol = value;
    } else {
        length = category_length;
        symbol = 0;
    }

    // peek next to determine run (is always AC)
    {
        int len;
        const uint8_t s    = get_category<false>(rstate, len, table_ac);
        const int run      = s >> 4;
        const int category = s & 0xf;

        if (category != 0) {
            run_length = run;
        } else {
            // either EOB or ZRL, which are treated as a symbol,
            //   so there are no zeros inbetween the DC value and EOB or ZRL
            run_length = 0;
        }
    }
};

__device__ void decode_next_symbol_ac(
    reader_state& rstate,
    int& length,
    int& symbol,
    int& run_length,
    const jpeggpu::huffman_table& table,
    int z)
{
    int category_length = 0;
    const uint8_t s     = get_category(rstate, category_length, table);
    const int run       = s >> 4;
    const int category  = s & 0xf;

    if (category != 0) {
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

        length = category_length + category;
        symbol = value;

        if (z + 1 <= 63) { // note: z already includes `run`
            // next value is ac coefficient, peek next to determine run
            int len;
            const uint8_t s    = get_category<false>(rstate, len, table);
            const int run      = s >> 4;
            const int category = s & 0xf;

            if (category != 0) {
                run_length = run;
            } else {
                // EOB or ZRL
                run_length = 0;
            }
        } else {
            // next table is dc
            run_length = 0;
        }
    } else {
        if (run == 15) {
            length     = category_length;
            symbol     = 0; // ZRL
            run_length = 15;

            if (z + 1 + 15 <= 63) {
                // there is an AC symbol after the ZRL
                int len;
                const uint8_t s    = get_category<false>(rstate, len, table);
                const int run      = s >> 4;
                const int category = s & 0xf;

                if (category != 0) {
                    run_length += run;
                } else {
                    // EOB or ZRL
                }
            } else {
                // next is dc
            }
        } else {
            length     = category_length;
            symbol     = 0; // EOB
            run_length = 63 - z;
        }
    }
};

/// \brief Extracts coefficients from the bitstream while switching between DC and AC Huffman
/// tables.
///
/// - If symbol equals ZRL, 15 will be returned for run_length.
/// - If symbol equals EOB, 63 - z will be returned for run_length, with z begin the current
///     index in the zig-zag sequence.
///
/// \param[inout] rstate
/// \param[out] length The number of processed bits. Will be non-zero.
/// \param[out] symbol The decoded coefficient, provided the code was not EOB or ZRL.
/// \param[out] run_length The run-length of zeroes which the coefficient is followed by.
/// \param[in] table
/// \param[in] z Current index in the zig-zag sequence.
__device__ void decode_next_symbol(
    reader_state& rstate,
    int& length,
    int& symbol,
    int& run_length,
    const jpeggpu::huffman_table& table_dc,
    const jpeggpu::huffman_table& table_ac,
    int z)
{
    if (z == 0) {
        decode_next_symbol_dc(rstate, length, symbol, run_length, table_dc, table_ac, z);
    } else {
        decode_next_symbol_ac(rstate, length, symbol, run_length, table_ac, z);
    }
}

enum class component {
    y, // Y (YCbCR) or C (CMYK)
    cb, // Cb (YCbCR) or M (CMYK)
    cr, // Cr (YCbCR) or Y (CMYK)
    k // k (CMYK)
};

/// \brief Infer image components based on data unit index `c` (in MCU).
__device__ component calc_component(int ssx, int ssy, int c, int num_components)
{
    const int num_luma_data_units = ssx * ssy;

    if (c < num_luma_data_units) {
        return component::y;
    }

    assert(num_components > 1);

    if (c == num_luma_data_units) {
        return component::cb;
    }

    assert(num_components > 2);

    if (c == num_luma_data_units + 1) {
        return component::cr;
    }

    assert(num_components > 3);

    if (c == num_luma_data_units + 2) {
        return component::k;
    }

    assert(false);
}

struct const_state {
    const uint8_t* scan;
    const uint8_t* scan_end;
    const jpeggpu::huffman_table* table_luma_dc;
    const jpeggpu::huffman_table* table_luma_ac;
    const jpeggpu::huffman_table* table_chroma_dc;
    const jpeggpu::huffman_table* table_chroma_ac;
    int ssx;
    int ssy;
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
    const jpeggpu::segment_info& segment_info,
    int segment_idx)
{
    assert(i >= segment_info.subseq_offset);
    const int i_rel = i - segment_info.subseq_offset;

    subsequence_info info;
    // start of i-th subsequence
    info.p = i_rel * jpeggpu::subsequence_size;
    info.n = 0;
    info.c = 0; // start from the first data unit of the Y component
    info.z = 0;

    reader_state rstate;
    // subsequence_size is multiple of eight
    rstate.data           = cstate.scan + segment_info.begin + (info.p / 8);
    rstate.data_end       = cstate.scan + segment_info.end;
    rstate.cache          = 0;
    rstate.cache_num_bits = 0;

    const int num_data_units_in_mcu = cstate.ssx * cstate.ssy + cstate.num_components - 1;

    int position_in_output = 0;
    if constexpr (do_write) {
        position_in_output = s_info[i].n + segment_idx * cstate.num_mcus_in_segment *
                                               num_data_units_in_mcu * jpeggpu::data_unit_size;
    }
    if constexpr (is_overflow) {
        info.p = s_info[i - 1].p;
        // FIXME previous testing did not take into account the `do_write` version? maybe
        // do not load `n` here, to achieve that `s_info.n` is the number of decoded symbols
        //   only for each subsequence (and not an aggregate)
        info.c = s_info[i - 1].c;
        info.z = s_info[i - 1].z;

        // overflowing from saved state, restore the reader state
        rstate.data        = cstate.scan + segment_info.begin + (info.p / 8);
        const int in_cache = (8 - (info.p % 8)) % 8;
        if (in_cache > 0) {
            rstate.cache          = *(rstate.data++);
            rstate.cache_num_bits = 8;
            discard_bits(rstate, 8 - in_cache);
        }
    }

    const int end_subseq = (i_rel + 1) * jpeggpu::subsequence_size; // first bit in next subsequence
    const int end_segment = (segment_info.end - segment_info.begin) * 8; // bit count in segment
    subsequence_info last_symbol; // the last detected codeword
    while (info.p < min(end_subseq, end_segment)) {
        // check if we have all blocks. this is needed since the scan is padded to a 8-bit multiple
        //   (so info.p cannot reliably be used to determine if the loop should break)
        //   this problem is excerbated by restart intevals, where padding occurs more frequently
        if (do_write && position_in_output >= (segment_idx + 1) * cstate.num_mcus_in_segment *
                                                  num_data_units_in_mcu * jpeggpu::data_unit_size) {
            break;
        }

        last_symbol = info;

        const component comp =
            calc_component(cstate.ssx, cstate.ssy, info.c, cstate.num_components);
        int length     = 0;
        int symbol     = 0;
        int run_length = 0;
        // always returns length > 0 if there are bits in `rstate` to ensure progress
        decode_next_symbol(
            rstate,
            length,
            symbol,
            run_length,
            comp == component::y ? *cstate.table_luma_dc : *cstate.table_chroma_dc,
            comp == component::y ? *cstate.table_luma_ac : *cstate.table_chroma_ac,
            info.z);
        if (do_write) {
            // TODO could make a separate kernel for this
            out[position_in_output / 64 * 64 + jpeggpu::order_natural[position_in_output % 64]] =
                symbol;
        }
        if (do_write) {
            position_in_output += run_length + 1;
        }
        info.p += length;
        info.n += run_length + 1;
        info.z += run_length + 1;

        if (info.z >= 64) {
            // the data unit is complete
            info.z = 0;
            ++info.c;

            // ass-1
            const int num_data_units_in_mcu = cstate.ssx * cstate.ssy + (cstate.num_components - 1);
            if (info.c >= num_data_units_in_mcu) {
                // mcu is complete
                info.c = 0;
            }
        }
    }

    return last_symbol;
}

/// \brief Intra sequence synchronization (alg-3:05-23).
///   Each thread handles one subsequence at a time. Starting from each unique subsequence,
///   decode one subsequence at a time until the result is equal to the result of a different
///   thread having decoded that subsequence. If that is the case, the result is correct and
///   this thread is done.
///
/// \tparam block_size "b", the number of adjacent subsequences that form a sequence.
template <int block_size>
__global__ void sync_intra_sequence(
    subsequence_info* s_info,
    int num_subsequences,
    const_state cstate,
    const jpeggpu::segment_info* segment_infos,
    const int* segment_indices)
{
    assert(block_size == blockDim.x);
    const int bi = blockIdx.x;
    const int si = threadIdx.x;

    const int seq_global = bi * block_size; // first subsequence index in sequence
    int subseq_global    = seq_global + si; // sequence index of thread

    if (subseq_global >= num_subsequences) {
        return;
    }

    // obtain the segment info for this subsequence
    const int segment_idx                = segment_indices[subseq_global];
    const jpeggpu::segment_info seg_info = segment_infos[segment_idx];
    // index of the final subsequence in this segment
    const int final_index                = seg_info.subseq_offset +
                            ceiling_div(
                                seg_info.end - seg_info.begin,
                                static_cast<unsigned int>(jpeggpu::subsequence_size_bytes)) -
                            1;
    assert(subseq_global <= final_index);

    bool synchronized = false;
    // index of the last subsequence in this sequence
    //   paper uses `+ block_size` but `end` should be an index
    const int end     = min(seq_global + block_size - 1, final_index);
    // alg-3:10
    {
        subsequence_info info = decode_subsequence<false, false>(
            subseq_global, nullptr, s_info, cstate, seg_info, segment_idx);
        s_info[subseq_global].p = info.p;
        // paper text does not mention `n` should be stored here, but if not storing `n`
        //   the first subsequence info's `n` will not be initialized. for simplicity, store all
        s_info[subseq_global].n = info.n;
        s_info[subseq_global].c = info.c;
        s_info[subseq_global].z = info.z;
    }
    __syncthreads(); // wait until result of next subsequence is available
    ++subseq_global;
    while (!synchronized && subseq_global <= end) {
        // overflow, so `decode_subsequence` reads from `s_info[subseq_global - 1]`
        subsequence_info info = decode_subsequence<true, false>(
            subseq_global, nullptr, s_info, cstate, seg_info, segment_idx);
        if (info.p == s_info[subseq_global].p && info.c == s_info[subseq_global].c &&
            info.z == s_info[subseq_global].z) {
            // the decoding process of this thread has found the same "outcome" for the
            //   `subseq_global`th subsequence as the next thread
            synchronized = true;
        }
        // (not in paper) wait until other threads have finished reading segment `subseq_global`
        __syncthreads();
        s_info[subseq_global] = info;
        ++subseq_global;
        // make the new result of segment `subseq_global` available to all threads
        __syncthreads();
    }
}

/// \brief Each thread handles one sequence, the last sequence is not handled.
template <int block_size>
__global__ void sync_inter_sequence(
    subsequence_info* s_info,
    int num_subsequences,
    const_state cstate,
    uint8_t* sequence_not_synced,
    int num_sequences,
    const jpeggpu::segment_info* segment_infos,
    const int* segment_indices)
{
    // Thread with global id `tid` handles sequence `tid + 1 = i`, since sequence zero needs no work
    const int bi = blockDim.x * blockIdx.x + threadIdx.x;
    if (bi >= num_sequences - 1) {
        return;
    }

    // first subsequence of sequence `i`, which we are flowing into from the last
    //   subsequence of sequence `i - 1` (i.e. `tid`)
    int subseq_global = (bi + 1) * block_size;

    // obtain the segment info for last subsequence of sequence `tid`
    const int segment_idx                = segment_indices[subseq_global - 1];
    const jpeggpu::segment_info seg_info = segment_infos[segment_idx];
    // index of the final subsequence in this segment
    const int final_index                = seg_info.subseq_offset +
                            ceiling_div(
                                seg_info.end - seg_info.begin,
                                static_cast<unsigned int>(jpeggpu::subsequence_size_bytes)) -
                            1;
    assert(subseq_global - 1 <= final_index);
    if (subseq_global - 1 == final_index) {
        // the last subsequence of sequence `tid` is the last subsequence in the segment,
        //   no overflow is needed
        sequence_not_synced[bi] = false;
        return;
    }

    bool synchronized = false;
    // index of the last subsequence in sequence i
    //   paper uses `+ block_size` but `end` should be an index
    const int end     = min(subseq_global + block_size - 1, final_index);
    while (!synchronized && subseq_global <= end) {
        // overflow, so `decode_subsequence` reads from `s_info[subseq_global - 1]`
        subsequence_info info = decode_subsequence<true, false>(
            subseq_global, nullptr, s_info, cstate, seg_info, segment_idx);
        if (info.p == s_info[subseq_global].p && info.c == s_info[subseq_global].c &&
            info.z == s_info[subseq_global].z) {
            // this means a synchronization point was found
            synchronized            = true;
            // paper uses `bi - 1`, we use bi since the array has `num_sequence - 1` elements
            sequence_not_synced[bi] = false; // set sequence i to "synced"
        }
        // TODO each sequence reads from the last subsequence of the previous sequence,
        //   which can still be written to by the previous sequence. the sync here will not prevent
        //   this problem if two problematic sequences are in different blocks.
        //   maybe this can be prevented by kernel waiting instead of a reduction on outputs?
        __syncthreads();
        s_info[subseq_global] = info;
        ++subseq_global;
        __syncthreads();
    }
}

__global__ void decode_write(
    int16_t* out,
    subsequence_info* s_info,
    int num_subsequences,
    const_state cstate,
    const jpeggpu::segment_info* segment_infos,
    const int* segment_indices)
{
    const int si = blockIdx.x * blockDim.x + threadIdx.x;
    if (si >= num_subsequences) {
        return;
    }

    const int segment_idx                = segment_indices[si];
    const jpeggpu::segment_info seg_info = segment_infos[segment_idx];

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
    jpeggpu::logger& logger,
    jpeggpu::reader& reader,
    uint8_t* d_scan_destuffed,
    segment_info* d_segment_infos,
    int* d_segment_indices,
    int16_t* d_out,
    const struct jpeggpu::scan& scan,
    stack_allocator& allocator,
    cudaStream_t stream)
{
    // TODO move allocations to decoder init
    // TODO move copy to separate function
    // TODO pick the right huffman table
    jpeggpu::huffman_table* d_huff;
    JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(&d_huff, 4 * sizeof(jpeggpu::huffman_table)));
    if (do_it) {
        CHECK_CUDA(cudaMemcpyAsync(
            d_huff + 0,
            reader.h_huff_tables[0][jpeggpu::HUFF_DC],
            sizeof(*reader.h_huff_tables[0][jpeggpu::HUFF_DC]),
            cudaMemcpyHostToDevice,
            stream));
        CHECK_CUDA(cudaMemcpyAsync(
            d_huff + 1,
            reader.h_huff_tables[0][jpeggpu::HUFF_AC],
            sizeof(*reader.h_huff_tables[0][jpeggpu::HUFF_AC]),
            cudaMemcpyHostToDevice,
            stream));
        CHECK_CUDA(cudaMemcpyAsync(
            d_huff + 2,
            reader.h_huff_tables[1][jpeggpu::HUFF_DC],
            sizeof(*reader.h_huff_tables[1][jpeggpu::HUFF_DC]),
            cudaMemcpyHostToDevice,
            stream));
        CHECK_CUDA(cudaMemcpyAsync(
            d_huff + 3,
            reader.h_huff_tables[1][jpeggpu::HUFF_AC],
            sizeof(*reader.h_huff_tables[1][jpeggpu::HUFF_AC]),
            cudaMemcpyHostToDevice,
            stream));
    }

    const int num_subsequences = scan.num_subsequences; // "N"
    constexpr int block_size   = 256; // "b", size in subsequences
    const int num_sequences =
        ceiling_div(num_subsequences, static_cast<unsigned int>(block_size)); // "B"
    if (do_it) {
        logger("num_subsequences: %d num_sequences: %d\n", num_subsequences, num_sequences);
    }

    // alg-1:01
    size_t total_data_size = 0;
    int num_data_units     = 0;
    for (int c = 0; c < reader.jpeg_stream.num_components; ++c) {
        total_data_size += reader.jpeg_stream.data_sizes_x[c] * reader.jpeg_stream.data_sizes_y[c];
        num_data_units +=
            (reader.jpeg_stream.data_sizes_x[c] / 8) * (reader.jpeg_stream.data_sizes_y[c] / 8);
    }

    // alg-1:05
    subsequence_info* d_s_info;
    JPEGGPU_CHECK_STAT(
        allocator.reserve<do_it>(&d_s_info, num_subsequences * sizeof(subsequence_info)));

    const const_state cstate = {
        d_scan_destuffed,
        // TODO this is not the end of data, but the end of allocation. the final subsequence may read garbage.
        d_scan_destuffed + (scan.end - scan.begin),
        d_huff + 0,
        d_huff + 1,
        d_huff + 2,
        d_huff + 3,
        reader.jpeg_stream.css.x[0],
        reader.jpeg_stream.css.y[0],
        reader.jpeg_stream.num_components,
        num_data_units,
        reader.jpeg_stream.restart_interval != 0
            ? reader.jpeg_stream.restart_interval
            : reader.jpeg_stream.num_mcus_x * reader.jpeg_stream.num_mcus_y};

    { // sync_decoders (Algorithm 3)
        if (do_it) {
            sync_intra_sequence<block_size><<<num_sequences, block_size, 0, stream>>>(
                d_s_info, num_subsequences, cstate, d_segment_infos, d_segment_indices);
            CHECK_CUDA(cudaGetLastError());
        }

        if (num_sequences > 1) {
            // the meaning of this array is flipped w.r.t. the paper,
            //   a one is produced if not synced
            uint8_t* d_sequence_not_synced = nullptr;
            JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(
                &d_sequence_not_synced, (num_sequences - 1) * sizeof(uint8_t)));
            if (do_it) {
                CHECK_CUDA(cudaMemsetAsync( // all are initialized to "not synced"
                    d_sequence_not_synced,
                    static_cast<uint8_t>(true),
                    (num_sequences - 1) * sizeof(uint8_t),
                    stream));
            }

            int* d_num_unsynced_sequence = nullptr;
            JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(&d_num_unsynced_sequence, sizeof(int)));

            void* d_tmp_storage      = nullptr;
            size_t tmp_storage_bytes = 0;
            CHECK_CUDA(cub::DeviceReduce::Sum(
                d_tmp_storage,
                tmp_storage_bytes,
                d_sequence_not_synced,
                d_num_unsynced_sequence,
                num_sequences - 1,
                stream));

            JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(&d_tmp_storage, tmp_storage_bytes));

            int h_num_unsynced_sequence;
            do {
                constexpr int block_size_inter = 256; // does not need to be `block_size`
                const int num_inter_blocks =
                    ceiling_div(num_sequences, static_cast<unsigned int>(block_size_inter));
                if (do_it) {
                    sync_inter_sequence<block_size>
                        <<<num_inter_blocks, block_size_inter, 0, stream>>>(
                            d_s_info,
                            num_subsequences,
                            cstate,
                            d_sequence_not_synced,
                            num_sequences,
                            d_segment_infos,
                            d_segment_indices);
                    CHECK_CUDA(cudaGetLastError());
                    logger("inter sequence sync done\n");
                }

                if (do_it) {
                    CHECK_CUDA(cub::DeviceReduce::Sum(
                        d_tmp_storage,
                        tmp_storage_bytes,
                        d_sequence_not_synced,
                        d_num_unsynced_sequence,
                        num_sequences - 1,
                        stream));
                    // TODO this requires synchronization, try kernel that uses atomics to perform inter sequence sync
                    CHECK_CUDA(cudaMemcpy(
                        &h_num_unsynced_sequence,
                        d_num_unsynced_sequence,
                        sizeof(int),
                        cudaMemcpyDeviceToHost));
                    logger("unsynced: %d\n", h_num_unsynced_sequence);
                }
            } while (!do_it && h_num_unsynced_sequence);
        }
    }

    // TODO consider SoA or do in-place
    // alg-1:07-08
    {
        subsequence_info* d_reduce_out;
        JPEGGPU_CHECK_STAT(
            allocator.reserve<do_it>(&d_reduce_out, num_subsequences * sizeof(subsequence_info)));
        if (do_it) {
            // TODO debug to satisfy initcheck
            CHECK_CUDA(cudaMemsetAsync(
                d_reduce_out, 0, num_subsequences * sizeof(subsequence_info), stream));
        }

        const subsequence_info init_value{0, 0, 0, 0};
        void* d_tmp_storage      = nullptr;
        size_t tmp_storage_bytes = 0;
        CHECK_CUDA(cub::DeviceScan::ExclusiveScanByKey(
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
            CHECK_CUDA(cudaMemsetAsync(d_tmp_storage, 0, tmp_storage_bytes, stream));
        }

        if (do_it) {
            CHECK_CUDA(cub::DeviceScan::ExclusiveScanByKey(
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
            CHECK_CUDA(cudaGetLastError());
        }
    }

    if (do_it) {
        // alg-1:09-15
        decode_write<<<num_sequences, block_size, 0, stream>>>(
            d_out, d_s_info, num_subsequences, cstate, d_segment_infos, d_segment_indices);
        CHECK_CUDA(cudaGetLastError());
    }

    return JPEGGPU_SUCCESS;
}

template jpeggpu_status jpeggpu::decode_scan<false>(
    jpeggpu::logger&,
    jpeggpu::reader&,
    uint8_t*,
    segment_info*,
    int*,
    int16_t*,
    const struct jpeggpu::scan&,
    stack_allocator&,
    cudaStream_t);

template jpeggpu_status jpeggpu::decode_scan<true>(
    jpeggpu::logger&,
    jpeggpu::reader&,
    uint8_t*,
    segment_info*,
    int*,
    int16_t*,
    const struct jpeggpu::scan&,
    stack_allocator&,
    cudaStream_t);
