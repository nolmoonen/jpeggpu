#include "decode_gpu.hpp"
#include "defs.hpp"
#include "marker.hpp"
#include "reader.hpp"
#include "util.hpp"

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <jpeggpu/jpeggpu.h>

#include <cassert>
#include <type_traits>
#include <vector>

namespace {

/// TODO remove these assumptions
/// Assumption 1 (ass-1): all non-luma is subsampled with the same factor
/// Assumption 2 (ass-2): Huffman table mapping

/// \brief "s", subsequence size in 32 bits. Paper uses 4 or 32 depending on the quality of the
///   encoded image. TODO value of 4 seems to get the decoder stuck.
constexpr int chunk_size       = 8; ///< "s", in 32 bits
constexpr int subsequence_size = chunk_size * 32; ///< size in bits
// subsequence size is in bits, it makes it easier if it is a multiple of eight for data reading
static_assert(subsequence_size % 8 == 0);

/// \brief Contains all required information about the last syncrhonization point for the
///   subsequence.
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
        //   table is used. TODO is this the correct return value?
        return 0;
    }
    return table.huffval[idx];
}

__device__ int get_value(int num_bits, int code)
{
    return code < ((1 << num_bits) >> 1) ? (code + ((-1) << num_bits) + 1) : code;
}

/// \brief End of block, indicates all remaining coefficients in the data unit are zero.
constexpr int symbol_eob = INT32_MAX - 1;

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
            discard_bits(rstate, rstate.cache_num_bits);
            length     = category_length + rstate.cache_num_bits;
            symbol     = symbol_eob; // arbitrary symbol
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
        const int run      = (s >> 4);
        const int category = s & 0xf;

        if (category != 0) {
            run_length = run;
        } else {
            // either EOB or ZRL, which is treated as a symbol
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
    // s = (run, category)
    const uint8_t s     = get_category(rstate, category_length, table);
    const int run       = (s >> 4);
    const int category  = s & 0xf;

    if (category != 0) {
        assert(0 < category && category < 17);
        load_bits(rstate, category);
        // there might not be `category` bits left
        if (rstate.cache_num_bits < category) {
            // eat all remaining so the `decode_subsequence` loop does not get stuck
            discard_bits(rstate, rstate.cache_num_bits);
            length     = category_length + rstate.cache_num_bits;
            symbol     = symbol_eob; // arbitrary symbol
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
            {
                int len;
                const uint8_t s    = get_category<false>(rstate, len, table);
                const int run      = (s >> 4);
                const int category = s & 0xf;

                if (category != 0) {
                    run_length = run;
                } else {
                    // EOB or ZRL
                    run_length = 0;
                }
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

            if (z + 1 <= 63) {
                // there may be a symbol after the ZRL
                {
                    int len;
                    const uint8_t s    = get_category<false>(rstate, len, table);
                    const int run      = (s >> 4);
                    const int category = s & 0xf;

                    if (category != 0) {
                        run_length += run;
                    }
                }
            } else {
                // next is dc
            }
        } else {
            length     = category_length;
            symbol     = symbol_eob;
            run_length = 63 - z;
        }
    }
};

/// \brief Extracts coefficients from the bitstream while switching between DC and AC Huffman
/// tables.
///
/// - If symbol equals ZRL, 15 will be returned for run_length
/// - If symbol equals EOB, 63 - z will be returned for run_length, with z begin the current
///     index in the zig-zag sequence
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
};

static_assert(std::is_trivially_copyable_v<const_state>);

/// \brief Algorithm 2.
///
/// \tparam is_overflow Whether `i` was decoded by another thread already. TODO word this better.
/// \tparam do_write Whether to write the coefficients to the output buffer.
template <bool is_overflow, bool do_write>
__device__ subsequence_info
decode_subsequence(int i, int16_t* out, subsequence_info* s_info, const_state cstate)
{
    subsequence_info info;
    info.p = i * subsequence_size; // start of i-th subsequence
    info.n = 0;
    info.c = 0; // start from the first data unit of the Y component
    info.z = 0;

    reader_state rstate;
    rstate.data           = cstate.scan + (info.p / 8); // subsequence_size is multiple of eight
    rstate.data_end       = cstate.scan_end;
    rstate.cache          = 0;
    rstate.cache_num_bits = 0;

    int position_in_output = 0;
    if constexpr (do_write) {
        position_in_output = s_info[i].n;
    }
    if constexpr (is_overflow) {
        // FIXME is this proper? if not doing this, an uninitialized read will occur due to not
        //   storing n in sync_intra
        info.p = s_info[i - 1].p;
        info.c = s_info[i - 1].c;
        info.z = s_info[i - 1].z;

        rstate.data        = cstate.scan + (info.p / 8);
        const int in_cache = (8 - (info.p % 8)) % 8; // bits still in cache
        if (in_cache > 0) {
            rstate.cache          = *(rstate.data++);
            rstate.cache_num_bits = 8;
            discard_bits(rstate, 8 - in_cache);
        }
    }

    subsequence_info last_symbol; // the last detected codeword
    const int scan_bit_size = (cstate.scan_end - cstate.scan) * 8;
    while (info.p < min((i + 1) * subsequence_size, scan_bit_size)) {
        const component comp =
            calc_component(cstate.ssx, cstate.ssy, info.c, cstate.num_components);
        int length     = 0;
        int symbol     = 0;
        int run_length = 0;
        // always returns length > 0 to ensure while loop does not get stuck
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
                symbol == symbol_eob ? 0 : symbol;
        }
        if (do_write) {
            position_in_output += run_length + 1;
        }
        info.p += length;
        info.n += run_length + 1;
        info.z += run_length + 1;

        // TODO is EOB check needed?
        if (info.z >= 64 || symbol == symbol_eob) {
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

        last_symbol = info;

        // check if we have all blocks. this is needed since the scan is padded to a 8-bit multiple
        //   this problem is excerbated by restart intevals, where padding occurs more frequently
        if (position_in_output >= cstate.num_data_units * 64) {
            break;
        }
    }

    return last_symbol;
}

/// \brief Each thread handles one subsequence.
///   alg-3:05-23
///
/// \tparam block_size "b", the number of adjacent subsequences that form a sequence.
template <int block_size>
__global__ void sync_intra_sequence(
    subsequence_info* s_info, int num_subsequences, const_state cstate)
{
    assert(block_size == blockDim.x);
    const int bi = blockIdx.x;
    const int si = threadIdx.x;

    const int seq_global = bi * block_size;
    int subseq_global    = seq_global + si;

    if (subseq_global >= num_subsequences) {
        return;
    }

    bool synchronized = false;
    // paper uses `+ block_size` but `end` should be an index
    const int end     = min(seq_global + block_size - 1, num_subsequences - 1);
    // alg-3:10
    {
        subsequence_info info =
            decode_subsequence<false, false>(subseq_global, nullptr, s_info, cstate);
        s_info[subseq_global].p = info.p;
        // paper text does not mention `n` should be stored here, but if not storing `n`
        //   the first (of block) subsequence info's `n` will not be initialized
        s_info[subseq_global].n = info.n;
        s_info[subseq_global].c = info.c;
        s_info[subseq_global].z = info.z;
    }
    __syncthreads(); // wait until data of next subsequence is available
    ++subseq_global;
    while (!synchronized && subseq_global <= end) {
        subsequence_info info =
            decode_subsequence<true, false>(subseq_global, nullptr, s_info, cstate);
        if (info.p == s_info[subseq_global].p && info.c == s_info[subseq_global].c &&
            info.z == s_info[subseq_global].z) {
            // the decoding process of this thread has found the same "outcome" for the
            //   `subseq_global`th subsequence as the thread before it
            synchronized = true;
        }
        // FIXME inserted a sync, s_info[subseq_global] may be read in another thread's
        //   decode_subsequence
        __syncthreads();
        s_info[subseq_global] = info;
        ++subseq_global;
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
    int num_sequences)
{
    assert(blockIdx.x == 0); // required for syncing to work
    const int bi = threadIdx.x;
    if (bi >= num_sequences - 1) {
        return;
    }

    // last subsequence of sequence i
    int subseq_global = (bi + 1) * block_size;
    bool synchronized = false;
    // paper uses `+ block_size` but `end` should be an index
    const int end     = min(subseq_global + block_size - 1, num_subsequences - 1);
    while (!synchronized && subseq_global <= end) {
        subsequence_info info =
            decode_subsequence<true, false>(subseq_global, nullptr, s_info, cstate);
        if (info.p == s_info[subseq_global].p && info.c == s_info[subseq_global].c &&
            info.z == s_info[subseq_global].z) {
            // this means a synchronization point was found
            synchronized            = true;
            // TODO paper says bi - 1 but this will be 0 for the first thread?
            sequence_not_synced[bi] = false;
        }
        __syncthreads();
        s_info[subseq_global] = info; // FIXME paper gives no index
        ++subseq_global;
        __syncthreads();
    }
}

__global__ void decode_write(
    int16_t* out, subsequence_info* s_info, int num_subsequences, const_state cstate)
{
    const int si = blockIdx.x * blockDim.x + threadIdx.x;
    if (si >= num_subsequences) {
        return;
    }

    // only first thread does not do overflow
    constexpr bool do_write = true;
    if (si == 0) {
        decode_subsequence<false, do_write>(si, out, s_info, cstate);
    } else {
        decode_subsequence<true, do_write>(si, out, s_info, cstate);
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

bool jpeggpu::is_gpu_decode_possible(const jpeggpu::reader& reader)
{
    // not supported if non-luminance planes do not have the same subsampling
    //   this makes figuring the component out easier (c in s_info)
    for (int c = 1; c < reader.num_components; ++c) {
        if (reader.css.x[c] != 1 || reader.css.y[c] != 1) {
            return false;
        }
    }

    // ass-2
    if (reader.huff_map[0][jpeggpu::HUFF_DC] != 0 || reader.huff_map[0][jpeggpu::HUFF_AC] != 0) {
        return false;
    }
    for (int c = 1; c < reader.num_components; ++c) {
        if (reader.huff_map[c][jpeggpu::HUFF_DC] != 1 ||
            reader.huff_map[c][jpeggpu::HUFF_AC] != 1) {
            return false;
        }
    }

    return true;
}

size_t jpeggpu::calculate_gpu_decode_memory(const jpeggpu::reader& reader)
{
    size_t required = 0;
    // d_scan (can be less due to destuffing)
    required += gpu_alloc_size(reader.scan_size);
    // d_huff
    required += gpu_alloc_size(4 * sizeof(jpeggpu::huffman_table));
    // d_out
    size_t total_data_size = 0;
    for (int c = 0; c < reader.num_components; ++c) {
        total_data_size += reader.data_sizes_x[c] * reader.data_sizes_y[c];
    }
    required += gpu_alloc_size(total_data_size * sizeof(int16_t));
    // d_s_info
    const size_t scan_bit_size = reader.scan_size * 8;
    const int num_subsequences =
        ceiling_div(scan_bit_size, static_cast<unsigned int>(subsequence_size));
    required += gpu_alloc_size(num_subsequences * sizeof(subsequence_info));
    // d_sequence_not_synced
    constexpr int block_size = 256;
    const int num_sequences  = ceiling_div(num_subsequences, static_cast<unsigned int>(block_size));
    required += gpu_alloc_size((num_sequences - 1) * sizeof(uint8_t));
    // d_num_unsynced_sequence
    required += gpu_alloc_size(sizeof(int));
    // d_tmp_storage (reduction)
    size_t tmp_storage_bytes_reduction = 0;
    CHECK_CUDA(cub::DeviceReduce::Sum(
        nullptr,
        tmp_storage_bytes_reduction,
        reinterpret_cast<uint8_t*>(0),
        reinterpret_cast<int*>(0),
        num_sequences - 1,
        cudaStreamDefault));
    required += gpu_alloc_size(tmp_storage_bytes_reduction);
    // d_reduce_out
    required += gpu_alloc_size(num_subsequences * sizeof(subsequence_info));
    // d_tmp_storage (scan)
    size_t tmp_storage_bytes_scan = 0;
    CHECK_CUDA(cub::DeviceScan::ExclusiveScan(
        nullptr,
        tmp_storage_bytes_scan,
        reinterpret_cast<subsequence_info*>(0),
        reinterpret_cast<subsequence_info*>(0),
        sum_subsequence_info{},
        subsequence_info{},
        num_subsequences,
        cudaStreamDefault));

    return required;
}

#define CHECK_STAT(call)                                                                           \
    do {                                                                                           \
        jpeggpu_status stat = call;                                                                \
        if (stat != JPEGGPU_SUCCESS) {                                                             \
            return JPEGGPU_INTERNAL_ERROR;                                                         \
        }                                                                                          \
    } while (0)

jpeggpu_status jpeggpu::process_scan(
    jpeggpu::reader& reader,
    int16_t* (&d_image_qdct)[jpeggpu::max_comp_count],
    void*& d_tmp,
    size_t& tmp_size,
    cudaStream_t stream)
{
    // destuff TODO GPU
    uint8_t* d_scan;
    int scan_size = 0;
    {
        std::vector<uint8_t> destuffed;
        destuffed.reserve(reader.scan_size);

        for (size_t i = 0; i < reader.scan_size; ++i) {
            const uint8_t byte = reader.scan_start[i];
            if (byte == 0xff) {
                assert(i + 1 < reader.scan_size);
                ++i;
                // skip next byte, check its value
                const uint8_t marker = reader.scan_start[i];
                // should be a stuffed byte or a restart marker
                assert(
                    marker == 0 ||
                    (jpeggpu::MARKER_RST0 <= marker && marker <= jpeggpu::MARKER_RST7));
                (void)marker; // surpress warning
                // stuffed byte or marker is subsequently ignored
            }
            destuffed.push_back(byte);
        }

        // this assumption allows to represent the bit offset with an int
        assert(destuffed.size() * 8 <= INT_MAX);

        scan_size = destuffed.size();

        CHECK_STAT(jpeggpu::gpu_alloc_reserve(
            reinterpret_cast<void**>(&d_scan), scan_size, d_tmp, tmp_size));
        CHECK_CUDA(cudaMemcpy(d_scan, destuffed.data(), scan_size, cudaMemcpyHostToDevice));
    }

    jpeggpu::huffman_table* d_huff;
    CHECK_STAT(jpeggpu::gpu_alloc_reserve(
        reinterpret_cast<void**>(&d_huff), 4 * sizeof(jpeggpu::huffman_table), d_tmp, tmp_size));
    CHECK_CUDA(cudaMemcpyAsync(
        d_huff + 0,
        &(reader.huff_tables[0][jpeggpu::HUFF_DC]),
        sizeof(jpeggpu::huffman_table),
        cudaMemcpyHostToDevice,
        stream));
    CHECK_CUDA(cudaMemcpyAsync(
        d_huff + 1,
        &(reader.huff_tables[0][jpeggpu::HUFF_AC]),
        sizeof(jpeggpu::huffman_table),
        cudaMemcpyHostToDevice,
        stream));
    CHECK_CUDA(cudaMemcpyAsync(
        d_huff + 2,
        &(reader.huff_tables[1][jpeggpu::HUFF_DC]),
        sizeof(jpeggpu::huffman_table),
        cudaMemcpyHostToDevice,
        stream));
    CHECK_CUDA(cudaMemcpyAsync(
        d_huff + 3,
        &(reader.huff_tables[1][jpeggpu::HUFF_AC]),
        sizeof(jpeggpu::huffman_table),
        cudaMemcpyHostToDevice,
        stream));

    const size_t scan_bit_size = scan_size * 8;
    const int num_subsequences =
        ceiling_div(scan_bit_size, static_cast<unsigned int>(subsequence_size)); // "N"
    constexpr int block_size = 256; // "b", size in subsequences
    const int num_sequences =
        ceiling_div(num_subsequences, static_cast<unsigned int>(block_size)); // "B"

    // alg-1:01
    size_t total_data_size = 0;
    int num_data_units     = 0;
    for (int c = 0; c < reader.num_components; ++c) {
        total_data_size += reader.data_sizes_x[c] * reader.data_sizes_y[c];
        num_data_units += (reader.data_sizes_x[c] / 8) * (reader.data_sizes_y[c] / 8);
    }
    int16_t* d_out;
    CHECK_STAT(jpeggpu::gpu_alloc_reserve(
        reinterpret_cast<void**>(&d_out), total_data_size * sizeof(int16_t), d_tmp, tmp_size));
    // initialize to zero, since only non-zeros are written
    CHECK_CUDA(cudaMemsetAsync(d_out, 0, total_data_size * sizeof(int16_t), stream));

    // alg-1:05
    subsequence_info* d_s_info;
    CHECK_STAT(jpeggpu::gpu_alloc_reserve(
        reinterpret_cast<void**>(&d_s_info),
        num_subsequences * sizeof(subsequence_info),
        d_tmp,
        tmp_size));

    const const_state cstate = {
        d_scan,
        d_scan + scan_size,
        d_huff + 0,
        d_huff + 1,
        d_huff + 2,
        d_huff + 3,
        reader.css.x[0],
        reader.css.y[0],
        reader.num_components,
        num_data_units};

    { // sync_decoders (Algorithm 3)

        sync_intra_sequence<block_size>
            <<<num_sequences, block_size, 0, stream>>>(d_s_info, num_subsequences, cstate);
        CHECK_CUDA(cudaGetLastError());

        if (num_sequences > 1) {
            // note: the meaning of this array is flipped, a one is produced if not synced
            uint8_t* d_sequence_not_synced;
            CHECK_STAT(jpeggpu::gpu_alloc_reserve(
                reinterpret_cast<void**>(&d_sequence_not_synced),
                (num_sequences - 1) * sizeof(uint8_t),
                d_tmp,
                tmp_size));
            CHECK_CUDA(cudaMemsetAsync( // all are initialized to "not synced"
                d_sequence_not_synced,
                static_cast<uint8_t>(true),
                (num_sequences - 1) * sizeof(uint8_t),
                stream));

            int* d_num_unsynced_sequence;
            CHECK_STAT(jpeggpu::gpu_alloc_reserve(
                reinterpret_cast<void**>(&d_num_unsynced_sequence), sizeof(int), d_tmp, tmp_size));

            void* d_tmp_storage      = nullptr;
            size_t tmp_storage_bytes = 0;
            CHECK_CUDA(cub::DeviceReduce::Sum(
                d_tmp_storage,
                tmp_storage_bytes,
                d_sequence_not_synced,
                d_num_unsynced_sequence,
                num_sequences - 1,
                stream));

            CHECK_STAT(jpeggpu::gpu_alloc_reserve(
                reinterpret_cast<void**>(&d_tmp_storage), tmp_storage_bytes, d_tmp, tmp_size));

            int h_num_unsynced_sequence;
            do {
                // TODO this means the subsequence size must be dynamic.
                //   for the syncing in the kernel to work, only one block can be launched
                const int block_size_inter = num_sequences - 1;
                sync_inter_sequence<block_size><<<1, block_size_inter, 0, stream>>>(
                    d_s_info, num_subsequences, cstate, d_sequence_not_synced, num_sequences);
                CHECK_CUDA(cudaDeviceSynchronize()); // FIXME remove
                std::cout << "inter sequence sync done\n";

                CHECK_CUDA(cub::DeviceReduce::Sum(
                    d_tmp_storage,
                    tmp_storage_bytes,
                    d_sequence_not_synced,
                    d_num_unsynced_sequence,
                    num_sequences - 1,
                    stream));
                CHECK_CUDA(cudaMemcpy(
                    &h_num_unsynced_sequence,
                    d_num_unsynced_sequence,
                    sizeof(int),
                    cudaMemcpyDeviceToHost));

                std::cout << "unsynced: " << h_num_unsynced_sequence << "\n";
            } while (h_num_unsynced_sequence);
        }
    }

    // FIXME debug
    std::vector<subsequence_info> h_s_info(num_subsequences);
    CHECK_CUDA(cudaMemcpy(
        h_s_info.data(),
        d_s_info,
        num_subsequences * sizeof(subsequence_info),
        cudaMemcpyDeviceToHost));

    // TODO consider SoA or do in-place
    // alg-1:07-08
    {
        subsequence_info* d_reduce_out;
        CHECK_STAT(jpeggpu::gpu_alloc_reserve(
            reinterpret_cast<void**>(&d_reduce_out),
            num_subsequences * sizeof(subsequence_info),
            d_tmp,
            tmp_size));
        // FIXME debug to satisfy initcheck
        CHECK_CUDA(
            cudaMemsetAsync(d_reduce_out, 0, num_subsequences * sizeof(subsequence_info), stream));

        const subsequence_info init_value{0, 0, 0, 0};
        void* d_tmp_storage      = nullptr;
        size_t tmp_storage_bytes = 0;
        CHECK_CUDA(cub::DeviceScan::ExclusiveScan(
            d_tmp_storage,
            tmp_storage_bytes,
            d_s_info,
            d_reduce_out,
            sum_subsequence_info{},
            init_value,
            num_subsequences,
            stream));

        CHECK_STAT(jpeggpu::gpu_alloc_reserve(
            reinterpret_cast<void**>(&d_tmp_storage), tmp_storage_bytes, d_tmp, tmp_size));
        // FIXME debug to satisfy initcheck
        CHECK_CUDA(cudaMemsetAsync(d_tmp_storage, 0, tmp_storage_bytes, stream));

        CHECK_CUDA(cub::DeviceScan::ExclusiveScan(
            d_tmp_storage,
            tmp_storage_bytes,
            d_s_info,
            d_reduce_out,
            sum_subsequence_info{},
            init_value,
            num_subsequences,
            stream));

        constexpr int block_size_assign = 256;
        const int grid_dim =
            ceiling_div(num_subsequences, static_cast<unsigned int>(block_size_assign));
        assign_sinfo_n<<<grid_dim, block_size_assign, 0, stream>>>(
            num_subsequences, d_s_info, d_reduce_out);
        CHECK_CUDA(cudaGetLastError());
    }

    // FIXME debug
    CHECK_CUDA(cudaMemcpy(
        h_s_info.data(),
        d_s_info,
        num_subsequences * sizeof(subsequence_info),
        cudaMemcpyDeviceToHost));

    // alg-1:09-15
    CHECK_CUDA(cudaStreamSynchronize(stream)); // FIXME debug
    decode_write<<<num_sequences, block_size, 0, stream>>>(
        d_out, d_s_info, num_subsequences, cstate);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream)); // FIXME debug

    // FIXME can the below (and above) deal with non-interleaved scans?

    std::vector<std::vector<int16_t>> data(reader.num_components);
    for (int c = 0; c < reader.num_components; ++c) {
        data[c].resize(reader.data_sizes_x[c] * reader.data_sizes_y[c]);
    }

    // TODO replace with GPU transpose
    std::vector<int16_t> h_out(total_data_size);
    CHECK_CUDA(cudaMemcpyAsync(
        h_out.data(), d_out, total_data_size * sizeof(int16_t), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // TODO fix for non-interleaved

    int data_unit_idx = 0;
    for (int y = 0; y < reader.num_mcus_y; ++y) {
        for (int x = 0; x < reader.num_mcus_x; ++x) {
            constexpr size_t data_unit_bytes = jpeggpu::data_unit_size * sizeof(int16_t);
            for (int c = 0; c < reader.num_components; ++c) {
                for (int ssy = 0; ssy < reader.css.y[c]; ++ssy) {
                    for (int ssx = 0; ssx < reader.css.x[c]; ++ssx) {
                        const int y_block = y * reader.css.y[c] + ssy;
                        const int x_block = x * reader.css.x[c] + ssx;
                        const size_t idx  = y_block * jpeggpu::block_size * reader.mcu_sizes_x[c] *
                                               reader.num_mcus_x +
                                           x_block * jpeggpu::block_size * jpeggpu::block_size;
                        std::memcpy(
                            data[c].data() + idx,
                            h_out.data() + (data_unit_idx++) * jpeggpu::data_unit_size,
                            data_unit_bytes);
                    }
                }
            }
        }
    }

    // FIXME for subsampled images, it may be needed to first rearrange the data unit order
    //   for the luminance plane

    // undo DC difference encoding
    // TODO deal with non-interleaved?
    int dc[jpeggpu::max_comp_count] = {};
    int mcu_count                   = 0;
    for (int y_mcu = 0; y_mcu < reader.num_mcus_y; ++y_mcu) {
        for (int x_mcu = 0; x_mcu < reader.num_mcus_x; ++x_mcu) {
            if (reader.seen_dri && mcu_count % reader.restart_interval == 0) {
                for (int c = 0; c < jpeggpu::max_comp_count; ++c) {
                    dc[c] = 0;
                }
            }

            // one MCU
            for (int c = 0; c < reader.num_components; ++c) {
                for (int y_ss = 0; y_ss < reader.css.y[c]; ++y_ss) {
                    for (int x_ss = 0; x_ss < reader.css.x[c]; ++x_ss) {
                        const int y_block = y_mcu * reader.css.y[c] + y_ss;
                        const int x_block = x_mcu * reader.css.x[c] + x_ss;
                        const size_t idx  = y_block * jpeggpu::block_size * reader.mcu_sizes_x[c] *
                                               reader.num_mcus_x +
                                           x_block * jpeggpu::block_size * jpeggpu::block_size;
                        int16_t* dst = &(data[c][idx]);
                        dst[0]       = dc[c] += dst[0];
                    }
                }
            }
            ++mcu_count;
        }
    }

    for (int c = 0; c < reader.num_components; ++c) {
        CHECK_CUDA(cudaMemcpy(
            d_image_qdct[c],
            data[c].data(),
            reader.data_sizes_x[c] * reader.data_sizes_y[c] * sizeof(int16_t),
            cudaMemcpyHostToDevice));
    }

    return JPEGGPU_SUCCESS;
}

#undef CHECK_STAT
