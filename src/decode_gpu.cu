#include "decode_gpu.hpp"
#include "marker.hpp"
#include "reader.hpp"
#include "util.hpp"

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <jpeggpu/jpeggpu.h>

#include <cassert>
#include <vector>

namespace {

/// Assumption 1 (ass-1): all non-luma is subsampled with the same factor
/// Assumption 2 (ass-2): Huffman table mapping

/// \brief "s", subsequence size in 32 bits. Paper uses 4 or 32 depending on the quality of the
///   encoded image.
constexpr int chunk_size       = 4;               ///< "s", in 32 bits
constexpr int subsequence_size = chunk_size * 32; ///< size in bits

/// \brief Contains all required information about the last syncrhonization point for the
///   subsequence.
struct subsequence_info {
    /// \brief Bit position in scan.
    ///   TODO size_t?
    int p;
    /// \brief The number of decoded symbols.
    int n;
    /// \brief The data unit index in the MCU (slightly deviates from paper). The color component
    ///   (meaning of `c` in the paper) can be inferred from this, together with subsampling
    ///   factors.
    int c;
    /// \brief Zig-zag index.
    int z;
};

struct reader_state {
    const uint8_t* data;
    const uint8_t* data_end;
    int32_t cache;
    int cache_num_bits;
};

__device__ void load_byte(reader_state& rstate)
{
    assert(rstate.data < rstate.data_end);
    assert(rstate.cache_num_bits + 8 < 32);

    // FIXME deal with marker and stuffing
    const uint8_t next_byte = *(rstate.data++);
    rstate.cache            = rstate.cache << 8 | next_byte;
    rstate.cache_num_bits += 8;
}

__device__ void load_bits(reader_state& rstate, int num_bits)
{
    do {
        if (rstate.data >= rstate.data_end) {
            return; // no more data to load
        }

        load_byte(rstate);
    } while (rstate.cache_num_bits < num_bits);
}

__device__ int select_bits(reader_state& rstate, int num_bits)
{
    assert(num_bits < 31);
    assert(rstate.cache_num_bits >= num_bits);

    // upper bits are zero
    return rstate.cache >> (rstate.cache_num_bits - num_bits);
}

__device__ void discard_bits(reader_state& rstate, int num_bits)
{
    assert(num_bits <= rstate.cache_num_bits);
    // set discarded bits to zero (upper bits in the cache)
    rstate.cache = rstate.cache & (1 << (rstate.cache_num_bits - num_bits) - 1);
    rstate.cache_num_bits -= num_bits;
}

__device__ int get_category(reader_state& rstate, int& length, const jpeggpu::huffman_table& table)
{
    load_bits(rstate, 16);

    int i, code;
    for (i = 0; i < 16; ++i) {
        code = select_bits(rstate, i + 1);
        if (code <= table.maxcode[i + 1] || i == 15) {
            break;
        }
    }
    discard_bits(rstate, i + 1);
    length        = i + 1;
    const int idx = table.valptr[i + 1] + (code - table.mincode[i + 1]);
    assert(0 < idx && idx < 256);
    return table.huffval[idx];
}

__device__ int get_value(int num_bits, int code)
{
    return code < ((1 << num_bits) >> 1) ? (code + ((-1) << num_bits) + 1) : code;
}

__device__ void decode_next_symbol_dc(
    reader_state& rstate,
    int& length,
    int& symbol,
    int& run_length,
    const jpeggpu::huffman_table& table)
{
    int category_length = 0;
    const int category  = get_category(rstate, category_length, table);

    if (category != 0) {
        assert(0 < category && category < 17);
        load_bits(rstate, category);
        const int offset = select_bits(rstate, category);
        const int value  = get_value(category, offset);

        length     = category_length + category;
        symbol     = value;
        run_length = 0;
    } else {
        length     = category_length;
        symbol     = 0;
        run_length = 0;
    }
};

/// \brief Zero run length, indicates a run of 16 zeros.
constexpr int symbol_zrl = INT32_MAX;
/// \brief End of block, indicates all remaining coefficients in the data unit are zero.
constexpr int symbol_eob = INT32_MAX - 1;

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
    const int s         = get_category(rstate, category_length, table);

    const int run      = s >> 4;
    const int category = s & 0xf;

    if (category != 0) {
        assert(0 < category && category < 17);
        load_bits(rstate, category);
        const int offset = select_bits(rstate, category);
        const int value  = get_value(category, offset);

        length     = category_length + category;
        symbol     = value;
        run_length = run;
    } else {
        if (run == 15) {
            length     = category_length;
            symbol     = symbol_zrl;
            run_length = run;
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
/// \param[out] length The number of processed bits.
/// \param[out] symbol The decoded coefficient, provided the code was not EOB or ZRL.
/// \param[out] run_length The run-length of zeroes which the coefficient is followed by.
/// \param[in] table
/// \param[in] z Current index in the zig-zag sequence.
/// \param[in] is_dc
__device__ void decode_next_symbol(
    reader_state& rstate,
    int& length,
    int& symbol,
    int& run_length,
    const jpeggpu::huffman_table& table,
    int z,
    bool is_dc)
{
    if (is_dc) {
        decode_next_symbol_dc(rstate, length, symbol, run_length, table);
    } else {
        decode_next_symbol_ac(rstate, length, symbol, run_length, table, z);
    }
}

enum class component {
    y,  // Y (YCbCR) or C (CMYK)
    cb, // Cb (YCbCR) or M (CMYK)
    cr, // Cr (YCbCR) or Y (CMYK)
    k   // k (CMYK)
};

/// \brief Infer image components based on data unit index `c` (in MCU).
__device__ component calc_component(int ssx, int ssy, int c, int num_components)
{
    const int num_luma_data_units = ssx * ssy;

    if (c < num_luma_data_units) {
        return component::y;
    }

    assert(num_components >= 2);

    if (c == num_luma_data_units) {
        return component::cb;
    }

    assert(num_components >= 3);

    if (c == num_luma_data_units + 1) {
        return component::cr;
    }

    assert(num_components >= 4);

    if (c == num_luma_data_units + 2) {
        return component::k;
    }

    assert(false);
}

__device__ bool same_component(int ssx, int ssy, int c0, int c1, int num_components)
{
    return calc_component(ssx, ssy, c0, num_components) ==
           calc_component(ssx, ssy, c1, num_components);
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
};

/// \brief Algorithm 2.
///
/// \tparam is_overflow Whether `i` was decoded by another thread already. TODO word this better.
/// \tparam do_write Whether to write the coefficients to the output buffer.
template <bool is_overflow, bool do_write>
__device__ subsequence_info decode_subsequence(
    int i,
    int16_t* out_0,
    int16_t* out_1,
    int16_t* out_2,
    int16_t* out_3,
    subsequence_info* s_info,
    const_state cstate)
{
    subsequence_info info;
    info.p = i * subsequence_size; // start of i-th subsequence
    info.n = 0;
    info.c = 0; // start from the first data unit of the Y component
    info.z = 0;

    reader_state rstate;
    rstate.data           = cstate.scan + info.p;
    rstate.data_end       = cstate.scan_end;
    rstate.cache          = 0;
    rstate.cache_num_bits = 0;

    bool is_dc = true;

    int position_in_output = 0;
    if constexpr (do_write) {
        position_in_output = s_info[i].n;
    }
    if constexpr (is_overflow) {
        info = s_info[i - 1];
    }
    bool end_of_subsequence_reached = false;
    subsequence_info last_symbol; // the last detected codeword
    while (!end_of_subsequence_reached) {
        last_symbol = info;
        const component comp =
            calc_component(cstate.ssx, cstate.ssy, info.c, cstate.num_components);
        const jpeggpu::huffman_table& table =
            comp == component::y ? (is_dc ? *cstate.table_luma_dc : *cstate.table_luma_ac)
                                 : (is_dc ? *cstate.table_chroma_dc : *cstate.table_chroma_ac);
        int length;
        int symbol;
        int run_length;
        decode_next_symbol(rstate, length, symbol, run_length, table, info.z, is_dc);
        is_dc = false; // only one dc symbol per data unit
        if (do_write) { // TODO could make a separate kernel for this
            switch (comp) {
            case component::y:
                out_0[position_in_output] = symbol;
                break;
            case component::cb:
                out_1[position_in_output] = symbol;
                break;
            case component::cr:
                out_2[position_in_output] = symbol;
                break;
            case component::k:
                out_3[position_in_output] = symbol;
                break;
            }
        }
        position_in_output = position_in_output + run_length + 1;
        info.p += length;
        info.n += run_length + 1;
        info.z += run_length + 1;
        if (info.z >= 64 || symbol == symbol_eob) {
            // the data unit is complete
            info.z = 0;
            info.c++;
            // ass-1
            const int num_data_units_in_mcu = cstate.ssx * cstate.ssy + (cstate.num_components - 1);
            if (info.c >= num_data_units_in_mcu) {
                info.c = 0;
            }
            const component comp_next =
                calc_component(cstate.ssx, cstate.ssy, info.c, cstate.num_components);
            if (comp != comp_next) {
                is_dc = true;
            }
        }
    }
    return last_symbol;
}

/// \brief Each thread handles one subsequence.
template <int block_size>
__global__ void sync_intra_sequence(
    int16_t* out_0,
    int16_t* out_1,
    int16_t* out_2,
    int16_t* out_3,
    subsequence_info* s_info,
    int num_subsequences,
    const_state cstate)
{
    assert(block_size == blockDim.x);
    const int bi = blockIdx.x;

    const int seq_global = bi * block_size;
    int subseq_global    = seq_global + threadIdx.x;
    bool synchronized    = false;
    const int end        = min(seq_global + block_size, num_subsequences - 1);
    // TODO should this line store n as well?
    s_info[subseq_global] =
        decode_subsequence<false, false>(subseq_global, out_0, out_1, out_2, out_3, s_info, cstate);
    __syncthreads();
    ++subseq_global;
    while (!synchronized && subseq_global <= end) {
        subsequence_info info = decode_subsequence<true, false>(
            subseq_global, out_0, out_1, out_2, out_3, s_info, cstate);
        if (info.p == s_info[subseq_global].p &&
            same_component(
                cstate.ssx, cstate.ssy, info.c, s_info[subseq_global].c, cstate.num_components) &&
            info.z == s_info[subseq_global].z) {
            // this means a synchronization point was found
            synchronized = true;
        }
        s_info[subseq_global] = info; // FIXME paper gives no index
        ++subseq_global;
        __syncthreads();
    }
}

/// \brief Each thread handles one sequence, the last sequence is not handled.
template <int block_size>
__global__ void sync_inter_sequence(
    int16_t* out_0,
    int16_t* out_1,
    int16_t* out_2,
    int16_t* out_3,
    subsequence_info* s_info,
    int num_subsequences,
    const_state cstate,
    int* sequence_synced)
{
    const int bi = blockDim.x * blockIdx.x + threadIdx.x;

    int subseq_global = (bi + 1) * block_size;
    bool synchronized = false;
    const int end     = min(subseq_global + block_size, num_subsequences - 1);
    while (!synchronized && subseq_global <= end) {
        subsequence_info info = decode_subsequence<true, false>(
            subseq_global, out_0, out_1, out_2, out_3, s_info, cstate);
        if (info.p == s_info[subseq_global].p &&
            same_component(
                cstate.ssx, cstate.ssy, info.c, s_info[subseq_global].c, cstate.num_components) &&
            info.z == s_info[subseq_global].z) {
            // this means a synchronization point was found
            synchronized            = true;
            sequence_synced[bi - 1] = true;
        }
        s_info[subseq_global] = info; // FIXME paper gives no index
        ++subseq_global;
        __syncthreads();
    }
}

__global__ void decode_write(
    int16_t* out_0,
    int16_t* out_1,
    int16_t* out_2,
    int16_t* out_3,
    subsequence_info* s_info,
    int num_subsequences,
    const_state cstate)
{
    const int si = blockIdx.x * blockDim.x + threadIdx.x;
    if (si < num_subsequences) {
        return;
    }

    // only first thread does not do overflow
    constexpr bool do_write = true;
    if (si == 0) {
        decode_subsequence<false, do_write>(si, out_0, out_1, out_2, out_3, s_info, cstate);
    } else {
        decode_subsequence<true, do_write>(si, out_0, out_1, out_2, out_3, s_info, cstate);
    }
}

struct sum_subsequence_info {
    __device__ __forceinline__ subsequence_info
    operator()(const subsequence_info& a, const subsequence_info& b) const
    {
        return {0, a.n + b.n, 0, 0};
    }
};

__global__ void assign(int num_subsequences, subsequence_info* dst, const subsequence_info* src)
{
    const int lid = blockDim.x * blockIdx.x + threadIdx.x;
    if (lid >= num_subsequences) {
        return;
    }

    dst[lid].n = src[lid].n;
}

} // namespace

jpeggpu_status process_scan(jpeggpu::reader& reader, cudaStream_t stream)
{
    const int ssx = reader.ss_x[0];
    const int ssy = reader.ss_y[0];
    // not supported if non-luminance planes do not have the same subsampling
    //   this makes figuring the component out easier (c in s_info)
    for (int c = 1; c < reader.num_components; ++c) {
        if (reader.ss_x[c] != 1 || reader.ss_y[c] != 1) {
            return JPEGGPU_NOT_SUPPORTED;
        }
    }

    // ass-2
    if (reader.huff_map[0][jpeggpu::HUFF_DC] != 0 || reader.huff_map[0][jpeggpu::HUFF_AC] != 0) {
        return JPEGGPU_NOT_SUPPORTED;
    }
    for (int c = 1; c < reader.num_components; ++c) {
        if (reader.huff_map[c][jpeggpu::HUFF_DC] != 1 ||
            reader.huff_map[c][jpeggpu::HUFF_AC] != 1) {
            return JPEGGPU_NOT_SUPPORTED;
        }
    }

    jpeggpu::huffman_table* d_huff;
    CHECK_CUDA(cudaMalloc(&d_huff, 4 * sizeof(jpeggpu::huffman_table)));
    CHECK_CUDA(cudaMemcpyAsync(
        &(d_huff[0]),
        &(reader.huff_tables[0][jpeggpu::HUFF_DC]),
        sizeof(jpeggpu::huffman_table),
        cudaMemcpyHostToDevice,
        stream));
    CHECK_CUDA(cudaMemcpyAsync(
        &(d_huff[1]),
        &(reader.huff_tables[0][jpeggpu::HUFF_AC]),
        sizeof(jpeggpu::huffman_table),
        cudaMemcpyHostToDevice,
        stream));
    CHECK_CUDA(cudaMemcpyAsync(
        &(d_huff[2]),
        &(reader.huff_tables[1][jpeggpu::HUFF_DC]),
        sizeof(jpeggpu::huffman_table),
        cudaMemcpyHostToDevice,
        stream));
    CHECK_CUDA(cudaMemcpyAsync(
        &(d_huff[3]),
        &(reader.huff_tables[1][jpeggpu::HUFF_AC]),
        sizeof(jpeggpu::huffman_table),
        cudaMemcpyHostToDevice,
        stream));

    assert(reader.scan_size <= INT_MAX);
    const size_t scan_bit_size = reader.scan_size * 8;
    const int num_subsequences =
        ceiling_div(scan_bit_size, static_cast<unsigned int>(subsequence_size)); // "N"
    constexpr int block_size = 256; // "b", size in subsequences
    const int num_sequences =
        ceiling_div(num_subsequences, static_cast<unsigned int>(block_size)); // "B"

    uint8_t* d_scan;
    CHECK_CUDA(cudaMalloc(&d_scan, reader.scan_size));
    CHECK_CUDA(cudaMemcpyAsync(
        d_scan, reader.scan_start, reader.scan_size, cudaMemcpyHostToDevice, stream));

    int16_t* d_out[jpeggpu::max_comp_count];
    for (int c = 0; c < reader.num_components; ++c) {
        CHECK_CUDA(cudaMalloc(
            &d_out[c], reader.data_sizes_x[c] * reader.data_sizes_y[c] * sizeof(int16_t)));
    }

    subsequence_info* d_s_info;
    CHECK_CUDA(cudaMalloc(&d_s_info, num_subsequences * sizeof(subsequence_info)));

    const const_state cstate = {
        d_scan,
        d_scan + reader.scan_size,
        d_huff + 0,
        d_huff + 1,
        d_huff + 2,
        d_huff + 3,
        ssx,
        ssy,
        reader.num_components};

    { // sync_decoders (Algorithm 3)
        sync_intra_sequence<block_size><<<num_sequences, block_size, 0, stream>>>(
            d_out[0], d_out[1], d_out[2], d_out[3], d_s_info, num_subsequences, cstate);
        CHECK_CUDA(cudaGetLastError());

        int* d_sequence_synced;
        CHECK_CUDA(cudaMalloc(&d_sequence_synced, num_sequences * sizeof(int)));

        bool exists_unsynced_sequence = true;
        do {
            constexpr int block_size_inter = 256;
            const int grid_dim =
                ceiling_div(num_sequences - 1, static_cast<unsigned int>(block_size_inter));
            sync_inter_sequence<block_size><<<grid_dim, block_size_inter, 0, stream>>>(
                d_out[0],
                d_out[1],
                d_out[2],
                d_out[3],
                d_s_info,
                num_subsequences,
                cstate,
                d_sequence_synced);
            exists_unsynced_sequence; // FIXME = reduce(d_sequence_synced);
        } while (exists_unsynced_sequence);
        CHECK_CUDA(cudaFree(d_sequence_synced));
    }

    // TODO consider SoA
    // 1.7 exclusive prefix sum on s_info.n
    // 1.8 s_info[0].n = 0
    {
        subsequence_info* d_reduce_out;
        CHECK_CUDA(cudaMalloc(&d_reduce_out, num_subsequences * sizeof(subsequence_info)));

        const subsequence_info init_value{0, 0, 0, 0};
        void* d_temp_storage      = nullptr;
        size_t temp_storage_bytes = 0;
        CHECK_CUDA(cub::DeviceScan::ExclusiveScan(
            d_temp_storage,
            temp_storage_bytes,
            d_s_info,
            d_reduce_out,
            sum_subsequence_info{},
            init_value,
            num_subsequences,
            stream));

        CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

        CHECK_CUDA(cub::DeviceScan::ExclusiveScan(
            d_temp_storage,
            temp_storage_bytes,
            d_s_info,
            d_reduce_out,
            sum_subsequence_info{},
            init_value,
            num_subsequences,
            stream));

        CHECK_CUDA(cudaFree(d_temp_storage));
        constexpr int block_size_assign = 128;
        const int grid_dim =
            ceiling_div(num_subsequences, static_cast<unsigned int>(block_size_assign));
        assign<<<grid_dim, block_size_assign, 0, stream>>>(
            num_subsequences, d_s_info, d_reduce_out);
    }

    decode_write<<<num_sequences, block_size, 0, stream>>>(
        d_out[0], d_out[1], d_out[2], d_out[3], d_s_info, num_subsequences, cstate);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaFree(d_s_info));

    // TODO reverse DC difference coding

    for (int c = 0; c < reader.num_components; ++c) {
        CHECK_CUDA(cudaMemcpyAsync(
            reader.data[c],
            d_out[c],
            reader.data_sizes_x[c] * reader.data_sizes_y[c] * sizeof(int16_t),
            cudaMemcpyDeviceToHost,
            stream));
    }
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_scan));

    CHECK_CUDA(cudaFree(d_huff));

    return JPEGGPU_SUCCESS;
}
