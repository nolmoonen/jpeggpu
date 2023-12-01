#include "decode_gpu.hpp"
#include "marker.hpp"
#include "reader.hpp"
#include "util.hpp"

#include <cuda_runtime.h>
#include <jpeggpu/jpeggpu.h>

#include <cassert>
#include <vector>

namespace {
/// \brief Contains all required information about the last syncrhonization point for the
///   subsequence.
struct subsequence_info {
    /// \brief Input pointer
    const uint8_t* p;
    /// \brief The number of decoded symbols.
    int n;
    /// \brief The data unit index in the MCU (slightly deviates from paper). The color component
    ///   (meaning of `c` in the paper) can be inferred from this, together with subsampling
    ///   factors.
    int c;
    /// \brief Zig-zag index.
    int z;
};

/// \brief Extracts coefficients from the bitstream while switching between DC and AC Huffman
/// tables.
///
/// \param[out] length The number of processed bits.
/// \param[out] symbol The decoded coefficient, provided the code was not EOB or ZRL.
/// \param[out] run_length The run- length of zeroes which the coefficient is followed by.
__device__ void decode_next_symbol(){
    // FIXME if symbol equals ZRL, 15 will be returned for run_length
    // FIXME if symbol equals EOB, 63-z will be returned for run_length, with z begin the current
    // index
    //   in the zig-zag sequence
};

__device__ bool is_eob(uint8_t sumbol)
{
    assert(false); // FIXME
    return false;
}

/// \brief Algorithm 2.
///
/// \tparam is_overflow Whether `i` was decoded by another thread already. TODO word this better.
/// \tparam do_write Whether to write the coefficients to the output buffer.
template <bool is_overflow, bool do_write>
__device__ subsequence_info decode_subsequence(int i, int16_t* out, subsequence_info* s_info)
{
    subsequence_info info;
    info.p; // FIXME = start of ith subsequence
    info.n = 0;
    info.c = 0; // start from the first data unit of the Y component
    info.z = 0;

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
        int length;
        uint8_t symbol;
        int run_length;
        // FIXME from which symbol does this start? scan_start + i * subsequence_size? 
        // decode_next_symbol(length, symbol, run_length);
        if (do_write) {
            out[position_in_output] = symbol;
        }
        position_in_output = position_in_output + run_length + 1;
        info.p += length;
        info.n += run_length + 1;
        info.z += run_length + 1;
        if (info.z >= 64 || is_eob(symbol)) {
            // the data unit is complete
            info.z = 0;
        }
        info.c++; // onto the next data unit
    }
    return last_symbol;
}

/// \brief Each thread handles one subsequence.
template <int block_size>
__global__ void sync_intra_sequence(int16_t* out, subsequence_info* s_info, int num_subsequences)
{
    assert(block_size == blockDim.x);
    const int bi = blockIdx.x;

    const int seq_global  = bi * block_size;
    int subseq_global     = seq_global + threadIdx.x;
    bool synchronized     = false;
    const int end         = min(seq_global + block_size, num_subsequences - 1);
    // TODO should this line store n as well?
    s_info[subseq_global] = decode_subsequence<false, false>(subseq_global, out, s_info);
    __syncthreads();
    ++subseq_global;
    while (!synchronized && subseq_global <= end) {
        subsequence_info info = decode_subsequence<true, false>(subseq_global, out, s_info);
        // TODO comparison with c is probably to ensure same Huffman table was used
        if (info.p == s_info[subseq_global].p && info.c == s_info[subseq_global].c &&
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
    int16_t* out, subsequence_info* s_info, int num_subsequences, int* sequence_synced)
{
    const int bi = blockDim.x * blockIdx.x + threadIdx.x;

    int subseq_global = (bi + 1) * block_size;
    bool synchronized = false;
    const int end     = min(subseq_global + block_size, num_subsequences - 1);
    while (!synchronized && subseq_global <= end) {
        subsequence_info info = decode_subsequence<true, false>(subseq_global, out, s_info);
        // TODO comparison with c is probably to ensure same Huffman table was used
        if (info.p == s_info[subseq_global].p && info.c == s_info[subseq_global].c &&
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

__global__ void decode_write(int16_t* out, subsequence_info* s_info, int num_subsequences)
{
    const int si = blockIdx.x * blockDim.x + threadIdx.x;
    if (si < num_subsequences) {
        return;
    }

    // only first thread does not do overflow
    constexpr bool do_write = true;
    if (si == 0) {
        decode_subsequence<false, do_write>(si, out, s_info);
    } else {
        decode_subsequence<true, do_write>(si, out, s_info);
    }
}

/// \brief Algorithm 1.
jpeggpu_status decode_gpu(const uint8_t* scan_start, size_t scan_size)
{
    assert(scan_size <= INT_MAX);
    constexpr int chunk_size       = 8;              // "s", size in 32 bits
    constexpr int subsequence_size = chunk_size * 4; // size in bytes
    const int num_subsequences =
        ceiling_div(scan_size, static_cast<unsigned int>(subsequence_size)); // "N"
    constexpr int block_size = 256; // "b", size in subsequences
    const int num_sequences =
        ceiling_div(num_subsequences, static_cast<unsigned int>(block_size)); // "B"

    // FIXME allocate and copy input on device

    // FIXME allocate output buffer
    int16_t* d_out = nullptr;

    subsequence_info* d_s_info;
    CHECK_CUDA(cudaMalloc(&d_s_info, num_subsequences * sizeof(subsequence_info)));

    { // sync_decoders (Algorithm 3)
        sync_intra_sequence<block_size><<<num_sequences, block_size, 0, cudaStreamDefault>>>(
            d_out, d_s_info, num_subsequences);
        CHECK_CUDA(cudaGetLastError());

        int* d_sequence_synced;
        CHECK_CUDA(cudaMalloc(&d_sequence_synced, num_sequences * sizeof(int)));

        bool exists_unsynced_sequence = true;
        do {
            constexpr int block_size_inter = 256;
            const int grid_dim =
                ceiling_div(num_sequences - 1, static_cast<unsigned int>(block_size_inter));
            sync_inter_sequence<block_size><<<grid_dim, block_size_inter, 0, cudaStreamDefault>>>(
                d_out, d_s_info, num_subsequences, d_sequence_synced);
            exists_unsynced_sequence; // FIXME = reduce(d_sequence_synced);
        } while (exists_unsynced_sequence);
        CHECK_CUDA(cudaFree(d_sequence_synced));
    }
    // FIXME exclusive prefix sum on s_info.n
    // FIXME s_info[0].n = 0
    decode_write<<<num_sequences, block_size, 0, cudaStreamDefault>>>(
        d_out, d_s_info, num_subsequences);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaFree(d_s_info));

    // TODO reverse DC difference coding

    // TODO zig-zag decoding, idct, quantization

    return JPEGGPU_SUCCESS;
}

} // namespace

jpeggpu_status process_scan(jpeggpu::reader& reader)
{
    return decode_gpu(reader.scan_start, reader.scan_size);
}
