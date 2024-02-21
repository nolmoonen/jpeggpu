#include "decoder_defs.hpp"
#include "destuff.hpp"
#include "marker.hpp"

#include <cub/device/device_scan.cuh>

#include <cuda_runtime.h>

#include <cassert>
#include <stdint.h>

__device__ bool is_byte_data(bool prev_is_stuffing, uint8_t byte, uint8_t& byte_write)
{
    // register 0xff00 as 0xff at the position of the 0x00, do not register at the position of 0xff
    // ignore 0xff?? at the positions of both 0xff and 0x??
    const bool is_data = (prev_is_stuffing && byte == 0) || (!prev_is_stuffing && byte != 0xff);
    byte_write         = prev_is_stuffing ? 0xff : byte;
    return is_data;
}

/// \brief Handle one byte each. Probably suboptimal, but simple.
///   For each byte, set [0,1] in `offset_data` whether it represents encoded scan data.
///   And [0,1] in `offset_segment` whether it is a restart marker.
///
/// \param[in] scan_stuffed
/// \param[out] offset_data For each stuffed byte: one if it is encoded scan data, else zero.
/// \param[out] offset_segment For each stuffed byte: one if it is a restart marker, else zero.
__global__ void destuff_categorize(
    const uint8_t* scan_stuffed, int scan_size, int* offset_data, int* offset_segment)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= scan_size) {
        return;
    }

    const bool prev_is_stuffing = tid == 0 ? false : scan_stuffed[tid - 1] == 0xff;
    const uint8_t byte          = scan_stuffed[tid];
    uint8_t byte_write; // unused
    const bool is_data = is_byte_data(prev_is_stuffing, byte, byte_write);
    offset_data[tid]   = is_data;
    const bool is_restart_marker =
        prev_is_stuffing && jpeggpu::MARKER_RST0 <= byte && byte <= jpeggpu::MARKER_RST7;
    offset_segment[tid] = is_restart_marker;
    assert(!(is_data && is_restart_marker));
}

/// \brief
///
/// \param[in] scan_stuffed
/// \param[in] offset_data For each stuffed byte: if it is encoded scan data, its index.
/// \param[in] offset_segment For each stuffed byte: if it is a restart marker, its index.
/// \param[out] scan For each destuffed byte: the encoded scan data bytes.
/// \param[out] restarts For each segment: the begin and end offsets in destuffed scan.
/// \param[out] segment_indices For each destuffed byte: the index of the segment it is in.
__global__ void destuff_write(
    const uint8_t* scan_stuffed,
    int scan_size,
    const int* offset_data,
    const int* offset_segment,
    uint8_t* scan,
    jpeggpu::segment_info* segments)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= scan_size) {
        return;
    }

    const bool prev_is_stuffing = tid == 0 ? false : scan_stuffed[tid - 1] == 0xff;
    const uint8_t byte          = scan_stuffed[tid];
    uint8_t byte_write;
    const bool is_data = is_byte_data(prev_is_stuffing, byte, byte_write);
    if (is_data) {
        scan[offset_data[tid]] = byte_write;
    }
    const bool is_restart_marker =
        prev_is_stuffing && jpeggpu::MARKER_RST0 <= byte && byte <= jpeggpu::MARKER_RST7;
    if (is_restart_marker) {
        // the destuffed index of the end byte (exclusive) of the next segment
        const int this_segment_end                      = offset_data[tid - 1];
        segments[offset_segment[tid]].end               = this_segment_end;
        // the destuffed index of the begin byte of the next segment
        const int next_segment_begin                    = offset_data[tid + 1];
        segments[offset_segment[tid] + 1].begin         = next_segment_begin;
        segments[offset_segment[tid] + 1].subseq_offset = 0; // FIXME debug satisfy initcheck
    }
    if (tid == 0) {
        segments[offset_segment[tid]].begin         = 0;
        segments[offset_segment[tid]].subseq_offset = 0; // FIXME debug satisfy initcheck
    } else if (tid == scan_size - 1) {
        // offset_data only holds the index if this byte is a data byte
        segments[offset_segment[tid]].end = is_data ? offset_data[tid] + 1 : offset_data[tid];
    }
}

__global__ void destuff_write2(
    const uint8_t* scan_stuffed,
    int scan_size,
    const int* offset_data,
    const int* offset_segment,
    const uint8_t*,
    const jpeggpu::segment_info* segments,
    int* offset_subsequence)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= scan_size) {
        return;
    }

    const bool prev_is_stuffing = tid == 0 ? false : scan_stuffed[tid - 1] == 0xff;
    const uint8_t byte          = scan_stuffed[tid];
    uint8_t byte_write;
    const bool is_data = is_byte_data(prev_is_stuffing, byte, byte_write);
    bool is_first      = false;
    if (is_data) {
        const jpeggpu::segment_info& seg_info = segments[offset_segment[tid]];
        // this data byte is the first in the subsequence
        is_first = (offset_data[tid] - seg_info.begin) % jpeggpu::subsequence_size_bytes == 0;
    }
    offset_subsequence[tid] = is_first;
}

__global__ void destuff_write3(
    const uint8_t* scan_stuffed,
    int scan_size,
    const int* offset_data,
    const int* offset_segment,
    const uint8_t*,
    jpeggpu::segment_info* segments,
    const int* offset_subsequence,
    int* segment_indices)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= scan_size) {
        return;
    }

    const bool prev_is_stuffing = tid == 0 ? false : scan_stuffed[tid - 1] == 0xff;
    const uint8_t byte          = scan_stuffed[tid];
    uint8_t byte_write;
    const bool is_data = is_byte_data(prev_is_stuffing, byte, byte_write);
    if (is_data) {
        const jpeggpu::segment_info& seg_info = segments[offset_segment[tid]];
        // this data byte is the first in the subsequence
        const bool is_first =
            (offset_data[tid] - seg_info.begin) % jpeggpu::subsequence_size_bytes == 0;
        if (is_first) {
            segment_indices[offset_subsequence[tid]] = offset_segment[tid];
        }
    }
    const bool is_restart_marker =
        prev_is_stuffing && jpeggpu::MARKER_RST0 <= byte && byte <= jpeggpu::MARKER_RST7;
    if (is_restart_marker) {
        segments[offset_segment[tid] + 1].subseq_offset = offset_subsequence[tid];
    }
    if (tid == 0) {
        segments[offset_segment[tid]].subseq_offset = 0;
    }
}

// TODO is a version without memcpies possible?
jpeggpu_status jpeggpu::destuff(
    reader& reader,
    uint8_t*& d_scan,
    int& scan_size,
    int& num_subsequences,
    segment_info*& d_segment_infos,
    int*& d_segment_indices,
    cudaStream_t stream)
{
    assert(reader.scan_size < INT_MAX);

    const int stuffed_scan_size      = reader.scan_size;
    constexpr int block_size_destuff = 256;
    const int num_blocks_destuff =
        ceiling_div(stuffed_scan_size, static_cast<unsigned int>(block_size_destuff));

    // stuffed input
    uint8_t* d_stuffed_scan = nullptr;
    CHECK_CUDA(cudaMalloc(&d_stuffed_scan, stuffed_scan_size));
    CHECK_CUDA(cudaMemcpyAsync(
        d_stuffed_scan, reader.scan_start, stuffed_scan_size, cudaMemcpyHostToDevice, stream));

    // exclusive prefix scan of encoded data bytes
    int* d_offset_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_offset_data, stuffed_scan_size * sizeof(int)));

    // exclusive prefix scan of restart markers
    int* d_offset_segment = nullptr;
    CHECK_CUDA(cudaMalloc(&d_offset_segment, stuffed_scan_size * sizeof(int)));

    // write `d_offset_data` and `d_offset_segment`
    destuff_categorize<<<num_blocks_destuff, block_size_destuff, 0, stream>>>(
        d_stuffed_scan, stuffed_scan_size, d_offset_data, d_offset_segment);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize()); // FIXME debug

    { // scan `d_offset_data`
        void* d_tmp_storage     = nullptr;
        size_t tmp_storage_size = 0;
        CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
            d_tmp_storage,
            tmp_storage_size,
            d_offset_data,
            d_offset_data,
            stuffed_scan_size,
            stream));
        CHECK_CUDA(cudaMalloc(&d_tmp_storage, tmp_storage_size));
        CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
            d_tmp_storage,
            tmp_storage_size,
            d_offset_data,
            d_offset_data,
            stuffed_scan_size,
            stream));
        CHECK_CUDA(cudaFree(d_tmp_storage));
    }
    { // scan `d_offset_segment`
        void* d_tmp_storage     = nullptr;
        size_t tmp_storage_size = 0;
        CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
            d_tmp_storage,
            tmp_storage_size,
            d_offset_segment,
            d_offset_segment,
            stuffed_scan_size,
            stream));
        CHECK_CUDA(cudaMalloc(&d_tmp_storage, tmp_storage_size));
        CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
            d_tmp_storage,
            tmp_storage_size,
            d_offset_segment,
            d_offset_segment,
            stuffed_scan_size,
            stream));
        CHECK_CUDA(cudaFree(d_tmp_storage));
    }

    // TODO interleaved
    const int num_segments = reader.num_segments;
    if (is_debug) {
        int num_segments_found = 0;
        CHECK_CUDA(cudaMemcpy(
            &num_segments_found,
            &d_offset_segment[stuffed_scan_size - 1],
            sizeof(int),
            cudaMemcpyDeviceToHost));
        ++num_segments_found;

        if (num_segments != num_segments_found) {
            (*reader.logger)(
                "num segments calculated: %d, num segments found: %d\n",
                num_segments,
                num_segments_found);
            return JPEGGPU_INTERNAL_ERROR;
        }

        int destuffed_scan_size = 0;
        CHECK_CUDA(cudaMemcpy(
            &destuffed_scan_size,
            &d_offset_data[stuffed_scan_size - 1],
            sizeof(int),
            cudaMemcpyDeviceToHost));
        // scan is exclusive, so if final byte is data, manually add it
        if (reader.scan_size >= 2 && reader.scan_start[reader.scan_size - 2] == 0xff &&
            reader.scan_start[reader.scan_size - 1] == 0) {
            ++destuffed_scan_size;
        }

        if (destuffed_scan_size > reader.scan_size) {
            std::cout << "destuffed scan size: " << destuffed_scan_size
                      << ", stuffed scan size: " << reader.scan_size << "\n";
            return JPEGGPU_INTERNAL_ERROR;
        }
    }
    (*reader.logger)("num segments %d\n", num_segments);

    // destuffed output
    d_scan = nullptr;
    CHECK_CUDA(cudaMalloc(&d_scan, reader.scan_size));

    d_segment_infos = nullptr;
    CHECK_CUDA(cudaMalloc(&d_segment_infos, num_segments * sizeof(jpeggpu::segment_info)));

    // completes data and segment infos
    destuff_write<<<num_blocks_destuff, block_size_destuff, 0, stream>>>(
        d_stuffed_scan,
        stuffed_scan_size,
        d_offset_data,
        d_offset_segment,
        d_scan,
        d_segment_infos);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize()); // FIXME debug

    std::vector<segment_info> h_segment_infos(num_segments);
    CHECK_CUDA(cudaMemcpy(
        h_segment_infos.data(),
        d_segment_infos,
        num_segments * sizeof(segment_info),
        cudaMemcpyDeviceToHost));
    num_subsequences = 0;
    for (int i = 0; i < num_segments; ++i) {
        if (jpeggpu::is_debug && h_segment_infos[i].end <= h_segment_infos[i].begin) {
            (*reader.logger)(
                "segment %d begin: %d, end: %d\n",
                i,
                h_segment_infos[i].begin,
                h_segment_infos[i].end);
            return JPEGGPU_INTERNAL_ERROR;
        }
        num_subsequences += ceiling_div(h_segment_infos[i].end - h_segment_infos[i].begin, 32u);
    }

    // for each destuffed byte, its subsequence
    int* d_offset_subsequence = nullptr;
    CHECK_CUDA(cudaMalloc(&d_offset_subsequence, stuffed_scan_size * sizeof(int)));

    destuff_write2<<<num_blocks_destuff, block_size_destuff, 0, stream>>>(
        d_stuffed_scan,
        stuffed_scan_size,
        d_offset_data,
        d_offset_segment,
        d_scan,
        d_segment_infos,
        d_offset_subsequence);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize()); // FIXME debug

    { // scan `d_offset_subsequence`
        void* d_tmp_storage     = nullptr;
        size_t tmp_storage_size = 0;
        CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
            d_tmp_storage,
            tmp_storage_size,
            d_offset_subsequence,
            d_offset_subsequence,
            stuffed_scan_size,
            stream));
        CHECK_CUDA(cudaMalloc(&d_tmp_storage, tmp_storage_size));
        CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
            d_tmp_storage,
            tmp_storage_size,
            d_offset_subsequence,
            d_offset_subsequence,
            stuffed_scan_size,
            stream));
        CHECK_CUDA(cudaFree(d_tmp_storage));
    }

    // for each subsequence, its segment
    d_segment_indices = nullptr;
    CHECK_CUDA(cudaMalloc(&d_segment_indices, num_subsequences * sizeof(int)));

    destuff_write3<<<num_blocks_destuff, block_size_destuff, 0, stream>>>(
        d_stuffed_scan,
        stuffed_scan_size,
        d_offset_data,
        d_offset_segment,
        d_scan,
        d_segment_infos,
        d_offset_subsequence,
        d_segment_indices);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize()); // FIXME debug

    if (jpeggpu::is_debug) {
        std::vector<int> h_segment_indices(num_subsequences);
        CHECK_CUDA(cudaMemcpy(
            h_segment_indices.data(),
            d_segment_indices,
            num_subsequences * sizeof(int),
            cudaMemcpyDeviceToHost));
        for (int i = 0; i < num_subsequences; ++i) {
            if (h_segment_indices[i] < 0 || h_segment_indices[i] >= num_segments) {
                (*reader.logger)(
                    "subquence %d invalid segment index %d\n", i, h_segment_indices[i]);
            }
        }
    }

    CHECK_CUDA(cudaFree(d_offset_subsequence));
    CHECK_CUDA(cudaFree(d_offset_segment));
    CHECK_CUDA(cudaFree(d_offset_data));
    CHECK_CUDA(cudaFree(d_stuffed_scan));

    return JPEGGPU_SUCCESS;
}

// clang-format off
// markers are d0 through d7
//          00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f 10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f
//          ?? ?? ?? ff 00 ?? ff 00 ?? ?? ?? ff d0 ?? ff 00 ?? ff d1 ?? ?? ff d2 ?? ff 00 ?? ?? ff d3 ?? ??
// pass 0
// off_data  1  1  1  0  1  0  0  1  1  1  1  0  0  1  0  1  1  0  0  1  1  0  0  1  0  1  1  1  0  0  1  1
// byte     ?? ?? ??    ff ??    ff ?? ?? ??       ??    ff ??       ?? ??       ??    ff ?? ??       ?? ??
// off_seg   0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  1  0  0  0  1  0  0  0  0  0  0  1  0  0
// scan exclusive, pass 1
// off_data  0  1  2  3  3  4  5  5  6  7  8  9  9  9 10 10 11 12 12 12 13 14 14 14 15 15 16 17 18 18 18 19
// off_seg   0  0  0  0  0  0  0  0  0  0  0  0  1  1  1  1  1  1  2  2  2  2  3  3  3  3  3  3  3  4  4  4
// pass 2
// seg_idx   0  0  0     0  0     0  0  0  0        1     1  1        2  2        3     3  3  3        4  4
// clang-format on
