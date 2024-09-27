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

#include "decode_destuff.hpp"
#include "decoder_defs.hpp"
#include "marker.hpp"

#include <cub/device/device_scan.cuh>

#include <cuda_runtime.h>

#include <cassert>
#include <stdint.h>

using namespace jpeggpu;

namespace {

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
__global__ void destuff_map_data_and_segment(
    const uint8_t* __restrict__ scan_stuffed,
    int scan_size,
    int* __restrict__ offset_data,
    int* __restrict__ offset_segment)
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

__global__ void destuff_write_and_map_subsequence(
    const uint8_t* __restrict__ scan_stuffed,
    int scan_size,
    const int* __restrict__ offset_data,
    const int* __restrict__ offset_segment,
    int* __restrict__ offset_subsequence,
    const segment* __restrict__ segments,
    int num_segments,
    uint8_t* __restrict__ scan_destuffed,
    int scan_size_destuffed)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= scan_size) {
        return;
    }

    const bool prev_is_stuffing = tid == 0 ? false : scan_stuffed[tid - 1] == 0xff;
    const uint8_t byte          = scan_stuffed[tid];
    uint8_t byte_write;
    const bool is_data = is_byte_data(prev_is_stuffing, byte, byte_write);

    bool is_first = false;
    if (is_data) {
        // write destuffed data
        const int segment_idx = offset_segment[tid];
        assert(segment_idx < num_segments);
        const segment& segment = segments[segment_idx];

        const int segment_offset = offset_data[tid];
        const int data_idx       = segment.subseq_offset * subsequence_size_bytes + segment_offset;
        assert(data_idx < scan_size_destuffed);
        scan_destuffed[data_idx] = byte_write;

        // this data byte is the first in the subsequence
        is_first = (segment_offset % subsequence_size_bytes) == 0;
    }

    offset_subsequence[tid] = is_first;
}

__global__ void write_segment_indices(
    const uint8_t* __restrict__ scan_stuffed,
    int scan_size,
    const int* __restrict__ offset_data,
    const int* __restrict__ offset_segment,
    int* __restrict__ offset_subsequence,
    const segment* __restrict__ segments,
    int num_segments,
    const uint8_t* __restrict__ scan_destuffed,
    int scan_size_destuffed,
    int* __restrict__ segment_indices)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= scan_size) {
        return;
    }

    const bool prev_is_stuffing = tid == 0 ? false : scan_stuffed[tid - 1] == 0xff;
    const uint8_t byte          = scan_stuffed[tid];
    uint8_t byte_write;
    const bool is_data = is_byte_data(prev_is_stuffing, byte, byte_write);

    // due to the inclusive scan, offset_subsequence is index + 1
    --offset_subsequence[tid];

    if (is_data) {
        // this data byte is the first in the subsequence
        const bool is_first = offset_data[tid] % subsequence_size_bytes == 0;
        if (is_first) {
            const int subseq_idx        = offset_subsequence[tid];
            segment_indices[subseq_idx] = offset_segment[tid];
        }
    }
}

} // namespace

template <bool do_it>
jpeggpu_status jpeggpu::destuff_scan(
    const jpeg_stream& info,
    const uint8_t* d_scan,
    uint8_t* d_scan_destuffed,
    const segment* d_segments,
    int* d_segment_indices,
    const scan& scan,
    stack_allocator& allocator,
    cudaStream_t stream,
    logger& logger)
{
    // by using many map and reduce operations the logic is completely parallelized and requires
    //   no synchronization with the host. an example of this process is given below

    // markers are d0 through d7, for this example subseq is 4 bytes
    // segment |                                      |                 |           |                    |
    // idx      00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f 10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f
    // val      ?? ?? ?? ff 00 ?? ff 00 ?? ?? ?? ff d0 ?? ff 00 ?? ff d1 ?? ?? ff d2 ?? ff 00 ?? ?? ff d3 ?? ??
    // destuff_map_data_and_segment
    // off_data  1  1  1  0  1  0  0  1  1  1  1  0  0  1  0  1  1  0  0  1  1  0  0  1  0  1  1  1  0  0  1  1
    // byte     ?? ?? ??    ff ??    ff ?? ?? ??       ??    ff ??       ?? ??       ??    ff ?? ??       ?? ??
    // off_seg   0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  1  0  0  0  1  0  0  0  0  0  0  1  0  0
    // exclusive sum off_seg
    // off_seg   0  0  0  0  0  0  0  0  0  0  0  0  1  1  1  1  1  1  2  2  2  2  3  3  3  3  3  3  3  4  4  4
    // exclusive sum off_data by off_seg
    // off_data  0  1  2  3  3  4  5  5  6  7  8  9  0  0  1  1  2  3  0  0  1  2  0  0  1  1  2  3  4  0  0  1
    // destuff_write_and_map_subsequence
    // destuff  ?? ?? ??    ff ??    ff ?? ?? ??       ??    ff ??       ?? ??       ??    ff ?? ??       ?? ??
    // off_sub   1  0  0  0  0  1  0  0  0  0  1  0  0  1  0  0  0  0  0  1  0  0  0  1  0  0  0  0  0  0  1  0
    // inclusive sum off_sub
    // off_sub   1  1  1  1  1  2  2  2  2  2  3  3  3  4  4  4  4  4  4  5  5  5  5  6  6  6  6  6  6  6  7  7
    // write_segment_indices
    // off_sub   0  0  0  0  0  1  1  1  1  1  2  2  2  3  3  3  3  3  3  4  4  4  4  5  5  5  5  5  5  5  6  6
    // seg_idx   0              0              0        1                 2           3                    4

    if (do_it) {
        // clear memory, only needed to satisfy `compute-sanitizer --tool=initcheck` since
        //   the segments are rounded up to subsequence size, some may not get written

        JPEGGPU_CHECK_CUDA(cudaMemsetAsync(
            d_scan_destuffed, 0, scan.num_subsequences * subsequence_size_bytes, stream));
    }

    const int stuffed_scan_size = scan.end - scan.begin;
    assert(stuffed_scan_size < INT_MAX);
    constexpr int block_size_destuff = 256;
    const int num_blocks_destuff =
        ceiling_div(stuffed_scan_size, static_cast<unsigned int>(block_size_destuff));

    // 1 if encoded data byte, 0 otherwise
    int* d_offset_data = nullptr;
    JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(&d_offset_data, stuffed_scan_size * sizeof(int)));

    // 1 if restart marker (after 0xFF), 0 otherwise
    int* d_offset_segment = nullptr;
    JPEGGPU_CHECK_STAT(
        allocator.reserve<do_it>(&d_offset_segment, stuffed_scan_size * sizeof(int)));

    if (do_it) {
        // write `d_offset_data` and `d_offset_segment`
        destuff_map_data_and_segment<<<num_blocks_destuff, block_size_destuff, 0, stream>>>(
            d_scan, stuffed_scan_size, d_offset_data, d_offset_segment);
        JPEGGPU_CHECK_CUDA(cudaGetLastError());
    }

    { // scan `d_offset_segment`
        void* d_tmp_storage     = nullptr;
        size_t tmp_storage_size = 0;
        JPEGGPU_CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
            d_tmp_storage,
            tmp_storage_size,
            d_offset_segment,
            d_offset_segment,
            stuffed_scan_size,
            stream));
        JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(&d_tmp_storage, tmp_storage_size));
        if (do_it) {
            JPEGGPU_CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
                d_tmp_storage,
                tmp_storage_size,
                d_offset_segment,
                d_offset_segment,
                stuffed_scan_size,
                stream));
        }
        // TODO since stream-ordered, should be able to reclaim tmp storage
    }

    // `d_offset_segment` is now segment index for every encoded data byte

    if (do_it && jpeggpu::is_debug) {
        std::vector<int> h_offset_segment(stuffed_scan_size);
        JPEGGPU_CHECK_CUDA(cudaMemcpy(
            h_offset_segment.data(),
            d_offset_segment,
            stuffed_scan_size * sizeof(int),
            cudaMemcpyDeviceToHost));
        if (h_offset_segment.back() + 1 != scan.num_segments) {
            logger.log("detected segment count inconsistent with calculated segment count\n");
            return JPEGGPU_INTERNAL_ERROR;
        }
    }

    { // scan `d_offset_data` by segment index
        void* d_tmp_storage     = nullptr;
        size_t tmp_storage_size = 0;
        JPEGGPU_CHECK_CUDA(cub::DeviceScan::ExclusiveSumByKey(
            d_tmp_storage,
            tmp_storage_size,
            d_offset_segment,
            d_offset_data,
            d_offset_data,
            stuffed_scan_size,
            cub::Equality{},
            stream));
        JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(&d_tmp_storage, tmp_storage_size));
        if (do_it) {
            JPEGGPU_CHECK_CUDA(cub::DeviceScan::ExclusiveSumByKey(
                d_tmp_storage,
                tmp_storage_size,
                d_offset_segment,
                d_offset_data,
                d_offset_data,
                stuffed_scan_size,
                cub::Equality{},
                stream));
        }
        // TODO since stream-ordered, should be able to reclaim tmp storage
    }

    // `d_offset_data` is now destuffed data index relative to segment

    // 1 if subsequence head, 0 otherwise
    int* d_offset_subsequence = nullptr;
    JPEGGPU_CHECK_STAT(
        allocator.reserve<do_it>(&d_offset_subsequence, stuffed_scan_size * sizeof(int)));

    const int destuffed_scan_size = scan.num_subsequences * subsequence_size_bytes;
    if (do_it) {
        destuff_write_and_map_subsequence<<<num_blocks_destuff, block_size_destuff, 0, stream>>>(
            d_scan,
            stuffed_scan_size,
            d_offset_data,
            d_offset_segment,
            d_offset_subsequence,
            d_segments,
            scan.num_segments,
            d_scan_destuffed,
            destuffed_scan_size);
        JPEGGPU_CHECK_CUDA(cudaGetLastError());
    }

    { // scan `d_offset_subsequence`
        void* d_tmp_storage     = nullptr;
        size_t tmp_storage_size = 0;
        JPEGGPU_CHECK_CUDA(cub::DeviceScan::InclusiveSum(
            d_tmp_storage,
            tmp_storage_size,
            d_offset_subsequence,
            d_offset_subsequence,
            stuffed_scan_size,
            stream));
        JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(&d_tmp_storage, tmp_storage_size));
        if (do_it) {
            JPEGGPU_CHECK_CUDA(cub::DeviceScan::InclusiveSum(
                d_tmp_storage,
                tmp_storage_size,
                d_offset_subsequence,
                d_offset_subsequence,
                stuffed_scan_size,
                stream));
        }
    }

    // `d_offset_subsequence` is now subsequence index + 1 for every encoded data byte

    if (do_it && jpeggpu::is_debug) {
        std::vector<int> h_offset_subsequence(stuffed_scan_size);
        JPEGGPU_CHECK_CUDA(cudaMemcpy(
            h_offset_subsequence.data(),
            d_offset_subsequence,
            stuffed_scan_size * sizeof(int),
            cudaMemcpyDeviceToHost));
        // h_offset_subsequence.back() is index + 1 of last stuffed byte, so it's equal to count
        if (h_offset_subsequence.back() != scan.num_subsequences) {
            logger.log(
                "detected subsequence count inconsistent with calculated subsequence count\n");
            return JPEGGPU_INTERNAL_ERROR;
        }
    }

    if (do_it) {
        write_segment_indices<<<num_blocks_destuff, block_size_destuff, 0, stream>>>(
            d_scan,
            stuffed_scan_size,
            d_offset_data,
            d_offset_segment,
            d_offset_subsequence,
            d_segments,
            scan.num_segments,
            d_scan_destuffed,
            destuffed_scan_size,
            d_segment_indices);
        JPEGGPU_CHECK_CUDA(cudaGetLastError());
    }

    // `d_offset_subsequence` is now subsequence index for every encoded data byte

    return JPEGGPU_SUCCESS;
}

template jpeggpu_status jpeggpu::destuff_scan<false>(
    const jpeg_stream&,
    const uint8_t*,
    uint8_t*,
    const segment*,
    int*,
    const scan&,
    stack_allocator&,
    cudaStream_t,
    logger&);
template jpeggpu_status jpeggpu::destuff_scan<true>(
    const jpeg_stream&,
    const uint8_t*,
    uint8_t*,
    const segment*,
    int*,
    const scan&,
    stack_allocator&,
    cudaStream_t,
    logger&);
