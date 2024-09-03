// Copyright (c) 2023-2024 Nol Moonen
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

#include "decoder.hpp"
#include "decode_dc.hpp"
#include "decode_huffman.hpp"
#include "decode_transpose.hpp"
#include "decoder_defs.hpp"
#include "defs.hpp"
#include "idct.hpp"
#include "marker.hpp"
#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <array>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdint.h>
#include <type_traits>
#include <vector>

/// TODO place in some development document
/// stream
///   The JPEG file defining an image with multiple components (color channels).
/// scan
///   A stream contains one or more scans. A scan contains one or more components.
///     The scan contains byte padding, which is removed in the destuffing process.
/// segment
///   Within a scan, there can be restart markers that define segments. The DC difference decoding does
///     not cross segment boundaries. In the scan, there may be sub-byte padding to ensure each
///     segments starts at a byte-boundary. The size of segments is arbitrary and is defined by the
///     encoding process.
/// sequence
///   For the purposes of GPU decoding, each thread handles a single subsequence, and all subsequences
///     handled by a thread block are grouped together as a sequence. Sequences may cover multiple segments.
/// subsequence
///   Practical fixed-sizes chunk of data that each thread handles. Subsequences do not cross segment boundaries.
///     The destuffing process pads the subsequence with zeroes such that each sequence starts on the beginning
///     of a subsequence.

using namespace jpeggpu;

jpeggpu_status jpeggpu::decoder::init()
{
    // TODO partial cleanup on error

    jpeggpu_status stat = JPEGGPU_SUCCESS;
    if ((stat = reader.startup()) != JPEGGPU_SUCCESS) {
        return stat;
    }

    allocator = stack_allocator{};
    logger    = jpeggpu::logger{};

    JPEGGPU_CHECK_STAT(d_segments.startup(4));

    return JPEGGPU_SUCCESS;
}

void jpeggpu::decoder::cleanup()
{
    d_segments.cleanup();
    reader.cleanup();
}

jpeggpu_status jpeggpu::decoder::parse_header(
    jpeggpu_img_info& img_info, const uint8_t* data, size_t size)
{
    reader.reset(data, data + size);
    jpeggpu_status stat = reader.read(logger);
    if (stat != JPEGGPU_SUCCESS) {
        return stat;
    }

    // set all fields of jpeggpu_img_info
    for (int c = 0; c < reader.jpeg_stream.num_components; ++c) {
        img_info.sizes_x[c] = reader.jpeg_stream.components[c].size.x;
        img_info.sizes_y[c] = reader.jpeg_stream.components[c].size.y;
    }
    for (int c = reader.jpeg_stream.num_components; c < max_comp_count; ++c) {
        img_info.sizes_x[c] = 0;
        img_info.sizes_y[c] = 0;
    }
    img_info.num_components = reader.jpeg_stream.num_components;
    for (int c = 0; c < reader.jpeg_stream.num_components; ++c) {
        img_info.subsampling.x[c] = reader.jpeg_stream.components[c].ss.x;
        img_info.subsampling.y[c] = reader.jpeg_stream.components[c].ss.y;
    }
    for (int c = reader.jpeg_stream.num_components; c < max_comp_count; ++c) {
        img_info.subsampling.x[c] = 0;
        img_info.subsampling.y[c] = 0;
    }

    return JPEGGPU_SUCCESS;
}

namespace {

/// \brief Reserves data in the temporary memory for the portion that should be copied from
///   device to host. Since these pointers are created twice in two different functions
///   `jpeggpu_decoder_transfer` and `jpeggpu_decoder_decode`/`jpeggpu_decoder_get_buffer_size`,
///   these allocation will be at the front of the allocation.
template <bool do_it>
jpeggpu_status reserve_transfer_data(
    const reader& reader,
    stack_allocator& allocator,
    uint8_t*& d_image_data,
    qtable*& d_qtables,
    huffman_table*& d_huff_tables,
    list<segment*>& d_segments)
{
    if (allocator.size != 0) {
        // should be first in allocation, to ensure it's in the same place
        return JPEGGPU_INTERNAL_ERROR;
    }

    d_image_data = nullptr;
    JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(&d_image_data, reader.get_file_size()));
    d_qtables = nullptr;
    JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(&d_qtables, reader.jpeg_stream.cnt_qtab));
    d_huff_tables = nullptr;
    JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(&d_huff_tables, reader.jpeg_stream.cnt_huff));
    for (int s = 0; s < d_segments.get_size(); ++s) {
        d_segments[s] = nullptr;
        JPEGGPU_CHECK_STAT(
            allocator.reserve<do_it>(&d_segments[s], reader.jpeg_stream.scans[s].num_segments));
    }

    return JPEGGPU_SUCCESS;
}

} // namespace

jpeggpu_status jpeggpu::decoder::transfer(void* d_tmp, size_t tmp_size, cudaStream_t stream)
{
    allocator.reset(d_tmp, tmp_size);

    const jpeg_stream& info = reader.jpeg_stream;

    uint8_t* d_image_data        = nullptr;
    qtable* d_qtables            = nullptr;
    huffman_table* d_huff_tables = nullptr;
    JPEGGPU_CHECK_STAT(d_segments.resize(info.num_scans));
    JPEGGPU_CHECK_STAT(reserve_transfer_data<true>(
        reader, allocator, d_image_data, d_qtables, d_huff_tables, d_segments));

    // JPEG stream
    JPEGGPU_CHECK_CUDA(cudaMemcpyAsync(
        d_image_data,
        reader.reader_state.image_begin,
        reader.get_file_size(),
        cudaMemcpyHostToDevice,
        stream));
    /// Huffman tables
    JPEGGPU_CHECK_CUDA(cudaMemcpyAsync(
        d_huff_tables,
        reader.h_huff_tables.ptr,
        info.cnt_huff * sizeof(*reader.h_huff_tables.ptr),
        cudaMemcpyHostToDevice,
        stream));
    // quantization tables
    JPEGGPU_CHECK_CUDA(cudaMemcpyAsync(
        d_qtables,
        reader.h_qtables.ptr,
        info.cnt_qtab * sizeof(*reader.h_qtables.ptr),
        cudaMemcpyHostToDevice,
        stream));
    // segment info
    for (int s = 0; s < info.num_scans; ++s) {
        JPEGGPU_CHECK_CUDA(cudaMemcpyAsync(
            d_segments[s],
            reader.h_scan_segments[s].ptr,
            info.scans[s].num_segments * sizeof(*reader.h_scan_segments[s].ptr),
            cudaMemcpyHostToDevice,
            stream));
    }

    return JPEGGPU_SUCCESS;
}

/// \tparam do_it If true, this function (and all other functions with this parameter) should not
///   perform any work. Instead, they should just walk through the entire decoding process to
///   calculate memory requirements.
template <bool do_it>
jpeggpu_status jpeggpu::decoder::decode_impl([[maybe_unused]] jpeggpu_img* img, cudaStream_t stream)
{
    // const jpeg_stream& info = reader.jpeg_stream;

    // uint8_t* d_image_data        = nullptr;
    // qtable* d_qtables            = nullptr;
    // huffman_table* d_huff_tables = nullptr;
    // JPEGGPU_CHECK_STAT(d_segments.resize(info.num_scans));
    // JPEGGPU_CHECK_STAT(reserve_transfer_data<do_it>(
    //     reader, allocator, d_image_data, d_qtables, d_huff_tables, d_segments));

    // for (int c = 0; c < info.num_components; ++c) {
    //     const size_t size = info.components[c].data_size.x * info.components[c].data_size.y;
    //     JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(&(d_image_qdct[c]), size * sizeof(int16_t)));
    // }

    // size_t total_data_size = 0;
    // for (int c = 0; c < info.num_components; ++c) {
    //     total_data_size += info.components[c].data_size.x * info.components[c].data_size.y;
    // }
    // int16_t* d_out;
    // JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(&d_out, total_data_size * sizeof(int16_t)));
    // if (do_it) {
    //     // initialize to zero, since only non-zeros are written by Huffman decoding
    //     JPEGGPU_CHECK_CUDA(cudaMemsetAsync(d_out, 0, total_data_size * sizeof(int16_t), stream));
    // }

    // // TODO if all scans are sequentially pushed through all destuff kernel stages and all decode stages,
    // //   instead of decoding each scan sequentially, could this improve performance due to less divergence?

    // // destuff the scan and decode the Huffman stream
    // size_t offset = 0;
    // for (int c = 0; c < info.num_scans; ++c) {
    //     const scan& scan = info.scans[c];

    //     uint8_t* d_scan_destuffed = nullptr;
    //     // all subsequences are "rounded up" in size, which allows for easier offset
    //     //   calculations in the Huffman decoder
    //     const size_t scan_byte_size = scan.num_subsequences * subsequence_size_bytes;
    //     // scan should be properly aligned by allocator to do optimized reads
    //     //   (reading more bytes than one at a time) in the Huffman decoder
    //     JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(&d_scan_destuffed, scan_byte_size));

    //     // for each subsequence, its segment
    //     int* d_segment_indices = nullptr;
    //     JPEGGPU_CHECK_STAT(
    //         allocator.reserve<do_it>(&d_segment_indices, scan.num_subsequences * sizeof(int)));

    //     const uint8_t* d_scan = d_image_data + scan.begin;
    //     JPEGGPU_CHECK_STAT(destuff_scan<do_it>(
    //         info,
    //         d_scan,
    //         d_scan_destuffed,
    //         d_segments[c],
    //         d_segment_indices,
    //         scan,
    //         allocator,
    //         stream,
    //         logger));

    //     JPEGGPU_CHECK_STAT(decode_scan<do_it>(
    //         info,
    //         d_scan_destuffed,
    //         d_segments[c],
    //         d_segment_indices,
    //         d_out + offset,
    //         scan,
    //         d_huff_tables,
    //         allocator,
    //         stream,
    //         logger));

    //     for (int c = 0; c < scan.num_components; ++c) {
    //         const component& comp = info.components[scan.component_indices[c]];
    //         offset += comp.data_size.x * comp.data_size.y;
    //     }
    // }

    // // TODO is "data unit" the correct terminology?

    // // after decoding, the data is as how it appears in the encoded stream: one data unit at a time
    // //   (i.e. 64 bytes), the data units possibly interleaved in the MCUs

    // for (int c = 0; c < info.num_scans; ++c) {
    //     // undo DC difference encoding
    //     JPEGGPU_CHECK_STAT(decode_dc<do_it>(info, info.scans[c], d_out, allocator, stream, logger));
    // }

    // if (do_it) {
    //     size_t offset = 0;
    //     for (int c = 0; c < info.num_scans; ++c) {
    //         // Convert data order from data unit at a time to raster order,
    //         //   and place the components from scan order to frame order.
    //         const scan& scan = info.scans[c];
    //         JPEGGPU_CHECK_STAT(
    //             decode_transpose(info, d_out + offset, scan, d_image_qdct, stream, logger));

    //         for (int c = 0; c < scan.num_components; ++c) {
    //             const component& comp = info.components[scan.component_indices[c]];
    //             offset += comp.data_size.x * comp.data_size.y;
    //         }
    //     }

    //     // invert DCT and output directly into user-provided buffer
    //     JPEGGPU_CHECK_STAT(
    //         idct(info, d_image_qdct, img->image, img->pitch, d_qtables, stream, logger));
    // }

    return JPEGGPU_SUCCESS;
}

jpeggpu_status jpeggpu::decoder::decode_get_size(size_t& tmp_size_param)
{
    allocator.reset();
    // TODO add check if stream is "dereferenced" when do_it is false, doing so is an error
    JPEGGPU_CHECK_STAT(decode_impl<false>(nullptr, nullptr));
    tmp_size_param = allocator.size;
    return JPEGGPU_SUCCESS;
}

jpeggpu_status jpeggpu::decoder::decode(
    jpeggpu_img* img, void* d_tmp_param, size_t tmp_size_param, cudaStream_t stream)
{
    if (!img) {
        // not invalid argument since this is an internal function
        return JPEGGPU_INTERNAL_ERROR;
    }

    const jpeg_stream& info = reader.jpeg_stream;
    for (int c = 0; c < info.num_components; ++c) {
        if (!img->image[c] || img->pitch[c] < info.components[c].size.x) {
            return JPEGGPU_INVALID_ARGUMENT;
        }
    }

    allocator.reset(d_tmp_param, tmp_size_param);
    JPEGGPU_CHECK_STAT(decode_impl<true>(img, stream));
    return JPEGGPU_SUCCESS;
}
