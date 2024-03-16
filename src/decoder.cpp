#include "decoder.hpp"
#include "convert.hpp"
#include "decode_dc.hpp"
#include "decode_huffman.hpp"
#include "decode_transpose.hpp"
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
///   A stream contains scans. Either a single scan contains all components (interleaved mode)
///     or there are multiple scans that each contain a single component.
/// segment
///   Within a scan, there can be restart markers that define segments. The DC difference decoding does
///     not cross segment boundaries. Between segments there may be sub-byte padding to ensure each
///     segments starts at a byte-boundary. The size of segments is arbitrary and is defined by the
///     encoding process.
/// sequence
///   For the purposes of GPU decoding, each thread handles a single subsequence, and all subsequences
///     handled by a thread block are grouped together as a sequence. Sequences may cover multiple segments.
/// subsequence
///   Practical fixed-sizes chunk of data that each thread handles. Subsequences do not cross segment boundaries.

using namespace jpeggpu;

jpeggpu_status jpeggpu::decoder::init()
{
    for (int c = 0; c < max_comp_count; ++c) {
        JPEGGPU_CHECK_CUDA(cudaMalloc(&(d_qtables[c]), sizeof(uint8_t) * data_unit_size));
    }
    for (int i = 0; i < max_huffman_count; ++i) {
        for (int j = 0; j < HUFF_COUNT; ++j) {
            JPEGGPU_CHECK_CUDA(cudaMalloc(&(d_huff_tables[i][j]), sizeof(*d_huff_tables[i][j])));
        }
    }

    jpeggpu_status stat = JPEGGPU_SUCCESS;
    if ((stat = reader.startup()) != JPEGGPU_SUCCESS) {
        return stat;
    }

    new (&allocator.stack) std::vector<jpeggpu::allocation>();

    return JPEGGPU_SUCCESS;
}

void jpeggpu::decoder::cleanup()
{
    reader.cleanup();

    for (int i = 0; i < max_huffman_count; ++i) {
        for (int j = 0; j < HUFF_COUNT; ++j) {
            cudaFree(d_huff_tables[i][j]);
            d_huff_tables[i][j] = nullptr;
        }
    }
    for (int c = 0; c < max_comp_count; ++c) {
        cudaFree(d_qtables[c]);
        d_qtables[c] = nullptr;
    }
}

jpeggpu_status jpeggpu::decoder::parse_header(
    jpeggpu_img_info& img_info, const uint8_t* data, size_t size)
{
    reader.reset(data, data + size);
    jpeggpu_status stat = reader.read();
    if (stat != JPEGGPU_SUCCESS) {
        return stat;
    }

    // TODO check reader consistency

    img_info.size_x = reader.jpeg_stream.size_x;
    img_info.size_y = reader.jpeg_stream.size_y;
    // TODO read metadata to determine color formats
    switch (reader.jpeg_stream.num_components) {
    case 1:
        reader.jpeg_stream.color_fmt = JPEGGPU_GRAY;
        reader.jpeg_stream.pixel_fmt = JPEGGPU_P0;
        break;
    case 3:
        reader.jpeg_stream.color_fmt = JPEGGPU_YCBCR;
        reader.jpeg_stream.pixel_fmt = JPEGGPU_P0P1P2;
        break;
    case 4:
        reader.jpeg_stream.color_fmt = JPEGGPU_CMYK;
        reader.jpeg_stream.pixel_fmt = JPEGGPU_P0P1P2P3;
        break;
    default:
        return JPEGGPU_NOT_SUPPORTED;
    }

    for (int c = 0; c < reader.jpeg_stream.num_components; ++c) {
        img_info.subsampling.x[c] =
            reader.jpeg_stream.ss_x_max / reader.jpeg_stream.components[c].ss_x;
        img_info.subsampling.y[c] =
            reader.jpeg_stream.ss_y_max / reader.jpeg_stream.components[c].ss_y;
    }
    for (int c = reader.jpeg_stream.num_components; c < max_comp_count; ++c) {
        img_info.subsampling.x[c] = 0;
        img_info.subsampling.y[c] = 0;
    }

    return JPEGGPU_SUCCESS;
}

namespace {

inline bool operator==(const jpeggpu_subsampling& lhs, const jpeggpu_subsampling& rhs)
{
    for (int c = 0; c < jpeggpu::max_comp_count; ++c) {
        if (lhs.x[c] != rhs.x[c] || lhs.y[c] != rhs.y[c]) {
            return false;
        }
    }
    return true;
}

inline bool operator!=(const jpeggpu_subsampling& lhs, const jpeggpu_subsampling& rhs)
{
    return !(lhs == rhs);
}

} // namespace

/// \tparam do_it If true, this function (and all other functions) should not perform any work. Instead,
///   They should just walk through the entire decoding process to calculate memory requirements.
template <bool do_it>
jpeggpu_status jpeggpu::decoder::decode_impl(
    jpeggpu_img& img,
    jpeggpu_color_format color_fmt,
    jpeggpu_pixel_format pixel_fmt,
    jpeggpu_subsampling subsampling,
    cudaStream_t stream)
{
    const jpeg_stream& info = reader.jpeg_stream;
    for (int c = 0; c < info.num_components; ++c) {
        const size_t size = info.components[c].data_size_x * info.components[c].data_size_y;
        JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(&(d_image_qdct[c]), size * sizeof(int16_t)));
        JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(&(d_image[c]), size * sizeof(uint8_t)));
    }
    const size_t file_size = reader.image_end - reader.image_begin;
    uint8_t* d_image_data  = nullptr;
    JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(&d_image_data, file_size));
    uint8_t* d_image_data_destuffed = nullptr;
    JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(&d_image_data_destuffed, file_size));

    // TODO put in separate API function
    if (do_it) {
        JPEGGPU_CHECK_CUDA(cudaMemcpyAsync(
            d_image_data, reader.image_begin, file_size, cudaMemcpyHostToDevice, stream));
        for (int i = 0; i < max_huffman_count; ++i) {
            for (int j = 0; j < HUFF_COUNT; ++j) {
                JPEGGPU_CHECK_CUDA(cudaMemcpyAsync(
                    d_huff_tables[i][j],
                    reader.h_huff_tables[i][j],
                    sizeof(*reader.h_huff_tables[i][j]),
                    cudaMemcpyHostToDevice,
                    stream));
            }
        }
        for (int c = 0; c < info.num_components; ++c) {
            JPEGGPU_CHECK_CUDA(cudaMemcpyAsync(
                d_qtables[c],
                reader.h_qtables[info.components[c].qtable_idx],
                sizeof(*reader.h_qtables[info.components[c].qtable_idx]),
                cudaMemcpyHostToDevice,
                stream));
        }
    }

    size_t total_data_size = 0;
    for (int c = 0; c < info.num_components; ++c) {
        total_data_size += info.components[c].data_size_x * info.components[c].data_size_y;
    }
    int16_t* d_out;
    JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(&d_out, total_data_size * sizeof(int16_t)));
    if (do_it) {
        // initialize to zero, since only non-zeros are written
        JPEGGPU_CHECK_CUDA(cudaMemsetAsync(d_out, 0, total_data_size * sizeof(int16_t), stream));
    }

    if (info.is_interleaved) {
        const scan& scan = info.scans[0];

        segment_info* d_segment_infos = nullptr;
        JPEGGPU_CHECK_STAT(
            allocator.reserve<do_it>(&d_segment_infos, scan.num_segments * sizeof(segment_info)));

        // for each subsequence, its segment
        int* d_segment_indices = nullptr;
        JPEGGPU_CHECK_STAT(
            allocator.reserve<do_it>(&d_segment_indices, scan.num_subsequences * sizeof(int)));

        const uint8_t* d_scan     = d_image_data + scan.begin;
        uint8_t* d_scan_destuffed = d_image_data_destuffed + scan.begin;

        JPEGGPU_CHECK_STAT(destuff_scan<do_it>(
            info,
            d_segment_infos,
            d_segment_indices,
            d_scan,
            d_scan_destuffed,
            scan,
            allocator,
            stream));

        JPEGGPU_CHECK_STAT(decode_scan<do_it>(
            info,
            d_scan_destuffed,
            d_segment_infos,
            d_segment_indices,
            d_out,
            scan,
            d_huff_tables,
            allocator,
            stream));
    } else {
        size_t offset = 0;
        for (int c = 0; c < info.num_scans; ++c) {
            const scan& scan = info.scans[c];

            // TODO if these allocations just once, the maximum across scans, they can be reused
            segment_info* d_segment_infos = nullptr;
            JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(
                &d_segment_infos, scan.num_segments * sizeof(segment_info)));

            // for each subsequence, its segment
            int* d_segment_indices = nullptr;
            JPEGGPU_CHECK_STAT(
                allocator.reserve<do_it>(&d_segment_indices, scan.num_subsequences * sizeof(int)));

            const uint8_t* d_scan     = d_image_data + scan.begin;
            uint8_t* d_scan_destuffed = d_image_data_destuffed + scan.begin;

            JPEGGPU_CHECK_STAT(destuff_scan<do_it>(
                info,
                d_segment_infos,
                d_segment_indices,
                d_scan,
                d_scan_destuffed,
                scan,
                allocator,
                stream));

            JPEGGPU_CHECK_STAT(decode_scan<do_it>(
                info,
                d_scan_destuffed,
                d_segment_infos,
                d_segment_indices,
                d_out + offset,
                scan,
                d_huff_tables,
                allocator,
                stream));

            const int comp_id = scan.ids[0];
            offset += info.components[comp_id].data_size_x * info.components[comp_id].data_size_y;
        }
    }

    // data is now as it appears in the encoded stream: one data unit at a time, possibly interleaved
    decode_dc<do_it>(info, d_out, allocator, stream);
    if (do_it) {
        // data is not in image order
        decode_transpose(info, d_out, d_image_qdct, stream);
        idct(info, d_image_qdct, d_image, d_qtables, stream);
    }

    // data will be planar, may be subsampled, may be RGB, YCbCr, CYMK, anything else
    const bool is_matching_css = false; // TODO find a good way to pass CSS to this function
    if (info.color_fmt != color_fmt || info.pixel_fmt != pixel_fmt || !is_matching_css) {
        if (do_it) {
            convert(
                info.size_x,
                info.size_y,
                jpeggpu::image_desc{
                    d_image[0],
                    info.components[0].data_size_x,
                    d_image[1],
                    info.components[1].data_size_x,
                    d_image[2],
                    info.components[2].data_size_x,
                    d_image[3],
                    info.components[3].data_size_x},
                info.color_fmt,
                info.pixel_fmt,
                {{info.components[0].ss_x,
                  info.components[1].ss_x,
                  info.components[2].ss_x,
                  info.components[3].ss_x},
                 {info.components[0].ss_y,
                  info.components[1].ss_y,
                  info.components[2].ss_y,
                  info.components[3].ss_y}},
                jpeggpu::image_desc{
                    img.image[0],
                    img.pitch[0],
                    img.image[1],
                    img.pitch[1],
                    img.image[2],
                    img.pitch[2],
                    img.image[3],
                    img.pitch[3]},
                color_fmt,
                pixel_fmt,
                subsampling,
                stream);
        }
    }

    return JPEGGPU_SUCCESS;
}

jpeggpu_status jpeggpu::decoder::decode(
    jpeggpu_img& img,
    jpeggpu_color_format color_fmt,
    jpeggpu_pixel_format pixel_fmt,
    jpeggpu_subsampling subsampling,
    void* d_tmp_param,
    size_t& tmp_size_param,
    cudaStream_t stream)
{
    allocator.reset();

    if (!d_tmp_param) {
        JPEGGPU_CHECK_STAT(decode_impl<false>(img, color_fmt, pixel_fmt, subsampling, stream));
        tmp_size_param = allocator.size;
    } else {
        allocator.alloc = {reinterpret_cast<char*>(d_tmp_param), tmp_size_param};
        JPEGGPU_CHECK_STAT(decode_impl<true>(img, color_fmt, pixel_fmt, subsampling, stream));
    }

    return JPEGGPU_SUCCESS;
}
