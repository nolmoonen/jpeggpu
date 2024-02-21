#include "decoder.hpp"
#include "convert.hpp"
#include "decode_gpu.hpp"
#include "defs.hpp"
#include "idct.hpp"
#include "idct_cpu.hpp"
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
        CHECK_CUDA(cudaMalloc(&d_qtables[c], sizeof(uint8_t) * data_unit_size));
    }

    jpeggpu_status stat = JPEGGPU_SUCCESS;
    if ((stat = reader.startup()) != JPEGGPU_SUCCESS) {
        return stat;
    }

    logger.do_logging = false;

    return JPEGGPU_SUCCESS;
}

void jpeggpu::decoder::cleanup()
{
    reader.cleanup();

    for (int c = 0; c < max_comp_count; ++c) {
        cudaFree(d_qtables[c]);
        d_qtables[c] = nullptr;
    }
}

jpeggpu_status jpeggpu::decoder::parse_header(
    jpeggpu_img_info& img_info, const uint8_t* data, size_t size)
{
    reader.reset(data, data + size, &logger);
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
    img_info.subsampling = reader.jpeg_stream.css;

    return JPEGGPU_SUCCESS;
}

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

size_t calculate_gpu_memory(const jpeggpu::reader& reader)
{
    size_t required = 0;
    for (int c = 0; c < reader.jpeg_stream.num_components; ++c) {
        const size_t size = reader.jpeg_stream.data_sizes_x[c] * reader.jpeg_stream.data_sizes_y[c];
        required += jpeggpu::gpu_alloc_size(size * sizeof(int16_t)); // d_image_qdct
        required += jpeggpu::gpu_alloc_size(size * sizeof(uint8_t)); // d_image
    }
    const size_t file_size = reader.image_end - reader.image_begin;
    required += jpeggpu::gpu_alloc_size(file_size); // image data
    required += jpeggpu::gpu_alloc_size(file_size); // destuffed image data
    required += jpeggpu::calculate_gpu_decode_memory(reader);

    return required;
}

#define CHECK_STAT(call)                                                                           \
    do {                                                                                           \
        jpeggpu_status stat = call;                                                                \
        if (stat != JPEGGPU_SUCCESS) {                                                             \
            return JPEGGPU_INTERNAL_ERROR;                                                         \
        }                                                                                          \
    } while (0)

jpeggpu_status jpeggpu::decoder::decode(
    jpeggpu_img& img,
    jpeggpu_color_format color_fmt,
    jpeggpu_pixel_format pixel_fmt,
    jpeggpu_subsampling subsampling,
    void* d_tmp_param,
    size_t& tmp_size_param,
    cudaStream_t stream)
{
    const size_t required_memory =
        calculate_gpu_memory(reader) + calculate_gpu_decode_memory(reader);
    if (d_tmp_param == nullptr) {
        tmp_size_param = required_memory;
        return jpeggpu_status::JPEGGPU_SUCCESS;
    }

    if (tmp_size_param < required_memory) {
        return jpeggpu_status::JPEGGPU_INVALID_ARGUMENT;
    }

    void* d_tmp     = d_tmp_param;
    size_t tmp_size = tmp_size_param;
    for (int c = 0; c < reader.jpeg_stream.num_components; ++c) {
        const size_t size = reader.jpeg_stream.data_sizes_x[c] * reader.jpeg_stream.data_sizes_y[c];
        CHECK_STAT(jpeggpu::gpu_alloc_reserve(
            reinterpret_cast<void**>(&(d_image_qdct[c])), size * sizeof(int16_t), d_tmp, tmp_size));
        CHECK_STAT(jpeggpu::gpu_alloc_reserve(
            reinterpret_cast<void**>(&(d_image[c])), size * sizeof(uint8_t), d_tmp, tmp_size));
    }
    const size_t file_size = reader.image_end - reader.image_begin;
    uint8_t* d_image_data  = nullptr;
    CHECK_STAT(jpeggpu::gpu_alloc_reserve(
        reinterpret_cast<void**>(&d_image_data), file_size, d_tmp, tmp_size));
    uint8_t* d_image_data_destuffed = nullptr;
    CHECK_STAT(jpeggpu::gpu_alloc_reserve(
        reinterpret_cast<void**>(&d_image_data_destuffed), file_size, d_tmp, tmp_size));

    CHECK_CUDA(cudaMemcpyAsync( // TODO put in separate API function
        d_image_data,
        reader.image_begin,
        file_size,
        cudaMemcpyHostToDevice,
        stream));

    if (jpeggpu::decode(
            logger,
            reader,
            d_image_data,
            d_image_data_destuffed,
            d_image_qdct,
            d_tmp,
            tmp_size,
            stream) != JPEGGPU_SUCCESS) {
        return JPEGGPU_INTERNAL_ERROR;
    }

    for (int c = 0; c < reader.jpeg_stream.num_components; ++c) {
        CHECK_CUDA(cudaMemcpyAsync(
            d_qtables[c],
            reader.h_qtables[reader.jpeg_stream.components[c].qtable_idx],
            sizeof(*reader.h_qtables[reader.jpeg_stream.components[c].qtable_idx]),
            cudaMemcpyHostToDevice,
            stream));
    }

    // TODO check that the number of scans seen is equal to the number of components

    // jpeggpu::idct_cpu(reader, d_image_qdct, d_image);
    idct(reader, d_image_qdct, d_image, d_qtables, stream);

    // data will be planar, may be subsampled, may be RGB, YCbCr, CYMK, anything else
    const struct reader::jpeg_stream& info = reader.jpeg_stream;
    if (info.color_fmt != color_fmt || info.pixel_fmt != pixel_fmt || info.css != subsampling) {
        convert(
            info.size_x,
            info.size_y,
            jpeggpu::image_desc{
                d_image[0],
                info.data_sizes_x[0],
                d_image[1],
                info.data_sizes_x[1],
                d_image[2],
                info.data_sizes_x[2],
                d_image[3],
                info.data_sizes_x[3]},
            info.color_fmt,
            info.pixel_fmt,
            info.css,
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

    return JPEGGPU_SUCCESS;
}

#undef CHECK_STAT
