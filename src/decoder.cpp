#include "decoder.hpp"
#include "convert.hpp"
#include "decode_cpu_legacy.hpp"
#include "decode_gpu.hpp"
#include "defs.hpp"
#include "idct.hpp"
#include "marker.hpp"
#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <array>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <type_traits>
#include <vector>

jpeggpu_status jpeggpu::decoder::parse_header(
    jpeggpu_img_info& img_info, const uint8_t* data, size_t size)
{
    reader.reset(data, data + size);
    jpeggpu_status stat = reader.read();
    if (stat != JPEGGPU_SUCCESS) {
        return stat;
    }

    // TODO check reader consistency

    img_info.size_x = reader.size_x;
    img_info.size_y = reader.size_y;
    // TODO read metadata to determine color formats
    switch (reader.num_components) {
    case 1:
        reader.color_fmt = JPEGGPU_GRAY;
        reader.pixel_fmt = JPEGGPU_P0;
        break;
    case 3:
        reader.color_fmt = JPEGGPU_YCBCR;
        reader.pixel_fmt = JPEGGPU_P0P1P2;
        break;
    case 4:
        reader.color_fmt = JPEGGPU_CMYK;
        reader.pixel_fmt = JPEGGPU_P0P1P2P3;
        break;
    default:
        return JPEGGPU_NOT_SUPPORTED;
    }
    img_info.subsampling = reader.css;

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

jpeggpu_status jpeggpu::decoder::decode(
    jpeggpu_img& img,
    jpeggpu_color_format color_fmt,
    jpeggpu_pixel_format pixel_fmt,
    jpeggpu_subsampling subsampling,
    void* d_tmp,
    size_t& tmp_size)
{
    if (d_tmp == nullptr) {
        tmp_size = size_t{20} << 20; // FIXME
        return jpeggpu_status::JPEGGPU_SUCCESS;
    }

    for (int c = 0; c < reader.num_components; ++c) {
        const size_t size = reader.data_sizes_x[c] * reader.data_sizes_y[c];
        CHECK_CUDA(cudaMalloc(&(d_image_qdct[c]), size * sizeof(uint16_t)));
        CHECK_CUDA(cudaMalloc(&(d_image[c]), size));
    }

    if (is_gpu_decode_possible(reader)) {
        process_scan(reader, d_image_qdct, cudaStreamDefault); // TODO stream
    } else {
        DBG_PRINT("falling back to CPU decode\n");
        process_scan_legacy(reader, d_image_qdct);
    }

    // TODO check that the number of scans seen is equal to the number of components
    jpeggpu::idct(reader, d_image_qdct, d_image);

    // data will be planar, may be subsampled, may be RGB, YCbCr, CYMK, anything else
    if (reader.color_fmt != color_fmt || reader.pixel_fmt != pixel_fmt ||
        reader.css != subsampling) {
        jpeggpu::convert(
            reader.size_x,
            reader.size_y,
            jpeggpu::image_desc{
                d_image[0],
                reader.data_sizes_x[0],
                d_image[1],
                reader.data_sizes_x[1],
                d_image[2],
                reader.data_sizes_x[2],
                d_image[3],
                reader.data_sizes_x[3]},
            reader.color_fmt,
            reader.pixel_fmt,
            reader.css,
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
            cudaStreamDefault); // TODO stream
    }

    return JPEGGPU_SUCCESS;
}
