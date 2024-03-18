#include "decode_transpose.hpp"
#include "defs.hpp"
#include "logger.hpp"
#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <cuda_runtime.h>

#include <cassert>

using namespace jpeggpu;

namespace {

// TODO first version, optimize
__global__ void transpose_interleaved(
    const int16_t* data_in,
    int16_t* data_out_0,
    int16_t* data_out_1,
    int16_t* data_out_2,
    int16_t* data_out_3,
    size_t data_size, /// Sum of all pixels in all components.
    int2
        size_0, /// Pixel size of first component, assumed to be a multiple of MCU size for this component.
    int2 size_1,
    int2 size_2,
    int2 size_3,
    int data_units_in_mcu, /// Number of data units in a MCU.
    int data_size_mcu_x, /// Number of MCUs in a row.
    int2 ss_0, /// Subsampling factor of first component, as defined in JPEG header.
    int2 ss_1,
    int2 ss_2,
    int2 ss_3)
{
    const size_t idx_pixel_in = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx_pixel_in >= data_size) return;

    const int idx_data_unit    = idx_pixel_in / data_unit_size;
    const int idx_in_data_unit = idx_pixel_in % data_unit_size;

    const int idx_mcu    = idx_data_unit / data_units_in_mcu;
    const int idx_in_mcu = idx_data_unit % data_units_in_mcu;

    int x_in_mcu      = 0;
    int y_in_mcu      = 0;
    int2 ss           = make_int2(0, 0);
    int2 size         = make_int2(0, 0);
    int16_t* data_out = nullptr;
    [&]() {
        int i = 0;

        ss       = ss_0;
        data_out = data_out_0;
        size     = size_0;
        for (y_in_mcu = 0; y_in_mcu < ss_0.y; ++y_in_mcu) {
            for (x_in_mcu = 0; x_in_mcu < ss_0.x; ++x_in_mcu) {
                if (idx_in_mcu == i++) return;
            }
        }
        ss       = ss_1;
        data_out = data_out_1;
        size     = size_1;
        for (y_in_mcu = 0; y_in_mcu < ss_1.y; ++y_in_mcu) {
            for (x_in_mcu = 0; x_in_mcu < ss_1.x; ++x_in_mcu) {
                if (idx_in_mcu == i++) return;
            }
        }
        ss       = ss_2;
        data_out = data_out_2;
        size     = size_2;
        for (y_in_mcu = 0; y_in_mcu < ss_2.y; ++y_in_mcu) {
            for (x_in_mcu = 0; x_in_mcu < ss_2.x; ++x_in_mcu) {
                if (idx_in_mcu == i++) return;
            }
        }
        ss       = ss_3;
        data_out = data_out_3;
        size     = size_3;
        for (y_in_mcu = 0; y_in_mcu < ss_3.y; ++y_in_mcu) {
            for (x_in_mcu = 0; x_in_mcu < ss_3.x; ++x_in_mcu) {
                if (idx_in_mcu == i++) return;
            }
        }
        assert(false);
        __builtin_unreachable();
    }();

    const uint16_t val = data_in[idx_pixel_in];

    const int x_mcu = idx_mcu % data_size_mcu_x;
    const int y_mcu = idx_mcu / data_size_mcu_x;

    const int x_data_unit = x_mcu * ss.x + x_in_mcu;
    const int y_data_unit = y_mcu * ss.y + y_in_mcu;

    const int x_in_data_unit = idx_in_data_unit % data_unit_vector_size;
    const int y_in_data_unit = idx_in_data_unit / data_unit_vector_size;

    const int x = x_data_unit * data_unit_vector_size + x_in_data_unit;
    const int y = y_data_unit * data_unit_vector_size + y_in_data_unit;

    const int idx_pixel_out = y * size.x + x;
    data_out[idx_pixel_out] = val;
}

// TODO first version, optimize
__global__ void transpose_non_interleaved(
    const int16_t* data_in,
    int16_t* data_out_0,
    int16_t* data_out_1,
    int16_t* data_out_2,
    int16_t* data_out_3,
    size_t data_size,
    int2 size_0,
    int2 size_1,
    int2 size_2,
    int2 size_3)
{
    const int idx_pixel_in = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx_pixel_in >= data_size) return;

    const int idx_data_unit    = idx_pixel_in / data_unit_size;
    const int idx_in_data_unit = idx_pixel_in % data_unit_size;

    int16_t* data_out = nullptr;
    int x_data_unit   = 0;
    int y_data_unit   = 0;
    int2 size         = make_int2(0, 0);
    [&]() {
        int i = idx_data_unit;

        int size_0_flat = size_0.x * size_0.y;
        if (i < size_0_flat) {
            data_out    = data_out_0;
            x_data_unit = i % size_0.x;
            y_data_unit = i / size_0.x;
            size        = size_0;
            return;
        }
        i -= size_0_flat;

        int size_1_flat = size_1.x * size_1.y;
        if (i < size_1_flat) {
            data_out    = data_out_1;
            x_data_unit = i % size_1.x;
            y_data_unit = i / size_1.x;
            size        = size_1;
            return;
        }
        i -= size_1_flat;

        int size_2_flat = size_2.x * size_2.y;
        if (i < size_2_flat) {
            data_out    = data_out_2;
            x_data_unit = i % size_2.x;
            y_data_unit = i / size_2.x;
            size        = size_2;
            return;
        }
        i -= size_2_flat;

        int size_3_flat = size_3.x * size_3.y;
        if (i < size_3_flat) {
            data_out    = data_out_3;
            x_data_unit = i % size_3.x;
            y_data_unit = i / size_3.x;
            size        = size_3;
            return;
        }
        i -= size_3_flat;
        assert(false);
        __builtin_unreachable();
    }();

    const uint16_t val = data_in[idx_pixel_in];

    const int x_in_data_unit = idx_in_data_unit % data_unit_vector_size;
    const int y_in_data_unit = idx_in_data_unit / data_unit_vector_size;

    const int x = x_data_unit * data_unit_vector_size + x_in_data_unit;
    const int y = y_data_unit * data_unit_vector_size + y_in_data_unit;

    const int idx_pixel_out = y * size.x + x;
    data_out[idx_pixel_out] = val;
}

} // namespace

jpeggpu_status jpeggpu::decode_transpose(
    const jpeg_stream& info,
    const int16_t* d_out,
    int16_t* (&d_image_qdct)[jpeggpu::max_comp_count],
    cudaStream_t stream)
{
    const size_t total_data_size = info.components[0].data_size_x * info.components[0].data_size_y +
                                   info.components[1].data_size_x * info.components[1].data_size_y +
                                   info.components[2].data_size_x * info.components[2].data_size_y +
                                   info.components[3].data_size_x * info.components[3].data_size_y;
    const dim3 transpose_block_dim(256);
    const dim3 transpose_grid_dim(
        ceiling_div(total_data_size, static_cast<unsigned int>(transpose_block_dim.x)));

    if (info.is_interleaved) {
        const int data_units_in_mcu = info.components[0].ss_x * info.components[0].ss_y +
                                      info.components[1].ss_x * info.components[1].ss_y +
                                      info.components[2].ss_x * info.components[2].ss_y +
                                      info.components[3].ss_x * info.components[3].ss_y;
        transpose_interleaved<<<transpose_grid_dim, transpose_block_dim, 0, stream>>>(
            d_out,
            d_image_qdct[0],
            d_image_qdct[1],
            d_image_qdct[2],
            d_image_qdct[3],
            total_data_size,
            make_int2(info.components[0].data_size_x, info.components[0].data_size_y),
            make_int2(info.components[1].data_size_x, info.components[1].data_size_y),
            make_int2(info.components[2].data_size_x, info.components[2].data_size_y),
            make_int2(info.components[3].data_size_x, info.components[3].data_size_y),
            data_units_in_mcu,
            info.num_mcus_x,
            make_int2(info.components[0].ss_x, info.components[0].ss_y),
            make_int2(info.components[1].ss_x, info.components[1].ss_y),
            make_int2(info.components[2].ss_x, info.components[2].ss_y),
            make_int2(info.components[3].ss_x, info.components[3].ss_y));
        JPEGGPU_CHECK_CUDA(cudaGetLastError());
    } else {
        transpose_non_interleaved<<<transpose_grid_dim, transpose_block_dim, 0, stream>>>(
            d_out,
            d_image_qdct[0],
            d_image_qdct[1],
            d_image_qdct[2],
            d_image_qdct[3],
            total_data_size,
            make_int2(info.components[0].data_size_x, info.components[0].data_size_y),
            make_int2(info.components[1].data_size_x, info.components[1].data_size_y),
            make_int2(info.components[2].data_size_x, info.components[2].data_size_y),
            make_int2(info.components[3].data_size_x, info.components[3].data_size_y));
        JPEGGPU_CHECK_CUDA(cudaGetLastError());
    }

    return JPEGGPU_SUCCESS;
}
