// Copyright (c) 2024 Nol Moonen
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "decode_transpose.hpp"
#include "defs.hpp"
#include "logger.hpp"
#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <cuda_runtime.h>

#include <cassert>

using namespace jpeggpu;

namespace {

// Every thread loads and stores one data unit row (8 values) of a total 8 * 2 = 16 bytes.
//   This is because the minimum memory transaction is 32 bytes, and each row must go to a
//   "random" address so coalescing is not possible for storing.
constexpr int num_values_per_thread = 8;

template <int num_data_units_in_mcu>
__global__ void transpose_interleaved(
    const int16_t* __restrict__ data_in,
    int16_t* __restrict__ data_out_0,
    int16_t* __restrict__ data_out_1,
    int16_t* __restrict__ data_out_2,
    int16_t* __restrict__ data_out_3,
    size_t data_size, /// Sum of all pixels in all components.
    /// Pixel size of first component, assumed to be a multiple of MCU size for this component.
    ivec2 size_0,
    ivec2 size_1,
    ivec2 size_2,
    ivec2 size_3,
    int data_size_mcu_x, /// Number of MCUs in a row.
    ivec2 num_blocks_in_mcu_0,
    ivec2 num_blocks_in_mcu_1,
    ivec2 num_blocks_in_mcu_2,
    ivec2 num_blocks_in_mcu_3)
{
    const size_t idx_pixel_in = (blockIdx.x * blockDim.x + threadIdx.x) * num_values_per_thread;

    assert(data_size % num_values_per_thread == 0);
    if (idx_pixel_in >= data_size) return;

    // Variables hold for both pixels.
    const int idx_data_unit = idx_pixel_in / data_unit_size;
    const int idx_mcu       = idx_data_unit / num_data_units_in_mcu;
    const int idx_in_mcu    = idx_data_unit % num_data_units_in_mcu;

    int x_in_mcu = 0;
    int y_in_mcu = 0;
    ivec2 num_blocks_in_mcu{0, 0};
    ivec2 size{0, 0};
    int16_t* data_out = nullptr;
    [&]() {
        int i = 0;

        num_blocks_in_mcu = num_blocks_in_mcu_0;
        data_out          = data_out_0;
        size              = size_0;
        for (y_in_mcu = 0; y_in_mcu < num_blocks_in_mcu_0.y; ++y_in_mcu) {
            for (x_in_mcu = 0; x_in_mcu < num_blocks_in_mcu_0.x; ++x_in_mcu) {
                if (idx_in_mcu == i++) return;
            }
        }
        num_blocks_in_mcu = num_blocks_in_mcu_1;
        data_out          = data_out_1;
        size              = size_1;
        for (y_in_mcu = 0; y_in_mcu < num_blocks_in_mcu_1.y; ++y_in_mcu) {
            for (x_in_mcu = 0; x_in_mcu < num_blocks_in_mcu_1.x; ++x_in_mcu) {
                if (idx_in_mcu == i++) return;
            }
        }
        num_blocks_in_mcu = num_blocks_in_mcu_2;
        data_out          = data_out_2;
        size              = size_2;
        for (y_in_mcu = 0; y_in_mcu < num_blocks_in_mcu_2.y; ++y_in_mcu) {
            for (x_in_mcu = 0; x_in_mcu < num_blocks_in_mcu_2.x; ++x_in_mcu) {
                if (idx_in_mcu == i++) return;
            }
        }
        num_blocks_in_mcu = num_blocks_in_mcu_3;
        data_out          = data_out_3;
        size              = size_3;
        for (y_in_mcu = 0; y_in_mcu < num_blocks_in_mcu_3.y; ++y_in_mcu) {
            for (x_in_mcu = 0; x_in_mcu < num_blocks_in_mcu_3.x; ++x_in_mcu) {
                if (idx_in_mcu == i++) return;
            }
        }
        assert(false);
        __builtin_unreachable();
    }();

    const int idx_in_data_unit = idx_pixel_in % data_unit_size;

    const int x_mcu = idx_mcu % data_size_mcu_x;
    const int y_mcu = idx_mcu / data_size_mcu_x;

    const int x_data_unit = x_mcu * num_blocks_in_mcu.x + x_in_mcu;
    const int y_data_unit = y_mcu * num_blocks_in_mcu.y + y_in_mcu;

    const int x_in_data_unit = idx_in_data_unit % data_unit_vector_size;
    const int y_in_data_unit = idx_in_data_unit / data_unit_vector_size;

    const int x = x_data_unit * data_unit_vector_size + x_in_data_unit;
    const int y = y_data_unit * data_unit_vector_size + y_in_data_unit;

    using load_type = uint4;
    static_assert(sizeof(load_type) == num_values_per_thread * sizeof(uint16_t));
    const load_type val = *reinterpret_cast<const load_type*>(data_in + idx_pixel_in);
    *reinterpret_cast<load_type*>(data_out + y * size.x + x) = val;
}

} // namespace

jpeggpu_status jpeggpu::decode_transpose(
    const jpeg_stream& info,
    const int16_t* d_out,
    const scan& scan,
    int16_t* (&d_image_qdct)[jpeggpu::max_comp_count],
    cudaStream_t stream,
    logger& logger)
{
    const scan_component(&scan_components)[max_comp_count] = scan.scan_components;
    const int num_scan_comp                                = scan.num_scan_components;

    const size_t total_data_size =
        (num_scan_comp > 0 ? scan_components[0].data_size.x * scan_components[0].data_size.y : 0) +
        (num_scan_comp > 1 ? scan_components[1].data_size.x * scan_components[1].data_size.y : 0) +
        (num_scan_comp > 2 ? scan_components[2].data_size.x * scan_components[2].data_size.y : 0) +
        (num_scan_comp > 3 ? scan_components[3].data_size.x * scan_components[3].data_size.y : 0);

    const dim3 transpose_block_dim(256);
    const dim3 transpose_grid_dim(ceiling_div(
        total_data_size / num_values_per_thread, static_cast<unsigned int>(transpose_block_dim.x)));

    const int comp_idx_0 = scan_components[0].component_idx;
    const int comp_idx_1 = scan_components[1].component_idx;
    const int comp_idx_2 = scan_components[2].component_idx;
    const int comp_idx_3 = scan_components[3].component_idx;

// Provide the num_data_units_in_mcu as a compile-time constant so the compiler can generate efficient
//   division and modulo instructions.
#define DISPATCH(NUM_DATA_UNITS_IN_MCU)                                              \
    transpose_interleaved<NUM_DATA_UNITS_IN_MCU>                                     \
        <<<transpose_grid_dim, transpose_block_dim, 0, stream>>>(                    \
            d_out,                                                                   \
            num_scan_comp > 0 ? d_image_qdct[comp_idx_0] : nullptr,                  \
            num_scan_comp > 1 ? d_image_qdct[comp_idx_1] : nullptr,                  \
            num_scan_comp > 2 ? d_image_qdct[comp_idx_2] : nullptr,                  \
            num_scan_comp > 3 ? d_image_qdct[comp_idx_3] : nullptr,                  \
            total_data_size,                                                         \
            num_scan_comp > 0 ? scan_components[0].data_size : ivec2{0, 0},          \
            num_scan_comp > 1 ? scan_components[1].data_size : ivec2{0, 0},          \
            num_scan_comp > 2 ? scan_components[2].data_size : ivec2{0, 0},          \
            num_scan_comp > 3 ? scan_components[3].data_size : ivec2{0, 0},          \
            scan.num_mcus.x,                                                         \
            num_scan_comp > 0 ? scan_components[0].num_blocks_in_mcu : ivec2{0, 0},  \
            num_scan_comp > 1 ? scan_components[1].num_blocks_in_mcu : ivec2{0, 0},  \
            num_scan_comp > 2 ? scan_components[2].num_blocks_in_mcu : ivec2{0, 0},  \
            num_scan_comp > 3 ? scan_components[3].num_blocks_in_mcu : ivec2{0, 0}); \
    JPEGGPU_CHECK_CUDA(cudaGetLastError());                                          \
    return JPEGGPU_SUCCESS;

    JPEGGPU_CHECK_STAT([&]() -> jpeggpu_status {
        // Specification states this should be in [1, 10], see SOS read function.
        switch (scan.num_data_units_in_mcu) {
        case 1:
            DISPATCH(1);
        case 2:
            DISPATCH(2);
        case 3:
            DISPATCH(3);
        case 4:
            DISPATCH(4);
        case 5:
            DISPATCH(5);
        case 6:
            DISPATCH(6);
        case 7:
            DISPATCH(7);
        case 8:
            DISPATCH(8);
        case 9:
            DISPATCH(9);
        case 10:
            DISPATCH(10);
        }
        return JPEGGPU_INTERNAL_ERROR;
    }());
#undef DISPATCH

    return JPEGGPU_SUCCESS;
}
