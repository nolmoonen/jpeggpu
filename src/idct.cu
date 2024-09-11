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

// https://developer.download.nvidia.com/assets/cuda/files/dct8x8.pdf

#include "defs.hpp"
#include "idct.hpp"
#include "reader.hpp"
#include "util.hpp"

#include <cuda_runtime.h>

using namespace jpeggpu;

namespace {

/// \brief Number of horizontal data units processed by one kernel block.
constexpr int num_data_units_x_block = 4;
/// \brief Number of vertical data units processed by one kernel block.
constexpr int num_data_units_y_block = 4;
/// \brief Number of horizontal data elements processed by one kernel block.
constexpr int num_data_x_block = num_data_units_x_block * data_unit_vector_size;
/// \brief Number of vertical data elements processed by one kernel block.
constexpr int num_data_y_block = num_data_units_y_block * data_unit_vector_size;

/// \brief Convert fixed-point value to short value.
__device__ inline int16_t unfixh(int x) { return static_cast<int16_t>((x + 0x8000) >> 16); }

/// \brief Convert fixed-point value to short value.
__device__ inline int unfixo(int x) { return (x + 0x1000) >> 13; }

/// \brief In-place IDCT on a vector of eight values.
__device__ void idct_vector(int& v0, int& v1, int& v2, int& v3, int& v4, int& v5, int& v6, int& v7)
{
    // fixed-point representations
    constexpr int cos_1_4   = 0x5a82;
    constexpr int sin_1_8   = 0x30fc;
    constexpr int cos_1_8   = 0x7642;
    constexpr int osin_1_16 = 0x063e;
    constexpr int osin_5_16 = 0x1a9b;
    constexpr int ocos_1_16 = 0x1f63;
    constexpr int ocos_5_16 = 0x11c7;

    int tmp10 = (v0 + v4) * cos_1_4;
    int tmp11 = (v0 - v4) * cos_1_4;
    int tmp12 = v2 * sin_1_8 - v6 * cos_1_8;
    int tmp13 = v6 * sin_1_8 + v2 * cos_1_8;

    int tmp20 = tmp10 + tmp13;
    int tmp21 = tmp11 + tmp12;
    int tmp22 = tmp11 - tmp12;
    int tmp23 = tmp10 - tmp13;

    int tmp30 = unfixo((v3 + v5) * cos_1_4);
    int tmp31 = unfixo((v3 - v5) * cos_1_4);

    v1 <<= 2;
    v7 <<= 2;

    int tmp40 = v1 + tmp30;
    int tmp41 = v7 + tmp31;
    int tmp42 = v1 - tmp30;
    int tmp43 = v7 - tmp31;

    int tmp50 = tmp40 * ocos_1_16 + tmp41 * osin_1_16;
    int tmp51 = tmp40 * osin_1_16 - tmp41 * ocos_1_16;
    int tmp52 = tmp42 * ocos_5_16 + tmp43 * osin_5_16;
    int tmp53 = tmp42 * osin_5_16 - tmp43 * ocos_5_16;

    v0 = unfixh(tmp20 + tmp50);
    v1 = unfixh(tmp21 + tmp53);
    v2 = unfixh(tmp22 + tmp52);
    v3 = unfixh(tmp23 + tmp51);
    v4 = unfixh(tmp23 - tmp51);
    v5 = unfixh(tmp22 - tmp52);
    v6 = unfixh(tmp21 - tmp53);
    v7 = unfixh(tmp20 - tmp50);
}

/// \brief In-place IDCT on a col of eight 16-bits elements.
__device__ void idct_col(int16_t* data, int stride)
{
    int v0 = data[0 * stride];
    int v1 = data[1 * stride];
    int v2 = data[2 * stride];
    int v3 = data[3 * stride];
    int v4 = data[4 * stride];
    int v5 = data[5 * stride];
    int v6 = data[6 * stride];
    int v7 = data[7 * stride];

    idct_vector(v0, v1, v2, v3, v4, v5, v6, v7);

    data[0 * stride] = v0;
    data[1 * stride] = v1;
    data[2 * stride] = v2;
    data[3 * stride] = v3;
    data[4 * stride] = v4;
    data[5 * stride] = v5;
    data[6 * stride] = v6;
    data[7 * stride] = v7;
}

/// \brief In-place IDCT on a row of eight 16-bits elements.
__device__ void idct_row(uint32_t* v8)
{
    uint32_t v01 = v8[0];
    uint32_t v23 = v8[1];
    uint32_t v45 = v8[2];
    uint32_t v67 = v8[3];

    int v0 = static_cast<int16_t>(v01);
    int v1 = static_cast<int16_t>(v01 >> 16);
    int v2 = static_cast<int16_t>(v23);
    int v3 = static_cast<int16_t>(v23 >> 16);
    int v4 = static_cast<int16_t>(v45);
    int v5 = static_cast<int16_t>(v45 >> 16);
    int v6 = static_cast<int16_t>(v67);
    int v7 = static_cast<int16_t>(v67 >> 16);

    idct_vector(v0, v1, v2, v3, v4, v5, v6, v7);

    v8[0] = (static_cast<uint32_t>(v1) << 16) | static_cast<uint16_t>(v0);
    v8[1] = (static_cast<uint32_t>(v3) << 16) | static_cast<uint16_t>(v2);
    v8[2] = (static_cast<uint32_t>(v5) << 16) | static_cast<uint16_t>(v4);
    v8[3] = (static_cast<uint32_t>(v7) << 16) | static_cast<uint16_t>(v6);
}

__global__ void idct_kernel(
    int16_t* image_qdct,
    uint8_t* image,
    int pitch,
    int data_size_x,
    int data_size_y,
    int size_x,
    int size_y,
    qtable* qtable)
{
    constexpr int shared_stride = (num_data_x_block + 2);
    // macroblock data in the image order
    __shared__ int16_t block[shared_stride * num_data_y_block];

    // threadIdx.x is index in four times eights threads, each completing one 8x8 data unit
    // (threadIdx.y, threadIdx.z) is macroblock index
    int shared_x = threadIdx.y * data_unit_vector_size + threadIdx.x;
    int shared_y = threadIdx.z * data_unit_vector_size;

    const int data_unit_x = blockIdx.x * num_data_units_x_block + threadIdx.y;
    const int data_unit_y = blockIdx.y * num_data_units_y_block + threadIdx.z;

    const bool is_inside = data_unit_x * 8 < data_size_x && data_unit_y * 8 < data_size_y;

    // load one 8-wide col of data
    // TODO load two values at once
    if (is_inside) {
        for (int i = 0; i < data_unit_vector_size; ++i) {
            const int off = (data_unit_y * data_unit_vector_size + i) * data_size_x +
                            data_unit_x * data_unit_vector_size + threadIdx.x;
            const int data_idx_in_data_unit = i * data_unit_vector_size + threadIdx.x;
            int16_t* bl_ptr                 = &block[shared_y * shared_stride + shared_x];
            const int16_t val               = image_qdct[off];
            const int8_t qval               = qtable->data[data_idx_in_data_unit];
            bl_ptr[i * shared_stride]       = val * qval;
        }
    }
    __syncthreads();

    if (is_inside) {
        idct_col(block + shared_y * shared_stride + shared_x, shared_stride);
    }
    __syncthreads();

    if (is_inside) {
        idct_row(reinterpret_cast<uint32_t*>(
            &block
                [(shared_y + threadIdx.x) * shared_stride + threadIdx.y * data_unit_vector_size]));
    }
    __syncthreads();

    // store one 8-wide col of data
    if (is_inside) {
        // cannot write multiple bytes at once since it is not guaranteed that the
        //   user-provided pitch is aligned.
        // TODO can check pitch for alignment

        const int off_x = data_unit_x * data_unit_vector_size + threadIdx.x;
        if (off_x >= size_x) {
            return;
        }

        for (int i = 0; i < data_unit_vector_size; ++i) {
            const int off_y = data_unit_y * data_unit_vector_size + i;

            if (off_y >= size_y) {
                return;
            }

            int16_t* bl_ptr = &block[shared_y * shared_stride + shared_x];

            // normalize
            const int16_t val = bl_ptr[i * shared_stride] + 128;
            const int off     = off_y * pitch + off_x;
            image[off]        = max(0, min(val, 255));
        }
    }
}

} // namespace

jpeggpu_status jpeggpu::idct(
    const jpeg_stream& info,
    int16_t* (&d_image_qdct)[max_comp_count],
    uint8_t* (&d_image)[max_comp_count],
    int (&pitch)[max_comp_count],
    qtable* (&d_qtable)[max_comp_count], // TODO can be 16 bit?
    cudaStream_t stream,
    logger& logger)
{
    for (int c = 0; c < info.num_components; ++c) {
        const dim3 num_blocks(
            ceiling_div(
                info.components[c].max_data_size.x, static_cast<unsigned int>(num_data_x_block)),
            ceiling_div(
                info.components[c].max_data_size.y, static_cast<unsigned int>(num_data_y_block)));
        const dim3 kernel_block_size(
            data_unit_vector_size, num_data_units_x_block, num_data_units_y_block);
        idct_kernel<<<num_blocks, kernel_block_size, 0, stream>>>(
            d_image_qdct[c],
            d_image[c],
            pitch[c],
            info.components[c].max_data_size.x,
            info.components[c].max_data_size.y,
            info.components[c].size.x,
            info.components[c].size.y,
            d_qtable[info.components[c].qtable_idx]);
        JPEGGPU_CHECK_CUDA(cudaGetLastError());
    }
    return JPEGGPU_SUCCESS;
}
