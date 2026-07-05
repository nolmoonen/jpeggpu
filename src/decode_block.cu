// Copyright (c) 2026 Nol Moonen
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

#include "decode_block.hpp"
#include "defs.hpp"
#include "huffman.cuh"
#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <cuda/std/utility>

using namespace jpeggpu;

namespace {

// TODO tune?
constexpr int thread_block_size = 256;

struct bit_reader {
    __device__ bit_reader(const uint8_t* data, int bit_off) : data(data)
    {
        buffer             = 0;
        num_bits_in_buffer = 0;
        read_bits(bit_off);
    }

    // TODO note: no need to provide zero
    // Read bytes until a marker is hit, at which point only zero is provided.
    __device__ uint8_t get_next_u8() { return *(data++); }

    __device__ void fill_bit_window()
    {
        if (num_bits_in_buffer <= 32) {
            while (num_bits_in_buffer <= 56) {
                const uint8_t byte = get_next_u8();

                buffer <<= 8;
                buffer |= byte;

                num_bits_in_buffer += 8;
            }
        }
    }

    __device__ int read_bits(int num_bits)
    {
        fill_bit_window();
        assert(num_bits_in_buffer >= num_bits);
        const uint64_t val =
            (buffer >> (num_bits_in_buffer - num_bits)) & ((uint64_t{1} << num_bits) - 1);
        num_bits_in_buffer -= num_bits;
        return val;
    }

    __device__ int peek_bits(int num_bits)
    {
        assert(num_bits <= num_bits_in_buffer);
        return (buffer >> (num_bits_in_buffer - num_bits)) & ((uint64_t{1} << num_bits) - 1);
    }

    const uint8_t* data;

    uint64_t buffer;
    int num_bits_in_buffer;
};

__device__ int huff_extend(int x, int s)
{
    // Table F.1 and Table F.2
    assert(s >= 1);
    int half = 1 << (s - 1);
    if (x >= half) {
        assert(x < (1 << s));
        return x;
    } else {
        return x - (1 << s) + 1;
    }
}

__device__ uint8_t* get_block_ptr_comp(
    uint8_t* coeffs,
    int pitch,
    int num_blocks_in_mcu_x,
    int num_blocks_in_mcu_y,
    int mcu_x,
    int mcu_y,
    int i_in_mcu,
    int num_blocks_x)
{
    const int y_in_mcu = i_in_mcu / num_blocks_in_mcu_x;
    const int x_in_mcu = i_in_mcu % num_blocks_in_mcu_x;

    const int block_y = mcu_y * num_blocks_in_mcu_y + y_in_mcu;
    const int block_x = mcu_x * num_blocks_in_mcu_x + x_in_mcu;
    return coeffs + block_y * data_unit_vector_size * pitch + block_x * data_unit_vector_size;
}

// High-Efficiency and Low-Power Architectures for 2-D DCT and IDCT Based on CORDIC Rotation, Sung et al. 2006
__device__ void idct_vector(float vals[8])
{
    constexpr float sung_a = 1.3870398453221475; // print(f'{math.sqrt(2)*math.cos(1*math.pi/16)}')
    constexpr float sung_b = 1.3065629648763766; // print(f'{math.sqrt(2)*math.cos(2*math.pi/16)}')
    constexpr float sung_c = 1.1758756024193588; // print(f'{math.sqrt(2)*math.cos(3*math.pi/16)}')
    constexpr float sung_d = 0.7856949583871023; // print(f'{math.sqrt(2)*math.cos(5*math.pi/16)}')
    constexpr float sung_e = 0.5411961001461971; // print(f'{math.sqrt(2)*math.cos(6*math.pi/16)}')
    constexpr float sung_f = 0.2758993792829431; // print(f'{math.sqrt(2)*math.cos(7*math.pi/16)}')
    constexpr float sung_t = 0.3535533905932737; // print(f'{1/math.sqrt(8)}')

    const float y0 = vals[0];
    const float y1 = vals[1];
    const float y2 = vals[2];
    const float y3 = vals[3];
    const float y4 = vals[4];
    const float y5 = vals[5];
    const float y6 = vals[6];
    const float y7 = vals[7];

    const float yg = (y0 + y4) + (y2 * sung_b + y6 * sung_e);
    const float yh = y7 * sung_f + y1 * sung_a + y3 * sung_c + y5 * sung_d;
    const float yi = (y0 + y4) - (y2 * sung_b + y6 * sung_e);
    const float yj = y7 * sung_a - y1 * sung_f + y3 * sung_d - y5 * sung_c;

    const float yk = (y0 - y4) + (y2 * sung_e - y6 * sung_b);
    const float yl = y1 * sung_c - y7 * sung_d - y3 * sung_f - y5 * sung_a;
    const float ym = (y0 - y4) - (y2 * sung_e - y6 * sung_b);
    const float yn = y1 * sung_d + y7 * sung_c - y3 * sung_a + y5 * sung_f;

    vals[0] = sung_t * (yg + yh);
    vals[7] = sung_t * (yg - yh);
    vals[4] = sung_t * (yi + yj);
    vals[3] = sung_t * (yi - yj);

    vals[1] = sung_t * (yk + yl);
    vals[5] = sung_t * (ym - yn);
    vals[2] = sung_t * (ym + yn);
    vals[6] = sung_t * (yk - yl);
}

__device__ float clampf(float x, float min, float max) { return fmaxf(min, fminf(x, max)); }

__launch_bounds__(thread_block_size) __global__ void decode_sequential(
    int num_blocks_x_0,
    int num_blocks_x_1,
    int num_blocks_x_2,
    int num_blocks_x_3,
    uint8_t* pixels_0,
    uint8_t* pixels_1,
    uint8_t* pixels_2,
    uint8_t* pixels_3,
    int pitch_0,
    int pitch_1,
    int pitch_2,
    int pitch_3,
    int num_blocks_in_mcu_0,
    int num_blocks_in_mcu_1,
    int num_blocks_in_mcu_2,
    int num_blocks_in_mcu_3,
    int num_blocks_in_mcu_x_0,
    int num_blocks_in_mcu_x_1,
    int num_blocks_in_mcu_x_2,
    int num_blocks_in_mcu_x_3,
    int num_blocks_in_mcu_y_0,
    int num_blocks_in_mcu_y_1,
    int num_blocks_in_mcu_y_2,
    int num_blocks_in_mcu_y_3,
    int num_blocks_in_mcu,
    int num_mcus_x,
    int num_blocks,
    const uint8_t* data,
    const huffman_table* huffman_tables,
    int dc_0, /// DC Huffman table index for scan component 0.
    int ac_0, /// AC Huffman table index for scan component 0.
    int dc_1, /// DC Huffman table index for scan component 1.
    int ac_1, /// AC Huffman table index for scan component 1.
    int dc_2, /// DC Huffman table index for scan component 2.
    int ac_2, /// AC Huffman table index for scan component 2.
    int dc_3, /// DC Huffman table index for scan component 3.
    int ac_3, /// AC Huffman table index for scan component 3.
    const int16_t* dcs,
    const int* bit_offsets,
    qtable* qtable_0,
    qtable* qtable_1,
    qtable* qtable_2,
    qtable* qtable_3)
{
    const int block_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_i >= num_blocks) {
        return;
    }

    const int i_of_mcu = block_i / num_blocks_in_mcu;

    const int mcu_y = i_of_mcu / num_mcus_x;
    const int mcu_x = i_of_mcu % num_mcus_x;

    int i_in_mcu_i = 0;
    int dc_idx     = 0;
    int ac_idx     = 0;
    qtable* qtable = nullptr;
    int pitch      = 0;
    uint8_t* block = nullptr;

    const int i_in_mcu        = block_i % num_blocks_in_mcu;
    int num_blocks_in_mcu_sum = 0;
    if (i_in_mcu < (num_blocks_in_mcu_sum += num_blocks_in_mcu_0)) {
        i_in_mcu_i = i_in_mcu - num_blocks_in_mcu_sum + num_blocks_in_mcu_0;
        dc_idx     = dc_0;
        ac_idx     = ac_0;
        qtable     = qtable_0;
        pitch      = pitch_0;
        block      = get_block_ptr_comp(
            pixels_0,
            pitch_0,
            num_blocks_in_mcu_x_0,
            num_blocks_in_mcu_y_0,
            mcu_x,
            mcu_y,
            i_in_mcu_i,
            num_blocks_x_0);
    } else if (i_in_mcu < (num_blocks_in_mcu_sum += num_blocks_in_mcu_1)) {
        i_in_mcu_i = i_in_mcu - num_blocks_in_mcu_sum + num_blocks_in_mcu_1;
        dc_idx     = dc_1;
        ac_idx     = ac_1;
        qtable     = qtable_1;
        pitch      = pitch_1;
        block      = get_block_ptr_comp(
            pixels_1,
            pitch_1,
            num_blocks_in_mcu_x_1,
            num_blocks_in_mcu_y_1,
            mcu_x,
            mcu_y,
            i_in_mcu_i,
            num_blocks_x_1);
    } else if (i_in_mcu < (num_blocks_in_mcu_sum += num_blocks_in_mcu_2)) {
        i_in_mcu_i = i_in_mcu - num_blocks_in_mcu_sum + num_blocks_in_mcu_2;
        dc_idx     = dc_2;
        ac_idx     = ac_2;
        qtable     = qtable_2;
        pitch      = pitch_2;
        block      = get_block_ptr_comp(
            pixels_2,
            pitch_2,
            num_blocks_in_mcu_x_2,
            num_blocks_in_mcu_y_2,
            mcu_x,
            mcu_y,
            i_in_mcu_i,
            num_blocks_x_2);
    } else { // i_in_mcu < (num_blocks_in_mcu_sum += num_blocks_in_mcu_3)
        i_in_mcu_i = i_in_mcu - num_blocks_in_mcu_sum + num_blocks_in_mcu_3;
        dc_idx     = dc_3;
        ac_idx     = ac_3;
        qtable     = qtable_3;
        pitch      = pitch_3;
        block      = get_block_ptr_comp(
            pixels_3,
            pitch_3,
            num_blocks_in_mcu_x_3,
            num_blocks_in_mcu_y_3,
            mcu_x,
            mcu_y,
            i_in_mcu_i,
            num_blocks_x_3);
    }

    const huffman_table& huff_dc = huffman_tables[dc_idx];
    const huffman_table& huff_ac = huffman_tables[ac_idx];

    float coeffs[data_unit_size] = {0};

    const int begin_bit  = bit_offsets[block_i];
    const int begin_byte = begin_bit / 8;

    bit_reader br(data + begin_byte, begin_bit % 8);

    br.fill_bit_window();
    uint32_t u32 = br.peek_bits(32);

    // skip dc, already read
    int category_length = 0;
    int s               = get_category(u32, category_length, huff_dc);
    br.read_bits(category_length);
    if (s > 0) {
        br.read_bits(s);
    }
    coeffs[0] = dcs[block_i] * qtable->data[0];

    for (int k = 1; k <= 63; k++) {
        br.fill_bit_window();
        u32 = br.peek_bits(32);

        int sr = get_category(u32, category_length, huff_ac);
        br.read_bits(category_length);
        int r = sr >> 4;
        int s = sr & 15;
        if (s > 0) {
            k += r;

            int bits  = br.read_bits(s);
            int coeff = huff_extend(bits, s);

            // FIXME qtable in which order?
            // coeffs[order_natural[k]] = coeff * qtable->data[k];
            coeffs[order_natural[k]] = coeff * qtable->data[order_natural[k]];
        } else if (r == 15) {
            k += 15;
            // TODO is multiple 15 zero allowed instead of EOB?
        } else {
            break;
        }
    }

    for (int r = 0; r < data_unit_vector_size; ++r) {
        idct_vector(coeffs + r * data_unit_vector_size);
    }

    for (int r = 0; r < data_unit_vector_size; ++r) {
        for (int c = r + 1; c < data_unit_vector_size; ++c) {
            cuda::std::swap(
                coeffs[r * data_unit_vector_size + c], coeffs[c * data_unit_vector_size + r]);
        }
    }

    for (int r = 0; r < data_unit_vector_size; ++r) {
        idct_vector(coeffs + r * data_unit_vector_size);
    }

    for (int r = 0; r < data_unit_vector_size; ++r) {
        for (int c = r + 1; c < data_unit_vector_size; ++c) {
            cuda::std::swap(
                coeffs[r * data_unit_vector_size + c], coeffs[c * data_unit_vector_size + r]);
        }
    }

    for (int r = 0; r < data_unit_vector_size; ++r) {
        for (int c = 0; c < data_unit_vector_size; ++c) {
            const uint8_t val =
                clampf(128.f + std::roundf(coeffs[r * data_unit_vector_size + c]), 0.f, 255.f);

            block[r * pitch + c] = val;
        }
    }
}

} // namespace

template <bool do_it>
jpeggpu_status jpeggpu::decode_block(
    const jpeg_stream& info,
    const uint8_t* d_scan_destuffed,
    int16_t* d_dcs,
    int* d_block_bit_offsets,
    const struct scan& scan,
    huffman_table* d_huff_tables,
    uint8_t* (&d_image)[max_comp_count],
    int (&pitch)[max_comp_count],
    qtable* (&d_qtable)[max_comp_count],
    stack_allocator& allocator,
    cudaStream_t stream,
    logger& logger)
{
    const int num_comps                                    = info.num_components;
    const scan_component(&scan_components)[max_comp_count] = scan.scan_components;
    const int num_scan_comp                                = scan.num_scan_components;
    auto& comps                                            = info.components;
    auto& scan_comps                                       = scan.scan_components;

    const int comp_idx_0 = scan_components[0].component_idx;
    const int comp_idx_1 = scan_components[1].component_idx;
    const int comp_idx_2 = scan_components[2].component_idx;
    const int comp_idx_3 = scan_components[3].component_idx;

    int num_scan_blocks = 0;
    for (int sc = 0; sc < num_scan_comp; ++sc) {
        const ivec2& num_blocks_comp =
            info.components[scan.scan_components[sc].component_idx].num_blocks;
        num_scan_blocks += num_blocks_comp.y * num_blocks_comp.x;
    }

    if (do_it) {
        // pitch is only set when do_it
        for (int sc = 0; sc < num_scan_comp; ++sc) {
            const scan_component& scan_comp = scan.scan_components[sc];
            if (pitch[scan_comp.component_idx] % 8 != 0) {
                // TODO relax constraint?
                logger.log("pitch not multiple of eight");
                return JPEGGPU_INVALID_ARGUMENT;
            }
        }
    }

    if (do_it) {
        const int num_thread_blocks =
            ceiling_div(num_scan_blocks, static_cast<unsigned int>(thread_block_size));
        decode_sequential<<<num_thread_blocks, thread_block_size, 0, stream>>>(
            num_comps > 0 ? comps[comp_idx_0].num_blocks.x : 0,
            num_comps > 1 ? comps[comp_idx_1].num_blocks.x : 0,
            num_comps > 2 ? comps[comp_idx_2].num_blocks.x : 0,
            num_comps > 3 ? comps[comp_idx_3].num_blocks.x : 0,
            num_scan_comp > 0 ? d_image[comp_idx_0] : nullptr,
            num_scan_comp > 1 ? d_image[comp_idx_1] : nullptr,
            num_scan_comp > 2 ? d_image[comp_idx_2] : nullptr,
            num_scan_comp > 3 ? d_image[comp_idx_3] : nullptr,
            num_scan_comp > 0 ? pitch[comp_idx_0] : 0,
            num_scan_comp > 1 ? pitch[comp_idx_1] : 0,
            num_scan_comp > 2 ? pitch[comp_idx_2] : 0,
            num_scan_comp > 3 ? pitch[comp_idx_3] : 0,
            num_comps > 0 ? scan_comps[0].num_blocks_in_mcu.x * scan_comps[0].num_blocks_in_mcu.y
                          : 0,
            num_comps > 1 ? scan_comps[1].num_blocks_in_mcu.x * scan_comps[1].num_blocks_in_mcu.y
                          : 0,
            num_comps > 2 ? scan_comps[2].num_blocks_in_mcu.x * scan_comps[2].num_blocks_in_mcu.y
                          : 0,
            num_comps > 3 ? scan_comps[3].num_blocks_in_mcu.x * scan_comps[3].num_blocks_in_mcu.y
                          : 0,
            num_comps > 0 ? scan_comps[0].num_blocks_in_mcu.x : 0,
            num_comps > 1 ? scan_comps[1].num_blocks_in_mcu.x : 0,
            num_comps > 2 ? scan_comps[2].num_blocks_in_mcu.x : 0,
            num_comps > 3 ? scan_comps[3].num_blocks_in_mcu.x : 0,
            num_comps > 0 ? scan_comps[0].num_blocks_in_mcu.y : 0,
            num_comps > 1 ? scan_comps[1].num_blocks_in_mcu.y : 0,
            num_comps > 2 ? scan_comps[2].num_blocks_in_mcu.y : 0,
            num_comps > 3 ? scan_comps[3].num_blocks_in_mcu.y : 0,
            scan.num_data_units_in_mcu,
            scan.num_mcus.x,
            num_scan_blocks,
            d_scan_destuffed,
            d_huff_tables,
            HUFF_COUNT * scan.scan_components[0].dc_idx + HUFF_DC,
            HUFF_COUNT * scan.scan_components[0].ac_idx + HUFF_AC,
            HUFF_COUNT * scan.scan_components[1].dc_idx + HUFF_DC,
            HUFF_COUNT * scan.scan_components[1].ac_idx + HUFF_AC,
            HUFF_COUNT * scan.scan_components[2].dc_idx + HUFF_DC,
            HUFF_COUNT * scan.scan_components[2].ac_idx + HUFF_AC,
            HUFF_COUNT * scan.scan_components[3].dc_idx + HUFF_DC,
            HUFF_COUNT * scan.scan_components[3].ac_idx + HUFF_AC,
            d_dcs,
            d_block_bit_offsets,
            num_scan_comp > 0 ? d_qtable[comp_idx_0] : nullptr,
            num_scan_comp > 1 ? d_qtable[comp_idx_1] : nullptr,
            num_scan_comp > 2 ? d_qtable[comp_idx_2] : nullptr,
            num_scan_comp > 3 ? d_qtable[comp_idx_3] : nullptr);
        JPEGGPU_CHECK_CUDA(cudaGetLastError());
    }

    return JPEGGPU_SUCCESS;
}

template jpeggpu_status jpeggpu::decode_block<false>(
    const jpeg_stream&,
    const uint8_t*,
    int16_t*,
    int*,
    const struct scan&,
    huffman_table*,
    uint8_t* (&)[max_comp_count],
    int (&)[max_comp_count],
    qtable* (&)[max_comp_count],
    stack_allocator&,
    cudaStream_t,
    logger&);

template jpeggpu_status jpeggpu::decode_block<true>(
    const jpeg_stream&,
    const uint8_t*,
    int16_t*,
    int*,
    const struct scan&,
    huffman_table*,
    uint8_t* (&)[max_comp_count],
    int (&)[max_comp_count],
    qtable* (&)[max_comp_count],
    stack_allocator&,
    cudaStream_t,
    logger&);
