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

#include "decode_huffman_ac_refinement.hpp"
#include "decode_huffman_common.hpp"
#include "decode_huffman_reader.hpp"
#include "logger.hpp"
#include "reader.hpp"

using namespace jpeggpu;

namespace {

struct const_state {
    huffman_table* huff_tables;
    int16_t* out_0;
    int16_t* out_1;
    int16_t* out_2;
    int16_t* out_3;
    int num_data_units_0;
    int num_data_units_1;
    int num_data_units_2;
    int num_data_units_3;
};

struct tmp {
    uint8_t* scan_destuffed;
    segment* segments;
    int* segment_indices;
    int huff_idx;
    int comp_idx;
    int ss;
    int se;
    int al;
    int scan_size; // scan.end - scan.begin
};

template <int block_size>
__global__ void tmp2(tmp* tmps, const_state cstate)
{
    tmp t = tmps[blockIdx.x];

    __shared__ huffman_table table;
    load_huffman_table<block_size>(cstate.huff_tables[t.huff_idx], table);
    __syncthreads();
    if (threadIdx.x != 0) return;

    ulonglong4 cache;

    reader_state_thread_cache<1> rstate;
    rstate.data      = t.scan_destuffed;
    rstate.data_end  = t.scan_destuffed + t.scan_size;
    rstate.cache     = &cache;
    rstate.cache_idx = 0;

    if (sizeof(uint4) <= rstate.data_end - rstate.data) {
        global_load_uint4(rstate, &(reinterpret_cast<uint4*>(rstate.cache)[0]));
    }
    if (sizeof(uint4) <= rstate.data_end - rstate.data) {
        global_load_uint4(rstate, &(reinterpret_cast<uint4*>(rstate.cache)[1]));
    }

    int16_t* out       = nullptr;
    int num_data_units = 0;
    assert(0 <= t.comp_idx && t.comp_idx < 4);
    switch (t.comp_idx) {
    case 0:
        out            = cstate.out_0;
        num_data_units = cstate.num_data_units_0;
        break;
    case 1:
        out            = cstate.out_1;
        num_data_units = cstate.num_data_units_1;
        break;
    case 2:
        out            = cstate.out_2;
        num_data_units = cstate.num_data_units_2;
        break;
    case 3:
        out            = cstate.out_3;
        num_data_units = cstate.num_data_units_3;
        break;
    }

    int num_blocks_to_skip = 0;
    for (int i = 0; i < num_data_units; ++i) {
        const int se_exclusive = t.se + 1;
        int j                  = t.ss;
        const int positive     = 1 << t.al;
        const int negative     = ((unsigned)-1) << t.al; // TODO negative shift undefined behavior?

        if (num_blocks_to_skip == 0) {
            for (; j < se_exclusive; ++j) {
                uint32_t data = load_32_bits(rstate);

                // ------------- at most 16 bits
                int k;
                int32_t code;
                huffman_table::entry entry;
                for (k = 0; k < 16; ++k) {
                    code                    = u32_select_bits(data, k + 1);
                    const bool is_last_iter = k == 15;
                    entry                   = table.entries[k];
                    if (code <= entry.maxcode || is_last_iter) {
                        break;
                    }
                }
                assert(1 <= k + 1 && k + 1 <= 16);
                // termination condition: 1 <= k + 1 <= 16, k + 1 is number of bits
                const int num_bits_consumed = k + 1;
                discard_bits(rstate, num_bits_consumed);
                data <<= num_bits_consumed;
                // --------------- at least 16 bits remaining

                const int idx = entry.valptr + (code - entry.mincode);
                assert(0 <= idx && idx < 256);
                const uint8_t s    = table.huffval[idx];
                const int run      = s >> 4;
                const int category = s & 0xf;

                if (category == 0 && run != 15) {
                    // ---------------------- at most 14 bits
                    // End of Band
                    // read the next `run` bits (at most 14), contains #eob blocks
                    const uint32_t eob_field = u32_select_bits(data, run);
                    discard_bits(rstate, run);
                    data <<= num_bits_consumed;

                    const int num_eob_blocks = eob_field + (uint32_t{1} << run);
                    num_blocks_to_skip += num_eob_blocks;
                    // ---------------------- at least 2 bits remaining
                    break;
                }

                int coeff;
                if (category != 0) { // if not taking this branch, run == 15 aka ZRL
                    assert(category == 1);
                    const int code = u32_select_bits(data, 1);
                    discard_bits(rstate, 1);
                    data <<= 1;

                    coeff = code ? positive : negative;
                }

                data = load_32_bits(rstate);
                // FIXME properly implement data loading
                int tmp = 0;

                // -------------------- at most 62 bits
                int num_zeroes = run + 1;
                for (; j < se_exclusive; ++j) {
                    if (tmp >= 30) // FIXME this is ugly
                    {
                        tmp  = 0;
                        data = load_32_bits(rstate);
                    }

                    const int coef_idx    = 64 * i + order_natural[j];
                    const bool is_nonzero = out[coef_idx];
                    if (is_nonzero) {
                        const int code = u32_select_bits(data, 1);
                        discard_bits(rstate, 1);
                        data <<= 1;
                        ++tmp;

                        if (code) {
                            if ((out[coef_idx] & positive) == 0) {
                                if (out[coef_idx] >= 0) {
                                    out[coef_idx] += positive;
                                } else {
                                    out[coef_idx] += negative;
                                }
                            }
                        }
                    } else {
                        --num_zeroes;
                        if (num_zeroes == 0) break;
                    }
                }
                assert(tmp <= 31); // may need one bit for below

                if (category != 0) {
                    assert(j <= 64);
                    const int coef_idx = 64 * i + order_natural[j];
                    out[coef_idx]      = coeff;
                }
            }
        }

        if (num_blocks_to_skip > 0) {
            // ---------------------- at most 64 bits?
            uint32_t data = load_32_bits(rstate);
            // FIXME properly implement data loading
            int tmp = 0;

            // skip through all remaining
            for (; j < se_exclusive; ++j) {
                if (tmp >= 31) // FIXME this is ugly
                {
                    tmp  = 0;
                    data = load_32_bits(rstate);
                }

                const int coef_idx    = 64 * i + order_natural[j];
                const bool is_nonzero = out[coef_idx];
                if (is_nonzero) {
                    const int code = u32_select_bits(data, 1);
                    discard_bits(rstate, 1);
                    data <<= 1;
                    ++tmp;

                    if (code) {
                        if ((out[coef_idx] & positive) == 0) {
                            if (out[coef_idx] >= 0) {
                                out[coef_idx] += positive;
                            } else {
                                out[coef_idx] += negative;
                            }
                        }
                    }
                }
            }
            assert(tmp <= 32);
            --num_blocks_to_skip;
        }
    }
}

} // namespace

template <bool do_it>
jpeggpu_status jpeggpu::decode_ac_refinement(
    const jpeg_stream& info,
    const ac_scan_pass& scan_pass,
    uint8_t* (&d_scan_destuffed)[ac_scan_pass::max_num_scans],
    const std::vector<segment*>& d_segments,
    int* (&d_segment_indices)[ac_scan_pass::max_num_scans],
    int16_t* (&d_out)[max_comp_count],
    huffman_table* d_huff_tables,
    stack_allocator& allocator,
    cudaStream_t stream,
    logger& logger)
{
    // FIXME deal with segments! probably can edit the loop to discard the remaining bits
    //   maybe destuffing code needs to be changed.

    tmp* d_tmps;
    JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(&d_tmps, scan_pass.num_scans * sizeof(tmp)));

    if (do_it) {
        std::vector<tmp> h_tmps;
        JPEGGPU_CHECK_STAT(nothrow_resize(h_tmps, scan_pass.num_scans));
        for (int i = 0; i < scan_pass.num_scans; ++i) {
            const scan& scan = info.scans[scan_pass.scan_indices[i]];
            assert(scan.num_components == 1);
            const scan_component& scan_comp = scan.scan_components[0];
            tmp& t                          = h_tmps[i];
            t.scan_destuffed                = d_scan_destuffed[i];
            t.segments                      = d_segments[i];
            t.segment_indices               = d_segment_indices[i];
            t.huff_idx                      = scan_comp.ac_idx;
            t.comp_idx                      = scan_comp.component_idx;
            t.ss                            = scan.spectral_start;
            t.se                            = scan.spectral_end;
            t.al                            = scan.successive_approx_lo;
            // TODO what are the implications of this? see note in decode_huffman.cu
            t.scan_size = scan.end - scan.begin;
        }

        JPEGGPU_CHECK_CUDA(cudaMemcpyAsync( // FIXME remove copy
            d_tmps,
            h_tmps.data(),
            scan_pass.num_scans * sizeof(tmp),
            cudaMemcpyHostToDevice,
            stream));

        // TODO remove assert once variable is checked
        assert((info.components[0].size.x * info.components[0].size.y) % 64 == 0);
        assert((info.components[1].size.x * info.components[1].size.y) % 64 == 0);
        assert((info.components[2].size.x * info.components[2].size.y) % 64 == 0);
        assert((info.components[3].size.x * info.components[3].size.y) % 64 == 0);
        const const_state cstate = {
            d_huff_tables,
            d_out[0],
            d_out[1],
            d_out[2],
            d_out[3],
            info.components[0].size.x * info.components[0].size.y / 64,
            info.components[1].size.x * info.components[1].size.y / 64,
            info.components[2].size.x * info.components[2].size.y / 64,
            info.components[3].size.x * info.components[3].size.y / 64};

        constexpr int block_size = 256;
        tmp2<block_size><<<scan_pass.num_scans, block_size, 0, stream>>>(d_tmps, cstate);
        JPEGGPU_CHECK_CUDA(cudaGetLastError());
    }

    return JPEGGPU_SUCCESS;
}

template jpeggpu_status jpeggpu::decode_ac_refinement<false>(
    const jpeg_stream&,
    const ac_scan_pass&,
    uint8_t* (&)[ac_scan_pass::max_num_scans],
    const std::vector<segment*>&,
    int* (&)[ac_scan_pass::max_num_scans],
    int16_t* (&)[max_comp_count],
    huffman_table*,
    stack_allocator&,
    cudaStream_t,
    logger&);

template jpeggpu_status jpeggpu::decode_ac_refinement<true>(
    const jpeg_stream&,
    const ac_scan_pass&,
    uint8_t* (&)[ac_scan_pass::max_num_scans],
    const std::vector<segment*>&,
    int* (&)[ac_scan_pass::max_num_scans],
    int16_t* (&)[max_comp_count],
    huffman_table*,
    stack_allocator&,
    cudaStream_t,
    logger&);