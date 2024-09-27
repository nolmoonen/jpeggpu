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

    __shared__ uint32_t storage[block_size];
    // TODO making this explicit is better
    // this is safe because the scan allocation is a multiple of 256
    const int num_words_in_scan = ceiling_div(t.scan_size, 4u);
    if (threadIdx.x < num_words_in_scan) {
        storage[threadIdx.x] = reinterpret_cast<const uint32_t*>(t.scan_destuffed)[threadIdx.x];
    }
    assert(num_words_in_scan <= block_size); // FIXME remove once iterative load happens
    __syncthreads();

    if (threadIdx.x != 0) return;

    int16_t* out       = nullptr;
    int num_data_units = 0;
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

    int word_idx      = 0;
    int bits_in_cache = 0; // can be zero if data has run out
    uint64_t cache    = 0;
    if (num_words_in_scan > 0) cache = uint64_t{storage[word_idx++]} << 32;
    if (num_words_in_scan > 1) cache |= storage[word_idx++];

    for (int i = 0; i < num_data_units; ++i) {
        const int se_exclusive = t.se + 1;
        for (int j = t.ss; j < se_exclusive; ++j) {
            if (bits_in_cache < 32) {
                const uint32_t word = word_idx++ < num_words_in_scan ? storage[word_idx++] : 0;
                cache |= uint64_t{word} << (32 - bits_in_cache);
                bits_in_cache += 32;
            }

            // ------------- at most 16 bits
            int i;
            int32_t code;
            huffman_table::entry entry;
            for (i = 0; i < 16; ++i) {
                code = u32_select_bits(static_cast<uint32_t>(cache >> 32), i + 1);
                const bool is_last_iter = i == 15;
                entry                   = table.entries[i];
                if (code <= entry.maxcode || is_last_iter) {
                    break;
                }
            }
            assert(1 <= i + 1 && i + 1 <= 16);
            // termination condition: 1 <= i + 1 <= 16, i + 1 is number of bits
            const int num_bits_consumed = i + 1;

            assert(num_bits_consumed <= bits_in_cache); // TODO make function
            bits_in_cache -= num_bits_consumed;
            cache <<= num_bits_consumed;

            const int idx = entry.valptr + (code - entry.mincode);
            assert(0 <= idx && idx < 256);
            const uint8_t s    = table.huffval[idx];
            const int run      = s >> 4;
            const int category = s & 0xf;

            const int positive = 1 << t.al;
            const int negative = ((unsigned)-1) << t.al; // TODO undefined behavior?

            if (category == 0 && run != 15) {
                // ---------------------- at most 14 bits
                // End of Band
                // read the next `run` bits (at most 14), contains #eob blocks
                const uint32_t eob_field = u32_select_bits(static_cast<uint32_t>(cache >> 32), run);

                assert(run <= bits_in_cache); // TODO make function
                bits_in_cache -= run;
                cache <<= run;

                // -1 to exclude the current block
                const int num_eob_blocks = eob_field + (uint32_t{1} << run) - 1;
                i += num_eob_blocks;

                // skip through all remaining
                for (; j < se_exclusive; ++j) {
                    const int coef_idx    = 64 * i + j;
                    const bool is_nonzero = out[coef_idx];
                    if (is_nonzero) {
                        const int code = u32_select_bits(static_cast<uint32_t>(cache >> 32), 1);

                        assert(1 <= bits_in_cache); // TODO make function
                        bits_in_cache -= 1;
                        cache <<= 1;

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
                continue;
            }

            int coeff;
            if (category != 0) { // if not taking this branch, run == 15 aka ZRL
                assert(category == 1);
                const int code = u32_select_bits(static_cast<uint32_t>(cache >> 32), 1);

                assert(1 <= bits_in_cache); // TODO make function
                bits_in_cache -= 1;
                cache <<= 1;

                coeff = code ? positive : negative;
            }

            // -------------------- at most 62 bits
            int num_zeroes = run + 1;
            for (; j < se_exclusive; ++j) {
                const int coef_idx    = 64 * i + j;
                const bool is_nonzero = out[coef_idx];
                if (is_nonzero) {
                    const int code = u32_select_bits(static_cast<uint32_t>(cache >> 32), 1);

                    assert(1 <= bits_in_cache); // TODO make function
                    bits_in_cache -= 1;
                    cache <<= 1;

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

            if (category != 0) {
                const int coef_idx = 64 * i + j;
                out[coef_idx]      = coeff;
            }
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
            const scan& scan                = info.scans[scan_pass.scan_indices[i]];
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
