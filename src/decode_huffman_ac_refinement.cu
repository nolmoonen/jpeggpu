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

#include <cub/device/device_scan.cuh>

#include <limits>

using namespace jpeggpu;

namespace {

struct compressed_symbol {
    uint8_t run_length;
    int16_t coeff;
    static constexpr int16_t eob = std::numeric_limits<int16_t>::max();
};

__global__ void map_nonzeroes(int num_data_units, const int16_t* out, int* offset)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_data_units) return;

    int num_symbols = 0;
    int run_length  = 0;
    for (int i = 0; i < 64; ++i) {
        const int16_t coefficient = out[64 * tid + i];
        if (coefficient == 0) {
            ++run_length;
        } else {
            ++num_symbols;
            run_length = 0;
        }
    }
    if (run_length > 0) {
        ++num_symbols;
    }
    offset[tid] = num_symbols;
}

// have to compress all ac since we do not know ss or se
__global__ void compress(
    int num_data_units, const int16_t* out, const int* offset, compressed_symbol* compressed)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_data_units) return;

    const int thread_offset = offset[tid];
    int num_symbols         = 0;
    int run_length          = 0;
    for (int i = 0; i < 64; ++i) {
        const int16_t coefficient = out[64 * tid + i];
        if (coefficient == 0) {
            ++run_length;
        } else {
            compressed_symbol csymbol;
            csymbol.run_length                      = run_length;
            csymbol.coeff                           = coefficient;
            compressed[thread_offset + num_symbols] = csymbol;
            ++num_symbols;
            run_length = 0;
        }
    }
    if (run_length > 0) {
        compressed_symbol csymbol;
        csymbol.run_length                      = 0;
        csymbol.coeff                           = compressed_symbol::eob;
        compressed[thread_offset + num_symbols] = csymbol;
    }
}

struct const_state {
    huffman_table* huff_tables;
};

struct scan_param {
    uint8_t* scan_destuffed;
    segment* segments;
    int* segment_indices;
    int huff_idx;
    int comp_idx;
    int ss;
    int se;
    int al;
    int scan_size; // scan.end - scan.begin
    int16_t* out;
    compressed_symbol* out_compressed;
    int num_data_units;
};

template <int block_size>
__global__ void ac_refine(scan_param* scan_params, const_state cstate)
{
    scan_param scan_param = scan_params[blockIdx.x];

    __shared__ huffman_table table;
    load_huffman_table<block_size>(cstate.huff_tables[scan_param.huff_idx], table);
    __syncthreads();
    ulonglong4 cache;

    reader_state_thread_cache<1> rstate;
    rstate.data      = scan_param.scan_destuffed;
    rstate.data_end  = scan_param.scan_destuffed + scan_param.scan_size;
    rstate.cache     = &cache;
    rstate.cache_idx = 0;

    if (threadIdx.x == 0 && sizeof(uint4) <= rstate.data_end - rstate.data) {
        global_load_uint4(rstate, &(reinterpret_cast<uint4*>(rstate.cache)[0]));
    }
    if (threadIdx.x == 0 && sizeof(uint4) <= rstate.data_end - rstate.data) {
        global_load_uint4(rstate, &(reinterpret_cast<uint4*>(rstate.cache)[1]));
    }

    if (threadIdx.x != 0) return;

    __shared__ int order_natural_shared[64];
    for (int w = 0; w < 64; ++w) {
        order_natural_shared[w] = order_natural[w];
    }

    __shared__ int16_t block[64];

    // int idx                = 0;
    int num_blocks_to_skip = 0;
    for (int a = 0; a < scan_param.num_data_units; ++a) {
        // FIXME does compressing this into zero/nonzero make sense?
        // load from global into shared
        constexpr int num_sectors = 64 * sizeof(int16_t) / sizeof(uint4);
        for (int w = 0; w < num_sectors; ++w) {
            reinterpret_cast<uint4*>(block)[w] =
                reinterpret_cast<uint4*>(scan_param.out + a * 64)[w];
        }

        // FIXME this works because we always have a zero as DC and ss=1,se=63
        //   generalize it!
        // compressed_symbol symbol = scan_param.out_compressed[idx++];
        // symbol.run_length--;

        const int se_exclusive = scan_param.se + 1;
        int j                  = scan_param.ss;
        const int positive     = 1 << scan_param.al;
        const int negative     = ((unsigned)-1)
                             << scan_param.al; // TODO negative shift undefined behavior?

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

                    // const compressed_symbol = scan_param.out_compressed[idx++];
                    // assert();

                    const int coef_idx    = order_natural_shared[j];
                    const bool is_nonzero = block[coef_idx];
                    if (is_nonzero) {
                        const int code = u32_select_bits(data, 1);
                        discard_bits(rstate, 1);
                        data <<= 1;
                        ++tmp;

                        if (code) {
                            if ((block[coef_idx] & positive) == 0) {
                                if (block[coef_idx] >= 0) {
                                    block[coef_idx] += positive;
                                } else {
                                    block[coef_idx] += negative;
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
                    const int coef_idx = order_natural_shared[j];
                    block[coef_idx]    = coeff;
                }
                assert(num_blocks_to_skip == 0);
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

                const int coef_idx    = order_natural_shared[j];
                const bool is_nonzero = block[coef_idx];
                if (is_nonzero) {
                    const int code = u32_select_bits(data, 1);
                    discard_bits(rstate, 1);
                    data <<= 1;
                    ++tmp;

                    if (code) {
                        if ((block[coef_idx] & positive) == 0) {
                            if (block[coef_idx] >= 0) {
                                block[coef_idx] += positive;
                            } else {
                                block[coef_idx] += negative;
                            }
                        }
                    }
                }
            }
            assert(tmp <= 32);
            --num_blocks_to_skip;
        }

        // store from shared into global
        for (int w = 0; w < num_sectors; ++w) {
            reinterpret_cast<uint4*>(scan_param.out + a * 64)[w] =
                reinterpret_cast<uint4*>(block)[w];
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

    // compress

    compressed_symbol* d_out_compressed[max_comp_count] = {};
    for (int c = 0; c < info.num_components; ++c) {
        const int num_data_units = info.components[c].size.x * info.components[c].size.y;
        int* d_offset            = nullptr;
        JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(&d_offset, num_data_units * sizeof(int)));
        constexpr int block_size = 256;
        const int num_blocks = ceiling_div(num_data_units, static_cast<unsigned int>(block_size));

        if (do_it) {
            map_nonzeroes<<<num_blocks, block_size, 0, stream>>>(
                num_data_units, d_out[c], d_offset);
            JPEGGPU_CHECK_CUDA(cudaGetLastError());
        }

        { // scan `d_offset`
            void* d_tmp_storage     = nullptr;
            size_t tmp_storage_size = 0;
            JPEGGPU_CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
                d_tmp_storage, tmp_storage_size, d_offset, d_offset, num_data_units, stream));
            JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(&d_tmp_storage, tmp_storage_size));
            if (do_it) {
                JPEGGPU_CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
                    d_tmp_storage, tmp_storage_size, d_offset, d_offset, num_data_units, stream));
            }
        }

        if (do_it && is_debug) {
            int num_compressed = 0;
            JPEGGPU_CHECK_CUDA(cudaMemcpyAsync(
                &num_compressed,
                &(d_offset[num_data_units - 1]),
                sizeof(int),
                cudaMemcpyDeviceToHost,
                stream));
            const ssize_t num_uncompressed_bytes = 64 * num_data_units * sizeof(int16_t);
            const ssize_t num_compressed_bytes   = num_compressed * sizeof(compressed_symbol);
            logger.log(
                "%ld uncompressed, %ld compressed (%f%)\n",
                num_uncompressed_bytes,
                num_compressed_bytes,
                (num_compressed_bytes - num_uncompressed_bytes) / (.01 * num_uncompressed_bytes));
        }

        // allocate the maximum amount
        const int num_coefficients = 64 * num_data_units;
        JPEGGPU_CHECK_STAT(allocator.reserve<do_it>(
            &(d_out_compressed[c]), num_coefficients * sizeof(compressed_symbol)));

        if (do_it) {
            compress<<<num_blocks, block_size, 0, stream>>>(
                num_data_units, d_out[c], d_offset, d_out_compressed[c]);
            JPEGGPU_CHECK_CUDA(cudaGetLastError());
        }
    }

    // decode

    scan_param* d_scan_params;
    JPEGGPU_CHECK_STAT(
        allocator.reserve<do_it>(&d_scan_params, scan_pass.num_scans * sizeof(scan_param)));

    if (do_it) {
        std::vector<scan_param> h_scan_params;
        JPEGGPU_CHECK_STAT(nothrow_resize(h_scan_params, scan_pass.num_scans));
        for (int i = 0; i < scan_pass.num_scans; ++i) {
            const scan& scan = info.scans[scan_pass.scan_indices[i]];
            assert(scan.num_components == 1);
            const scan_component& scan_comp = scan.scan_components[0];
            const int comp_idx              = scan_comp.component_idx;
            scan_param& t                   = h_scan_params[i];
            t.scan_destuffed                = d_scan_destuffed[i];
            t.segments                      = d_segments[i];
            t.segment_indices               = d_segment_indices[i];
            t.huff_idx                      = scan_comp.ac_idx;
            t.comp_idx                      = comp_idx;
            t.ss                            = scan.spectral_start;
            t.se                            = scan.spectral_end;
            t.al                            = scan.successive_approx_lo;
            // TODO what are the implications of this? see note in decode_huffman.cu
            t.scan_size      = scan.end - scan.begin;
            t.out            = d_out[comp_idx];
            t.out_compressed = d_out_compressed[comp_idx];
            t.num_data_units =
                info.components[comp_idx].size.x * info.components[comp_idx].size.y / 64;
        }

        JPEGGPU_CHECK_CUDA(cudaMemcpyAsync( // FIXME remove copy
            d_scan_params,
            h_scan_params.data(),
            scan_pass.num_scans * sizeof(scan_param),
            cudaMemcpyHostToDevice,
            stream));

        const const_state cstate = {d_huff_tables};

        constexpr int block_size = 32;
        ac_refine<block_size>
            <<<scan_pass.num_scans, block_size, 0, stream>>>(d_scan_params, cstate);
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
