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
#include "logger.hpp"
#include "reader.hpp"

using namespace jpeggpu;

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
    // create a AoS on device, one for each scan
    // launch one kernel with a thread block for each scan

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
