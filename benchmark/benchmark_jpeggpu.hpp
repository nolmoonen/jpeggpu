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

#ifndef JPEGGPU_BENCHMARK_BENCHMARK_JPEGGPU_HPP_
#define JPEGGPU_BENCHMARK_BENCHMARK_JPEGGPU_HPP_

#include "benchmark_common.hpp"

#include <jpeggpu/jpeggpu.h>

#include <cuda_runtime.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdint.h>
#include <vector>

#define CHECK_JPEGGPU(call)                                                    \
    do {                                                                       \
        jpeggpu_status stat = call;                                            \
        if (stat != JPEGGPU_SUCCESS) {                                         \
            std::cerr << "jpeggpu error \"" << jpeggpu_get_status_string(stat) \
                      << "\" at: " __FILE__ ":" << __LINE__ << "\n";           \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

void bench_jpeggpu(const uint8_t* file_data, size_t file_size)
{
    cudaStream_t stream = 0;

    jpeggpu_decoder_t decoder;
    CHECK_JPEGGPU(jpeggpu_decoder_startup(&decoder));

    jpeggpu_img_info img_info;
    CHECK_JPEGGPU(jpeggpu_decoder_parse_header(decoder, &img_info, file_data, file_size));

    jpeggpu_img d_img;
    for (int c = 0; c < img_info.num_components; ++c) {
        CHECK_CUDA(cudaMalloc(&d_img.image[c], img_info.sizes_x[c] * img_info.sizes_y[c]));
        d_img.pitch[c] = img_info.sizes_x[c];
    }

    void* d_tmp     = nullptr;
    size_t tmp_size = 0;

    const auto run_iter = [&]() {
        CHECK_JPEGGPU(jpeggpu_decoder_parse_header(decoder, &img_info, file_data, file_size));

        size_t this_tmp_size = 0;
        CHECK_JPEGGPU(jpeggpu_decoder_get_buffer_size(decoder, &this_tmp_size));

        if (this_tmp_size > tmp_size) {
            if (d_tmp) {
                CHECK_CUDA(cudaFree(d_tmp));
            }
            d_tmp = nullptr;
            CHECK_CUDA(cudaMalloc(&d_tmp, this_tmp_size));
            tmp_size = this_tmp_size;
        }

        CHECK_JPEGGPU(jpeggpu_decoder_transfer(decoder, d_tmp, tmp_size, stream));

        CHECK_JPEGGPU(jpeggpu_decoder_decode(decoder, &d_img, d_tmp, this_tmp_size, stream));

        CHECK_CUDA(cudaStreamSynchronize(stream));
    };

    run_iter(); // warmup; force allocation

    double sum_latency{};
    double max_latency{std::numeric_limits<double>::lowest()};
    for (int i = 0; i < num_iter; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        run_iter();
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double elapsed_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        sum_latency += elapsed_us;
        max_latency = std::max(max_latency, elapsed_us);
    }
    const double avg_latency = sum_latency / num_iter / us_in_ms;
    max_latency /= us_in_ms;

    const double total_seconds = sum_latency / us_in_s;
    const double throughput    = num_iter / total_seconds;

    if (d_tmp) CHECK_CUDA(cudaFree(d_tmp));
    for (int c = 0; c < img_info.num_components; ++c) {
        CHECK_CUDA(cudaFree(d_img.image[c]));
    }

    CHECK_JPEGGPU(jpeggpu_decoder_cleanup(decoder));

    printf(" jpeggpu");
    print_measurement(throughput, avg_latency, max_latency);
}

#endif // JPEGGPU_BENCHMARK_BENCHMARK_JPEGGPU_HPP_
