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

#define CHECK_JPEGGPU(call)                                                                        \
    do {                                                                                           \
        jpeggpu_status stat = call;                                                                \
        if (stat != JPEGGPU_SUCCESS) {                                                             \
            std::cerr << "jpeggpu error \"" << jpeggpu_get_status_string(stat)                     \
                      << "\" at: " __FILE__ ":" << __LINE__ << "\n";                               \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (0)

void bench_jpeggpu(const char* file_data, size_t file_size)
{
    cudaStream_t stream = 0;

    jpeggpu_decoder_t decoder;
    CHECK_JPEGGPU(jpeggpu_decoder_startup(&decoder));

    jpeggpu_img_info img_info;
    CHECK_JPEGGPU(jpeggpu_decoder_parse_header(
        decoder, &img_info, reinterpret_cast<const uint8_t*>(file_data), file_size));

    size_t image_bytes = 0;
    jpeggpu_img d_img;
    for (int c = 0; c < img_info.num_components; ++c) {
        const size_t plane_bytes = img_info.sizes_x[c] * img_info.sizes_y[c];
        CHECK_CUDA(cudaMalloc(&d_img.image[c], plane_bytes));
        d_img.pitch[c] = img_info.sizes_x[c];
        image_bytes += plane_bytes;
    }

    void* d_tmp     = nullptr;
    size_t tmp_size = 0;

    const auto run_iter = [&]() {
        CHECK_JPEGGPU(jpeggpu_decoder_parse_header(
            decoder, &img_info, reinterpret_cast<const uint8_t*>(file_data), file_size));

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

    run_iter();

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
    const double avg_latency = sum_latency / num_iter / 1e3;
    max_latency /= 1e3;

    const double total_seconds = sum_latency / 1e6;
    const double throughput    = num_iter / total_seconds;

    if (d_tmp) CHECK_CUDA(cudaFree(d_tmp));
    for (int c = 0; c < img_info.num_components; ++c) {
        CHECK_CUDA(cudaFree(d_img.image[c]));
    }

    CHECK_JPEGGPU(jpeggpu_decoder_cleanup(decoder));

    printf(
        "jpeggpu singlethread               %5.2f              %5.2f              %5.2f\n",
        throughput,
        avg_latency,
        max_latency);
}

#endif // JPEGGPU_BENCHMARK_BENCHMARK_JPEGGPU_HPP_
