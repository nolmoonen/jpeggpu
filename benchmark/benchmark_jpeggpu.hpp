#ifndef JPEGGPU_BENCHMARK_BENCHMARK_JPEGGPU_HPP_
#define JPEGGPU_BENCHMARK_BENCHMARK_JPEGGPU_HPP_

#include "benchmark_common.hpp"

#include <jpeggpu/jpeggpu.h>

#include <cuda_runtime.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdint.h>
#include <thread>
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

void bench_jpeggpu_st(const char* file_data, size_t file_size)
{
    cudaStream_t stream = nullptr;

    jpeggpu_decoder_t decoder;
    CHECK_JPEGGPU(jpeggpu_decoder_startup(&decoder));

    jpeggpu_img_info img_info;
    CHECK_JPEGGPU(jpeggpu_decoder_parse_header(
        decoder, &img_info, reinterpret_cast<const uint8_t*>(file_data), file_size));

    size_t image_bytes = 0;
    jpeggpu_img d_img;
    std::vector<char*> h_img(img_info.num_components);
    for (int c = 0; c < img_info.num_components; ++c) {
        const size_t plane_bytes = img_info.sizes_x[c] * img_info.sizes_y[c];
        CHECK_CUDA(cudaMalloc(&d_img.image[c], plane_bytes));
        d_img.pitch[c] = img_info.sizes_x[c];
        image_bytes += plane_bytes;
        CHECK_CUDA(cudaMallocHost(&h_img[c], plane_bytes));
    }
    d_img.color_fmt   = JPEGGPU_OUT_NO_CONVERSION;
    d_img.subsampling = img_info.subsampling;

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

        for (int c = 0; c < img_info.num_components; ++c) {
            const size_t plane_bytes = img_info.sizes_x[c] * img_info.sizes_y[c];
            CHECK_CUDA(cudaMemcpyAsync(
                h_img[c], d_img.image[c], plane_bytes, cudaMemcpyDeviceToHost, stream));
        }

        CHECK_CUDA(cudaStreamSynchronize(stream));
    };

    run_iter();

    double sum_latency{};
    double max_latency{};
    for (int i = 0; i < num_iter; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        run_iter();
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        sum_latency += elapsed_ms;
        max_latency = std::max(max_latency, elapsed_ms);
    }
    const double avg_latency = sum_latency / num_iter;

    const double total_seconds = sum_latency / 1e3;
    const double throughput    = num_iter / total_seconds;

    if (d_tmp) CHECK_CUDA(cudaFree(d_tmp));
    for (int c = 0; c < img_info.num_components; ++c) {
        CHECK_CUDA(cudaFreeHost(h_img[c]));
        CHECK_CUDA(cudaFree(d_img.image[c]));
    }

    CHECK_JPEGGPU(jpeggpu_decoder_cleanup(decoder));

    std::cout << "jpeggpu singlethreaded, throughput: " << std::fixed << std::setw(5)
              << std::setprecision(2) << throughput << " image/s, avg latency: " << avg_latency
              << "ms, max latency: " << max_latency << "ms\n";
}

struct bench_jpeggpu_state {
    const char* file_data;
    size_t file_size;
    cudaStream_t stream_common;
    std::mutex mutex;
};

struct bench_jpeggpu_thread_state {
    cudaStream_t stream;
    cudaEvent_t event_h2d;
    cudaEvent_t event_d2h;
    jpeggpu_decoder_t decoder;
    int num_components;
    jpeggpu_img_info img_info;
    jpeggpu_img d_img;
    std::vector<char*> h_img;
    void* d_tmp;
    size_t tmp_size;
    double sum_latency; // in milliseconds
    double max_latency; // in milliseconds
};

void bench_jpeggpu_run_iter(
    bench_jpeggpu_state& bench_state, bench_jpeggpu_thread_state& bench_thread_state)
{
    CHECK_JPEGGPU(jpeggpu_decoder_parse_header(
        bench_thread_state.decoder,
        &bench_thread_state.img_info,
        reinterpret_cast<const uint8_t*>(bench_state.file_data),
        bench_state.file_size));

    size_t this_tmp_size = 0;
    CHECK_JPEGGPU(jpeggpu_decoder_get_buffer_size(bench_thread_state.decoder, &this_tmp_size));

    if (this_tmp_size > bench_thread_state.tmp_size) {
        if (bench_thread_state.d_tmp) {
            CHECK_CUDA(cudaFree(bench_thread_state.d_tmp));
        }
        bench_thread_state.d_tmp = nullptr;
        CHECK_CUDA(cudaMalloc(&bench_thread_state.d_tmp, this_tmp_size));
        bench_thread_state.tmp_size = this_tmp_size;
    }

    {
        std::lock_guard<std::mutex>(bench_state.mutex);
        CHECK_JPEGGPU(jpeggpu_decoder_transfer(
            bench_thread_state.decoder,
            bench_thread_state.d_tmp,
            bench_thread_state.tmp_size,
            bench_thread_state.stream));
        CHECK_CUDA(cudaEventRecord(bench_thread_state.event_h2d, bench_thread_state.stream));
        CHECK_CUDA(cudaStreamWaitEvent(bench_state.stream_common, bench_thread_state.event_h2d));
        CHECK_JPEGGPU(jpeggpu_decoder_decode(
            bench_thread_state.decoder,
            &bench_thread_state.d_img,
            bench_thread_state.d_tmp,
            this_tmp_size,
            bench_state.stream_common));
        CHECK_CUDA(cudaEventRecord(bench_thread_state.event_d2h, bench_state.stream_common));
        CHECK_CUDA(cudaStreamWaitEvent(bench_thread_state.stream, bench_thread_state.event_d2h));
        for (int c = 0; c < bench_thread_state.num_components; ++c) {
            const size_t plane_bytes =
                bench_thread_state.img_info.sizes_x[c] * bench_thread_state.img_info.sizes_y[c];
            CHECK_CUDA(cudaMemcpyAsync(
                bench_thread_state.h_img[c],
                bench_thread_state.d_img.image[c],
                plane_bytes,
                cudaMemcpyDeviceToHost,
                bench_thread_state.stream));
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(bench_thread_state.stream));
};

void bench_jpeggpu_mt_thread_startup(
    bench_jpeggpu_state& bench_state, bench_jpeggpu_thread_state& bench_thread_state)
{
    CHECK_CUDA(cudaStreamCreateWithFlags(&bench_thread_state.stream, cudaStreamNonBlocking));
    CHECK_CUDA(cudaEventCreate(&bench_thread_state.event_h2d));
    CHECK_CUDA(cudaEventCreate(&bench_thread_state.event_d2h));

    CHECK_JPEGGPU(jpeggpu_decoder_startup(&bench_thread_state.decoder));

    CHECK_JPEGGPU(jpeggpu_decoder_parse_header(
        bench_thread_state.decoder,
        &bench_thread_state.img_info,
        reinterpret_cast<const uint8_t*>(bench_state.file_data),
        bench_state.file_size));

    bench_thread_state.h_img.resize(bench_thread_state.img_info.num_components);
    size_t image_bytes = 0;
    for (int c = 0; c < bench_thread_state.img_info.num_components; ++c) {
        const size_t plane_bytes =
            bench_thread_state.img_info.sizes_x[c] * bench_thread_state.img_info.sizes_y[c];
        CHECK_CUDA(cudaMalloc(&bench_thread_state.d_img.image[c], plane_bytes));
        bench_thread_state.d_img.pitch[c] = bench_thread_state.img_info.sizes_x[c];
        image_bytes += plane_bytes;
        CHECK_CUDA(cudaMallocHost(&bench_thread_state.h_img[c], plane_bytes));
    }
    bench_thread_state.d_img.color_fmt   = JPEGGPU_OUT_NO_CONVERSION;
    bench_thread_state.d_img.subsampling = bench_thread_state.img_info.subsampling;

    bench_thread_state.d_tmp    = nullptr;
    bench_thread_state.tmp_size = 0;

    bench_jpeggpu_run_iter(bench_state, bench_thread_state);
}

void bench_jpeggpu_mt_thread_perform(
    bench_jpeggpu_state& bench_state, bench_jpeggpu_thread_state& bench_thread_state)
{
    bench_thread_state.sum_latency = 0.0;
    bench_thread_state.max_latency = -std::numeric_limits<double>::max();
    for (int i = 0; i < num_iter; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        bench_jpeggpu_run_iter(bench_state, bench_thread_state);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        bench_thread_state.sum_latency += elapsed_ms;
        bench_thread_state.max_latency = std::max(bench_thread_state.max_latency, elapsed_ms);
    }
}

void bench_jpeggpu_mt_thread_cleanup(
    bench_jpeggpu_state&, bench_jpeggpu_thread_state& bench_thread_state)
{
    if (bench_thread_state.d_tmp) CHECK_CUDA(cudaFree(bench_thread_state.d_tmp));
    for (int c = 0; c < bench_thread_state.num_components; ++c) {
        CHECK_CUDA(cudaFreeHost(bench_thread_state.h_img[c]));
        CHECK_CUDA(cudaFree(bench_thread_state.d_img.image[c]));
    }

    CHECK_JPEGGPU(jpeggpu_decoder_cleanup(bench_thread_state.decoder));

    CHECK_CUDA(cudaEventDestroy(bench_thread_state.event_d2h));
    CHECK_CUDA(cudaEventDestroy(bench_thread_state.event_h2d));
    CHECK_CUDA(cudaStreamDestroy(bench_thread_state.stream));
}

void bench_jpeggpu_mt(const char* file_data, size_t file_size)
{
    bench_jpeggpu_state bench_state;
    bench_state.file_data = file_data;
    bench_state.file_size = file_size;
    CHECK_CUDA(cudaStreamCreateWithFlags(&bench_state.stream_common, cudaStreamNonBlocking));

    bench_jpeggpu_thread_state bench_thread_state_0;
    bench_jpeggpu_thread_state bench_thread_state_1;

    bench_jpeggpu_mt_thread_startup(bench_state, bench_thread_state_0);
    bench_jpeggpu_mt_thread_startup(bench_state, bench_thread_state_1);

    const auto t0 = std::chrono::high_resolution_clock::now();
    std::thread worker_0(
        bench_jpeggpu_mt_thread_perform, std::ref(bench_state), std::ref(bench_thread_state_0));
    std::thread worker_1(
        bench_jpeggpu_mt_thread_perform, std::ref(bench_state), std::ref(bench_thread_state_1));

    worker_0.join();
    worker_1.join();
    const auto t1 = std::chrono::high_resolution_clock::now();

    bench_jpeggpu_mt_thread_cleanup(bench_state, bench_thread_state_0);
    bench_jpeggpu_mt_thread_cleanup(bench_state, bench_thread_state_1);

    CHECK_CUDA(cudaStreamDestroy(bench_state.stream_common));

    const double sum_latency = bench_thread_state_0.sum_latency + bench_thread_state_1.sum_latency;
    const double max_latency =
        std::max(bench_thread_state_0.max_latency, bench_thread_state_1.max_latency);
    const double avg_latency = sum_latency / (2 * num_iter);

    const double total_seconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1e3;
    const double throughput = (2 * num_iter) / total_seconds;

    std::cout << "jpeggpu multithreaded, throughput: " << std::fixed << std::setw(5)
              << std::setprecision(2) << throughput << " image/s, avg latency: " << avg_latency
              << "ms, max latency: " << max_latency << "ms\n";
}

void bench_jpeggpu(const char* file_data, size_t file_size)
{
    bench_jpeggpu_st(file_data, file_size);
    bench_jpeggpu_mt(file_data, file_size);
}

#endif // JPEGGPU_BENCHMARK_BENCHMARK_JPEGGPU_HPP_
