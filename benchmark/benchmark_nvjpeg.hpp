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

#ifndef JPEGGPU_BENCHMARK_BENCHMARK_NVJPEG_HPP_
#define JPEGGPU_BENCHMARK_BENCHMARK_NVJPEG_HPP_

#include "benchmark_common.hpp"

#include <nvjpeg.h>

#include <cuda_runtime.h>

#include <chrono>
#include <condition_variable>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdint.h>
#include <thread>
#include <vector>

#define CHECK_NVJPEG(call)                                                                     \
    do {                                                                                       \
        nvjpegStatus_t stat = call;                                                            \
        if (stat != NVJPEG_STATUS_SUCCESS) {                                                   \
            std::cerr << "nvJPEG error \"" << static_cast<int>(stat) << "\" at: " __FILE__ ":" \
                      << __LINE__ << "\n";                                                     \
            std::exit(EXIT_FAILURE);                                                           \
        }                                                                                      \
    } while (0)

int device_malloc(void*, void** ptr, size_t size, cudaStream_t)
{
    CHECK_CUDA(cudaMalloc(ptr, size));
    return 0;
}

int device_free(void*, void* ptr, size_t, cudaStream_t)
{
    CHECK_CUDA(cudaFree(ptr));
    return 0;
}

int pinned_malloc(void*, void** ptr, size_t size, cudaStream_t)
{
    CHECK_CUDA(cudaMallocHost(ptr, size));
    return 0;
}

int pinned_free(void*, void* ptr, size_t, cudaStream_t)
{
    CHECK_CUDA(cudaFreeHost(ptr));
    return 0;
}

struct nvjpeg_state {
    void startup()
    {
        device_allocator = {&device_malloc, &device_free, nullptr};
        pinned_allocator = {&pinned_malloc, &pinned_free, nullptr};

        const nvjpegBackend_t backend = NVJPEG_BACKEND_GPU_HYBRID;
        const int flags               = 0;
        CHECK_NVJPEG(nvjpegCreateEx(backend, nullptr, nullptr, flags, &nvjpeg_handle));
        CHECK_NVJPEG(nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state));
        CHECK_NVJPEG(nvjpegDecoderCreate(nvjpeg_handle, backend, &nvjpeg_decoder));
        CHECK_NVJPEG(
            nvjpegDecoderStateCreate(nvjpeg_handle, nvjpeg_decoder, &nvjpeg_decoupled_state));
        CHECK_NVJPEG(nvjpegBufferDeviceCreateV2(nvjpeg_handle, &device_allocator, &device_buffer));
        CHECK_NVJPEG(nvjpegBufferPinnedCreateV2(nvjpeg_handle, &pinned_allocator, &pinned_buffer));
        CHECK_NVJPEG(nvjpegJpegStreamCreate(nvjpeg_handle, &jpeg_stream));
        CHECK_NVJPEG(nvjpegDecodeParamsCreate(nvjpeg_handle, &nvjpeg_decode_params));

        CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(nvjpeg_decoupled_state, device_buffer));
        CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(nvjpeg_decoupled_state, pinned_buffer));
    }

    void cleanup()
    {
        CHECK_NVJPEG(nvjpegDecodeParamsDestroy(nvjpeg_decode_params));
        CHECK_NVJPEG(nvjpegJpegStreamDestroy(jpeg_stream));
        CHECK_NVJPEG(nvjpegBufferPinnedDestroy(pinned_buffer));
        CHECK_NVJPEG(nvjpegBufferDeviceDestroy(device_buffer));
        CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_decoupled_state));
        CHECK_NVJPEG(nvjpegDecoderDestroy(nvjpeg_decoder));
        CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_state));
        CHECK_NVJPEG(nvjpegDestroy(nvjpeg_handle));
    }

    nvjpegDevAllocatorV2_t device_allocator;
    nvjpegPinnedAllocatorV2_t pinned_allocator;

    nvjpegHandle_t nvjpeg_handle;
    nvjpegJpegState_t nvjpeg_state;
    nvjpegJpegDecoder_t nvjpeg_decoder;
    nvjpegJpegState_t nvjpeg_decoupled_state;

    nvjpegBufferDevice_t device_buffer;
    nvjpegBufferPinned_t pinned_buffer;
    nvjpegJpegStream_t jpeg_stream;
    nvjpegDecodeParams_t nvjpeg_decode_params;
};

void bench_nvjpeg(const uint8_t* file_data, size_t file_size)
{
    cudaStream_t stream = 0;

    nvjpeg_state state;
    state.startup();

    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    int channels;
    nvjpegChromaSubsampling_t subsampling;
    CHECK_NVJPEG(nvjpegGetImageInfo(
        state.nvjpeg_handle,
        reinterpret_cast<const unsigned char*>(file_data),
        file_size,
        &channels,
        &subsampling,
        widths,
        heights));

    nvjpegImage_t d_img;
    for (int c = 0; c < channels; ++c) {
        CHECK_CUDA(cudaMalloc(&d_img.channel[c], widths[c] * heights[c]));
        d_img.pitch[c] = widths[c];
    }

    CHECK_NVJPEG(
        nvjpegDecodeParamsSetOutputFormat(state.nvjpeg_decode_params, NVJPEG_OUTPUT_UNCHANGED));

    const auto run_iter = [&]() {
        const int save_metadata = 0;
        const int save_stream   = 0;
        CHECK_NVJPEG(nvjpegJpegStreamParse(
            state.nvjpeg_handle,
            reinterpret_cast<const unsigned char*>(file_data),
            file_size,
            save_metadata,
            save_stream,
            state.jpeg_stream));

        CHECK_NVJPEG(nvjpegDecodeJpegHost(
            state.nvjpeg_handle,
            state.nvjpeg_decoder,
            state.nvjpeg_decoupled_state,
            state.nvjpeg_decode_params,
            state.jpeg_stream));

        CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(
            state.nvjpeg_handle,
            state.nvjpeg_decoder,
            state.nvjpeg_decoupled_state,
            state.jpeg_stream,
            stream));

        CHECK_NVJPEG(nvjpegDecodeJpegDevice(
            state.nvjpeg_handle,
            state.nvjpeg_decoder,
            state.nvjpeg_decoupled_state,
            &d_img,
            stream));

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

    for (int c = 0; c < channels; ++c) {
        CHECK_CUDA(cudaFree(d_img.channel[c]));
    }

    state.cleanup();

    printf("  nvJPEG");
    print_measurement(throughput, avg_latency, max_latency);
}

#endif // JPEGGPU_BENCHMARK_BENCHMARK_NVJPEG_HPP_
